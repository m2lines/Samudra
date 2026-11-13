# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Autoregressive rollout of a ladder rung, written like a Samudra eval.

    python rollout.py CACHE OUT_DIR [--rung lpc|persistence] [--forcing]

Fits the rung on the cache's training window, exactly as run.py does, then
starts from the two *true* states at the beginning of the test window and steps
all 77 channels forward together. Each new state is the old one plus the rung's
mean tendency, and is fed back as input: an autoregressive rollout, as
`samudra eval` produces. Forcing (with --forcing) is the true, prescribed
forcing at the input times, as Samudra receives it.

`persistence` repeats the last true state forever: the zero-skill reference.

Writes OUT_DIR/predictions.zarr (Zarr v2) in the layout of Samudra's eval output:
one (time, lat, lon) float32 variable per channel, NaN on land, times from the
third test step onward, so `python -m samudra.metrics` can score it unchanged.
Fails loudly if the rollout becomes non-finite.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from models import FEATURES, FORCING_VARS, Ladder, patch_features
from run import Cache, build_samples, subset

# The only nonlinear features: products of the state's velocity and gradient.
ADVECTION = [FEATURES.index("u_dTdx"), FEATURES.index("v_dTdy")]


def fit_lpc(
    cache: Cache, c: str, forcing: bool, n_train: int, rng, linear: bool = False
):
    A = cache.arr("train", c)
    mean_c = float(np.nanmean(np.asarray(A)[:, cache.masks[cache.meta["levels"][c]]]))
    tr = build_samples(cache, "train", c, mean_c, forcing)
    if len(tr.y) > n_train:
        tr = subset(tr, rng.choice(len(tr.y), n_train, replace=False))
    if linear:
        tr.F[:, ADVECTION] = 0.0  # a constant-zero column gets zero weight
    lad = Ladder(seed=int(rng.integers(1 << 31))).fit(tr)
    return lad.w, mean_c


class Roller:
    """Fit a rung, then step every channel forward together.

    prescribed_velocity: compute the advection features from the *true*
    velocities instead of the rollout's own. This removes the only quadratic
    feedback (rollout velocity x rollout gradient) and keeps everything else.
    It is a diagnostic, not a forecast: it uses future truth.
    """

    def __init__(
        self,
        cache: Cache,
        rung: str,
        forcing: bool,
        n_train: int = 400_000,
        seed: int = 0,
        prescribed_velocity: bool = False,
    ):
        self.cache, self.rung, self.forcing = cache, rung, forcing
        self.prescribed_velocity = prescribed_velocity
        self.linear = rung == "lpc_linear"
        self.chans = cache.meta["channels"]
        rng = np.random.default_rng(seed)
        self.fits = {}
        if rung != "persistence":
            for c in self.chans:
                self.fits[c] = fit_lpc(cache, c, forcing, n_train, rng, self.linear)
        # In memory, not memory-mapped, so later writes to the cache by other jobs
        # cannot change a rollout in progress.
        self.test = {c: np.asarray(cache.arr("test", c)) for c in self.chans}
        self.forc = (
            {f: np.asarray(cache.arr("test", f)) for f in FORCING_VARS}
            if forcing
            else {}
        )
        self.wet = {c: cache.masks[cache.meta["levels"][c]] for c in self.chans}
        self.n_test = next(iter(self.test.values())).shape[0]

    def initial(self) -> tuple[dict, dict]:
        prev = {c: self.test[c][0].astype(np.float64) for c in self.chans}
        cur = {c: self.test[c][1].astype(np.float64) for c in self.chans}
        return prev, cur

    def features(self, c: str, cur: dict, prev: dict, t: int) -> np.ndarray:
        others, up, dn, un, vn = self.cache.neighbours(c)
        if self.prescribed_velocity:
            u, v = (
                self.test[un][t].astype(np.float64),
                self.test[vn][t].astype(np.float64),
            )
        else:
            u, v = cur[un], cur[vn]
        F = patch_features(
            cur[c],
            prev[c],
            [(cur[o], prev[o]) for o in others],
            (cur[up], prev[up]) if up else None,
            (cur[dn], prev[dn]) if dn else None,
            u,
            v,
            self.fits[c][1],
            self.cache.inv_cos,
            forcing=[(self.forc[f][t], self.forc[f][t - 1]) for f in FORCING_VARS]
            if self.forcing
            else None,
        )
        if self.linear:
            F[..., ADVECTION] = 0.0
        return F

    def weights(self, c: str) -> np.ndarray:
        """(H, W, d) weights: each cell gets its latitude band's weight vector."""
        w_band = self.fits[c][0]
        return np.stack([w_band[b] for b in range(3)])[self.cache.band]

    def step(self, cur: dict, prev: dict, t: int, check: bool = True) -> dict:
        nxt = {}
        for c in self.chans:
            if self.rung == "persistence":
                nxt[c] = cur[c]
                continue
            F = self.features(c, cur, prev, t)
            tend = np.einsum("hwd,hwd->hw", F, self.weights(c))
            nx = np.where(self.wet[c], cur[c] + tend, np.nan)
            # Checked in float32, the precision the predictions are written in: a
            # state can grow exponentially for hundreds of steps while staying
            # finite in float64 yet overflow when saved.
            if check and not np.isfinite(nx[self.wet[c]].astype(np.float32)).all():
                raise FloatingPointError(f"rollout became non-finite at t={t} in {c}")
            nxt[c] = nx
        return nxt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument(
        "--rung",
        choices=["lpc", "lpc_linear", "persistence"],
        default="lpc",
        help="lpc_linear: lpc without the two advection (velocity x gradient) features",
    )
    ap.add_argument("--forcing", action="store_true")
    ap.add_argument("--n-train", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test only")
    a = ap.parse_args()

    import xarray as xr

    cache = Cache(a.cache)
    t0 = time.time()
    roller = Roller(cache, a.rung, a.forcing, a.n_train, a.seed)
    print(
        f"[rollout] fitted {len(roller.fits)} channels ({time.time() - t0:.0f}s)",
        flush=True,
    )
    chans = roller.chans
    n_steps = (
        roller.n_test - 2 if a.max_steps <= 0 else min(a.max_steps, roller.n_test - 2)
    )
    n_test = roller.n_test
    prev, cur = roller.initial()
    H, W = cur[chans[0]].shape
    out = {c: np.full((n_steps, H, W), np.nan, np.float32) for c in chans}
    t0 = time.time()
    for s in range(n_steps):
        nxt = roller.step(
            cur, prev, t=s + 1
        )  # t: latest input state in the test window
        for c in chans:
            out[c][s] = nxt[c]
        prev, cur = cur, nxt
        if s % 50 == 0:
            print(
                f"[rollout] step {s + 1}/{n_steps} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    # Coordinates as in Samudra's predictions.zarr: 1-D lat/lon of the Gaussian
    # grid and the store's own time objects for the target steps.
    src = xr.open_zarr(cache.meta["source"])
    stamps = cache.meta["splits"]["test"][2 : 2 + n_steps]
    tidx = [str(v) for v in src.time.values]
    times = src.time.values[[tidx.index(s) for s in stamps]]
    lat = cache.lat[:, 0]
    if not np.allclose(cache.lat, lat[:, None]):
        raise ValueError("rollout output assumes a rectilinear (Gaussian) grid")
    ds = xr.Dataset(
        {c: (("time", "lat", "lon"), out[c]) for c in chans},
        coords={"time": times, "lat": lat, "lon": src.x.values},
        attrs={
            "model_path": f"compression_baselines ladder rung={a.rung} forcing={a.forcing}",
            "source": cache.meta["source"],
        },
    )
    a.out.mkdir(parents=True, exist_ok=True)
    enc = {c: {"chunks": (25, H, W)} for c in chans}
    ds.to_zarr(a.out / "predictions.zarr", mode="w", encoding=enc)  # zarr<3: v2
    (a.out / "rollout_meta.json").write_text(
        json.dumps(
            {
                "rung": a.rung,
                "forcing": a.forcing,
                "steps": n_steps,
                "first_time": stamps[0],
                "last_time": stamps[-1],
                "complete": n_steps == n_test - 2,
                "seconds": time.time() - t0,
            },
            indent=1,
        )
    )
    print(f"[rollout] wrote {a.out / 'predictions.zarr'} ({n_steps} steps)")


if __name__ == "__main__":
    main()
