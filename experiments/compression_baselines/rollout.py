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

# The only nonlinear features: products of the state's velocity and gradient.
ADVECTION = [FEATURES.index("u_dTdx"), FEATURES.index("v_dTdy")]
from run import Cache, build_samples, subset


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
    chans = cache.meta["channels"]
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    fits = {}
    linear = a.rung == "lpc_linear"
    if a.rung != "persistence":
        for c in chans:
            fits[c] = fit_lpc(cache, c, a.forcing, a.n_train, rng, linear)
        print(
            f"[rollout] fitted {len(fits)} channels ({time.time() - t0:.0f}s)",
            flush=True,
        )

    # Loaded into memory, not memory-mapped, so later writes to the cache by
    # other jobs cannot change a rollout in progress.
    test = {c: np.asarray(cache.arr("test", c)) for c in chans}
    forc = (
        {f: np.asarray(cache.arr("test", f)) for f in FORCING_VARS} if a.forcing else {}
    )
    wet = {c: cache.masks[cache.meta["levels"][c]] for c in chans}
    band = cache.band
    n_test = next(iter(test.values())).shape[0]
    n_steps = n_test - 2 if a.max_steps <= 0 else min(a.max_steps, n_test - 2)
    prev = {c: np.asarray(test[c][0], dtype=np.float64) for c in chans}
    cur = {c: np.asarray(test[c][1], dtype=np.float64) for c in chans}
    H, W = cur[chans[0]].shape
    out = {c: np.full((n_steps, H, W), np.nan, np.float32) for c in chans}
    t0 = time.time()
    for s in range(n_steps):
        t = s + 1  # index of the latest input state in the test window
        nxt = {}
        for c in chans:
            if a.rung == "persistence":
                nxt[c] = cur[c]
                continue
            w_band, mean_c = fits[c]
            others, up, dn, un, vn = cache.neighbours(c)
            F = patch_features(
                cur[c],
                prev[c],
                [(cur[o], prev[o]) for o in others],
                (cur[up], prev[up]) if up else None,
                (cur[dn], prev[dn]) if dn else None,
                cur[un],
                cur[vn],
                mean_c,
                cache.inv_cos,
                forcing=[
                    (np.asarray(forc[f][t]), np.asarray(forc[f][t - 1]))
                    for f in FORCING_VARS
                ]
                if a.forcing
                else None,
            )
            if linear:
                F[..., ADVECTION] = 0.0
            tend = np.zeros((H, W))
            for b, wv in w_band.items():
                m = wet[c] & (band == b)
                tend[m] = F[m] @ wv
            nx = np.where(wet[c], cur[c] + tend, np.nan)
            if not np.isfinite(nx[wet[c]]).all():
                raise FloatingPointError(
                    f"rollout became non-finite at step {s} in {c}"
                )
            nxt[c] = nx
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
