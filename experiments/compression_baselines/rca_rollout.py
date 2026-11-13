# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Root-cause analysis of the ladder rollout blow-up.

    python rca_rollout.py CACHE OUT [--steps 40] [--check-against OLD_PREDICTIONS_ZARR]

Uses rollout.Roller, so every step is exactly what rollout.py computes.

1. Find the seed: roll the full `lpc` rung (with forcing) forward without
   stopping, and find the earliest (step, channel, cell) whose error grows more
   than tenfold in one step while exceeding one bin of physical sense (|error| > 1
   in the channel's units).
2. Trace it: replay from the start and record, at the seed cell, the state, the
   truth, every feature and every feature's contribution (weight x feature) to the
   predicted change, step by step.
3. Extrapolation: compare the seed channel's advection features during the
   rollout with their training distribution.
4. Weights: sign and standardized size of the advection weights, all channels.
5. Counterfactual: the same rung with the advection features computed from the
   *true* velocities (rollout.Roller prescribed_velocity), for the whole test
   window. This removes the quadratic feedback and keeps everything else.

Writes OUT/rca.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from models import FEATURES, FORCING_FEATURES
from rollout import ADVECTION, Roller
from run import Cache, build_samples

NAMES = FEATURES[:-1] + FORCING_FEATURES + ["bias"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--check-against", default=None)
    a = ap.parse_args()
    cache = Cache(a.cache)
    roller = Roller(cache, "lpc", forcing=True)
    chans, test, wet = roller.chans, roller.test, roller.wet
    rep: dict = {"cache": str(a.cache)}

    # 0. The refactored stepping reproduces the earlier rollout.
    if a.check_against:
        import xarray as xr

        old = xr.open_zarr(a.check_against)
        prev, cur = roller.initial()
        diffs = []
        for s in range(min(5, old.sizes["time"])):
            nxt = roller.step(cur, prev, s + 1)
            diffs += [
                float(np.nanmax(np.abs(nxt[c].astype(np.float32) - old[c].values[s])))
                for c in chans
            ]
            prev, cur = cur, nxt
        rep["refactor_max_abs_diff_first_5_steps"] = max(diffs)

    # 1. Seed of the blow-up.
    prev, cur = roller.initial()
    err = {c: [] for c in chans}
    with np.errstate(all="ignore"):
        for s in range(a.steps):
            nxt = roller.step(cur, prev, s + 1, check=False)
            for c in chans:
                err[c].append(np.where(wet[c], np.abs(nxt[c] - test[c][s + 2]), np.nan))
            prev, cur = cur, nxt
    seed = None
    for s in range(1, a.steps):
        best = None
        for c in chans:
            e0, e1 = err[c][s - 1], err[c][s]
            with np.errstate(all="ignore"):
                r = np.where((e1 > 1) & (e0 > 0), e1 / e0, 0)
            r = np.nan_to_num(r, nan=np.inf, posinf=np.inf)
            if np.nanmax(r) > 10:
                cell = np.unravel_index(int(np.nanargmax(r)), r.shape)
                if best is None or r[cell] > best[2]:
                    best = (c, cell, float(r[cell]))
        if best:
            seed = (s, *best)
            break
    if seed is None:
        raise SystemExit(f"no explosive growth within {a.steps} steps")
    s_seed, c_seed, (iy, ix), ratio = seed
    lat = float(cache.lat[iy, ix])
    rep["seed"] = {
        "step": s_seed + 1,
        "channel": c_seed,
        "cell": [int(iy), int(ix)],
        "lat": lat,
        "x_index": int(ix),
        "growth_ratio": ratio,
    }

    # 2. Trace the seed cell from the start.
    lvl_chans = (
        [f"{v}_{c_seed.rsplit('_', 1)[1]}" for v in ("thetao", "so", "uo", "vo")]
        if c_seed != "zos"
        else ["zos"]
    )
    w_seed = roller.weights(c_seed)[iy, ix]
    prev, cur = roller.initial()
    trace = []
    with np.errstate(all="ignore"):
        for s in range(min(s_seed + 3, a.steps)):
            F = roller.features(c_seed, cur, prev, s + 1)[iy, ix]
            contrib = F * w_seed
            nxt = roller.step(cur, prev, s + 1, check=False)
            trace.append(
                {
                    "step": s + 1,
                    "state": {c: float(nxt[c][iy, ix]) for c in lvl_chans},
                    "truth": {c: float(test[c][s + 2][iy, ix]) for c in lvl_chans},
                    "error_seed": float(
                        nxt[c_seed][iy, ix] - test[c_seed][s + 2][iy, ix]
                    ),
                    "features": dict(zip(NAMES, map(float, F))),
                    "contributions": dict(zip(NAMES, map(float, contrib))),
                    "tendency": float(contrib.sum()),
                }
            )
            prev, cur = cur, nxt
    rep["trace"] = trace

    # 3. Advection features at the seed vs their training distribution.
    A = cache.arr("train", c_seed)
    mean_c = float(np.nanmean(np.asarray(A)[:, wet[c_seed]]))
    tr = build_samples(cache, "train", c_seed, mean_c, forcing=True)
    band_seed = int(cache.band[iy, ix])
    m = tr.band == band_seed
    rep["training_range"] = {}
    for j in ADVECTION:
        col = tr.F[m, j]
        q = np.quantile(col, [1e-4, 0.5, 1 - 1e-4])
        sd = float(col.std())
        rollout_vals = [t["features"][NAMES[j]] for t in trace]
        rep["training_range"][NAMES[j]] = {
            "band": band_seed,
            "q0001_median_q9999": [float(v) for v in q],
            "std": sd,
            "rollout_values_at_seed": rollout_vals,
            "max_abs_z_in_rollout": float(np.nanmax(np.abs(rollout_vals)) / sd),
        }

    # 4. Advection weights over all channels: raw, and standardized (x feature std).
    wstats = []
    prev0, cur0 = roller.initial()  # true states at t-1, t
    for c in chans:
        F = roller.features(c, cur0, prev0, 1)
        for b in range(3):
            wv = roller.fits[c][0][b]
            mb = wet[c] & (cache.band == b)
            for j in ADVECTION:
                sd = float(F[mb, j].std()) if mb.any() else 0.0
                wstats.append(
                    {
                        "channel": c,
                        "band": b,
                        "feature": NAMES[j],
                        "w": float(wv[j]),
                        "w_x_std": float(wv[j]) * sd,
                    }
                )
    rep["advection_weights"] = wstats
    rep["advection_weight_signs"] = {
        f: {
            "positive": sum(1 for w in wstats if w["feature"] == f and w["w"] > 0),
            "negative": sum(1 for w in wstats if w["feature"] == f and w["w"] < 0),
        }
        for f in (NAMES[j] for j in ADVECTION)
    }

    # 5. Counterfactual: advection from true velocities, whole test window.
    roller.prescribed_velocity = True
    prev, cur = roller.initial()
    n = roller.n_test - 2
    cf = {"completed": True, "steps": n}
    rms = {c: [] for c in ("thetao_6", "thetao_8", c_seed)}
    try:
        for s in range(n):
            nxt = roller.step(cur, prev, s + 1)
            for c in rms:
                e = (nxt[c] - test[c][s + 2])[wet[c]]
                rms[c].append(float(np.sqrt(np.mean(e**2))))
            prev, cur = cur, nxt
    except FloatingPointError as exc:
        cf.update(completed=False, failed_at_step=s + 1, error=str(exc))
    cf["rmse_by_step"] = rms
    cf["persistence_rmse_last"] = {
        c: float(
            np.sqrt(np.mean((test[c][1] - test[c][min(s, n - 1) + 2])[wet[c]] ** 2))
        )
        for c in rms
    }
    rep["counterfactual_prescribed_velocity"] = cf

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rca.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("seed", "advection_weight_signs")}, indent=1))
    if "refactor_max_abs_diff_first_5_steps" in rep:
        print(
            "refactor check, max |diff| over 5 steps:",
            rep["refactor_max_abs_diff_first_5_steps"],
        )
    print("seed cell trace (step, error, tendency, top 4 contributions):")
    for t in trace[-8:]:
        top = sorted(t["contributions"].items(), key=lambda kv: -abs(kv[1]))[:4]
        print(
            f"  {t['step']:3d} err {t['error_seed']:.3g} tend {t['tendency']:.3g} "
            + ", ".join(f"{k}={v:.3g}" for k, v in top)
        )
    for k, v in rep["training_range"].items():
        print(
            f"{k}: training q0.0001/median/q0.9999 {v['q0001_median_q9999']}, max |z| in rollout {v['max_abs_z_in_rollout']:.3g}"
        )
    print(
        "counterfactual (true velocities):",
        {k: cf[k] for k in cf if k != "rmse_by_step"},
    )


if __name__ == "__main__":
    main()
