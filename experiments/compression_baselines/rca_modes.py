# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Leading unstable mode of the ladder's linear rollout operator.

    python rca_modes.py CACHE OUT [--steps 80] [--eps 1e-6]

The `lpc_linear` step is affine in the two-level state (x[t-1], x[t]):
x[t+1] = M (x[t-1], x[t]) + forcing terms + constants. Two rollouts with the same
forcing whose states differ by a small perturbation therefore differ by exactly
M^n applied to it. Renormalizing that difference every step is power iteration:
the per-step growth factor converges to the largest |eigenvalue| of M, and the
difference converges to its eigenvector, the leading mode.

Each channel is measured in units of its variable's sigma_delta (the spread of
five-day changes, as in the bins), so modes in temperature and velocity are
comparable. Writes OUT/modes.json: growth factor per step, and the converged
mode's share of squared norm by channel, by variable, by depth and by latitude
band, plus its largest cells.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rollout import Roller
from run import Cache, sigma_delta, var_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cache = Cache(a.cache)
    roller = Roller(cache, "lpc_linear", forcing=True)
    chans = roller.chans
    sig = {v: sigma_delta(cache, v) for v in dict.fromkeys(map(var_of, chans))}
    scale = {c: sig[var_of(c)] for c in chans}
    wet = roller.wet
    rng = np.random.default_rng(a.seed)

    def norm(dc: dict, dp: dict) -> float:
        tot = 0.0
        for c in chans:
            tot += float(np.sum((dc[c][wet[c]] / scale[c]) ** 2))
            tot += float(np.sum((dp[c][wet[c]] / scale[c]) ** 2))
        return float(np.sqrt(tot))

    prev, cur = roller.initial()
    # Random perturbation in sigma_delta units, on wet cells, both time levels.
    dprev = {
        c: np.where(wet[c], rng.normal(size=wet[c].shape) * scale[c], 0) for c in chans
    }
    dcur = {
        c: np.where(wet[c], rng.normal(size=wet[c].shape) * scale[c], 0) for c in chans
    }
    n0 = norm(dcur, dprev)
    dprev = {c: dprev[c] * a.eps / n0 for c in chans}
    dcur = {c: dcur[c] * a.eps / n0 for c in chans}
    growth = []
    for s in range(a.steps):
        t = s + 1
        nxt = roller.step(cur, prev, t, check=False)
        pc = {c: cur[c] + dcur[c] for c in chans}
        pp = {c: prev[c] + dprev[c] for c in chans}
        pn = roller.step(pc, pp, t, check=False)
        dnext = {c: np.where(wet[c], pn[c] - nxt[c], 0) for c in chans}
        g = norm(dnext, dcur) / norm(dcur, dprev)
        growth.append(g)
        # renormalize the pair (dnext, dcur) back to size eps
        k = a.eps / norm(dnext, dcur)
        dprev, dcur = {c: dcur[c] * k for c in chans}, {c: dnext[c] * k for c in chans}
        prev, cur = cur, nxt
        if s % 10 == 0:
            print(f"[modes] step {t}: growth over the step {g:.4f}", flush=True)
    # Composition of the converged mode (the latest state component).
    e = {c: (dcur[c][wet[c]] / scale[c]) ** 2 for c in chans}
    tot = sum(float(v.sum()) for v in e.values())
    by_chan = {c: float(e[c].sum() / tot) for c in chans}
    by_var: dict[str, float] = {}
    for c, v in by_chan.items():
        by_var[var_of(c)] = by_var.get(var_of(c), 0.0) + v
    by_depth: dict[float, float] = {}
    for c, v in by_chan.items():
        d = cache.meta["depth"][c]
        by_depth[d] = by_depth.get(d, 0.0) + v
    lat = cache.lat
    bands = {
        "tropics |lat|<20": np.abs(lat) < 20,
        "mid 20-50": (np.abs(lat) >= 20) & (np.abs(lat) < 50),
        "high >=50": np.abs(lat) >= 50,
    }
    by_band = {
        name: float(
            sum(
                np.sum(np.where(wet[c] & m, (dcur[c] / scale[c]) ** 2, 0))
                for c in chans
            )
            / tot
        )
        for name, m in bands.items()
    }
    # cells with at least one land cell among their 8 neighbours (surface mask)
    m0 = wet[chans[0]]
    coastal = np.zeros_like(m0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (dy, dx) != (0, 0):
                coastal |= ~np.roll(np.roll(m0, dy, 0), dx, 1)
    cells = []
    for c in chans:
        f = np.where(wet[c], (dcur[c] / scale[c]) ** 2 / tot, 0)
        for idx in np.argsort(f.ravel())[-3:]:
            iy, ix = np.unravel_index(idx, f.shape)
            land = sum(
                not wet[c][(iy + dy) % wet[c].shape[0], (ix + dx) % wet[c].shape[1]]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if (dy, dx) != (0, 0)
            )
            cells.append((float(f[iy, ix]), c, float(lat[iy, ix]), int(ix), int(land)))
    cells.sort(reverse=True)
    rep: dict[str, Any] = {
        "growth_per_step": growth,
        "growth_last10_mean": float(np.mean(growth[-10:])),
        "mode_share_by_variable": by_var,
        "mode_share_by_depth": {str(k): v for k, v in sorted(by_depth.items())},
        "mode_share_by_lat_band": by_band,
        "mode_share_top_channels": dict(
            sorted(by_chan.items(), key=lambda kv: -kv[1])[:10]
        ),
        "mode_top_cells": [
            {
                "share": s_,
                "channel": c,
                "lat": la,
                "x_index": x,
                "land_neighbours_of_8": ln,
            }
            for s_, c, la, x, ln in cells[:15]
        ],
        "share_of_wet_cells_next_to_land": float(np.mean(coastal[wet[chans[0]]])),
        "mode_share_on_cells_next_to_land": float(
            sum(
                np.sum(np.where(wet[c] & coastal, (dcur[c] / scale[c]) ** 2, 0))
                for c in chans
            )
            / tot
        ),
        "cell_area_share_by_band": {
            name: float(m[wet[chans[0]]].mean()) for name, m in bands.items()
        },
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "modes.json").write_text(json.dumps(rep, indent=1))
    print(
        json.dumps({k: v for k, v in rep.items() if k != "growth_per_step"}, indent=1)
    )


if __name__ == "__main__":
    main()
