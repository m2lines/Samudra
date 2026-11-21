# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Von Neumann stability of one channel's fitted linear stencil.

    python rca_vonneumann.py CACHE CHANNEL [--band 0]

Fits the `lpc_linear` rung for CHANNEL exactly as rollout.py does and keeps only
the terms that act on the channel itself (its own neighbour differences, last
change, the Laplacian of the last change, and the mean anomaly). On a uniform
periodic grid, a Fourier mode exp(i(kx x + ky y)) then evolves by a two-level
recurrence

    x[n+1] = (1 + S + b + L + c) x[n] - (b + L) x[n-1]

with S = sum_d w_d (exp(i k.d) - 1) over the 8 neighbour differences, b the
weight on the last change, L = l (mean of the 4 nearest exp(i k.d) - 1) for the
Laplacian-of-change weight l, and c the mean-anomaly weight. The amplification
factor is the larger root |lambda| of lambda^2 - A lambda + B = 0. This ignores
coupling to other variables and levels, and land, so it explains the local
instability rather than reproducing the global operator exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from models import DIRS, FEATURES
from rollout import fit_lpc
from run import Cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("channel")
    ap.add_argument("--band", type=int, default=0)
    a = ap.parse_args()
    cache = Cache(a.cache)
    w_band, _ = fit_lpc(
        cache, a.channel, True, 400_000, np.random.default_rng(0), linear=True
    )
    w = w_band[a.band]
    wd = {d: w[FEATURES.index(f"d_{d[0]}_{d[1]}")] for d in DIRS}
    b, lap, c = (w[FEATURES.index(n)] for n in ("tend", "lap_tend", "anom"))
    print(f"{a.channel} band {a.band}: neighbour weights {{(dy,dx): w}} =")
    print("  " + ", ".join(f"{d}: {v:+.4f}" for d, v in wd.items()))
    print(
        f"  last change b = {b:+.4f}, Laplacian of change l = {lap:+.4f}, mean anomaly c = {c:+.2e}"
    )
    k = np.linspace(-np.pi, np.pi, 181)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    S = sum(v * (np.exp(1j * (KY * d[0] + KX * d[1])) - 1) for d, v in wd.items())
    L = lap * (sum(np.exp(1j * (KY * d[0] + KX * d[1])) for d in DIRS[:4]) / 4 - 1)
    A = 1 + S + b + L + c
    B = b + L
    disc = np.sqrt(A**2 - 4 * B + 0j)
    lam = np.maximum(np.abs((A + disc) / 2), np.abs((A - disc) / 2))
    iy, ix = np.unravel_index(np.argmax(lam), lam.shape)
    print(
        f"max |lambda| = {lam.max():.4f} at (ky, kx) = ({k[iy]:+.3f}, {k[ix]:+.3f}) "
        f"[pi = grid-scale]; |lambda| at k=0: {lam[90, 90]:.4f}"
    )
    # Zonal-only slice (ky = 0): the stencil acting along x.
    zr = lam[90]
    print(
        f"zonal slice ky=0: max |lambda| = {zr.max():.4f} at kx = {k[np.argmax(zr)]:+.3f}"
    )
    # The one-sided east difference alone: x <- x + a (x_east - x).
    a_e = wd[(0, 1)]
    print(
        f"east difference alone (a = {a_e:+.4f}): stable iff 0 <= a <= 1; "
        f"grid-scale growth |1 - 2a| = {abs(1 - 2 * a_e):.4f}"
    )


if __name__ == "__main__":
    main()
