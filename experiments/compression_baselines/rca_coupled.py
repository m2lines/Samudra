# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Spectral radius of the fitted, coupled linear operator on a uniform grid.

    python rca_coupled.py CACHE OUT [--nk 13]

rca_vonneumann.py treats each channel alone. Here every channel's `lpc_linear`
update keeps its coupling terms too: the last change of the other variables at
its level ("other0..3") and of the levels above and below ("up_tend",
"down_tend"). On a uniform periodic grid with no land, each Fourier mode k
evolves by a two-level linear recurrence over all channels,

    x[n+1] = A(k) x[n] - B(k) x[n-1],
    A = diag(1 + S + b + L + c) + W,   B = diag(b + L) + W,

with S, b, L, c as in rca_vonneumann.py and W[i, j] the weight of channel j's
last change in channel i's update. The growth factor is the spectral radius of
the companion matrix [[A, -B], [I, 0]], maximized over k. It is computed per
latitude band (each band has its own weights), with all coupling, with vertical
coupling only, and with no coupling, to show which terms create instability.
Forcing, bias and advection terms do not act on the state perturbation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from models import DIRS, FEATURES
from rollout import fit_lpc
from run import Cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--nk", type=int, default=13)
    a = ap.parse_args()
    cache = Cache(a.cache)
    chans = cache.meta["channels"]
    pos = {c: i for i, c in enumerate(chans)}
    n = len(chans)
    rng = np.random.default_rng(0)
    fits = {c: fit_lpc(cache, c, True, 400_000, rng, linear=True)[0] for c in chans}
    k = np.linspace(-np.pi, np.pi, a.nk)
    rep: dict = {}
    for band in range(3):
        coeff = {}
        for c in chans:
            w = fits[c][band]
            others, up, dn, _, _ = cache.neighbours(c)
            coup = {}
            for j, o in enumerate(others[:4]):
                coup[o] = ("vars", w[FEATURES.index(f"other{j}")])
            if up:
                coup[up] = ("vertical", w[FEATURES.index("up_tend")])
            if dn:
                coup[dn] = ("vertical", w[FEATURES.index("down_tend")])
            coeff[c] = {
                "wd": {d: w[FEATURES.index(f"d_{d[0]}_{d[1]}")] for d in DIRS},
                "b": w[FEATURES.index("tend")],
                "l": w[FEATURES.index("lap_tend")],
                "c": w[FEATURES.index("anom")],
                "coup": coup,
            }
        res = {}
        for mode in ("all coupling", "vertical coupling only", "no coupling"):
            best: tuple[float, dict[str, Any]] = (0.0, {})
            for ky in k:
                for kx in k:
                    A = np.zeros((n, n), complex)
                    B = np.zeros((n, n), complex)
                    for c, cf in coeff.items():
                        i = pos[c]
                        S = sum(
                            v * (np.exp(1j * (ky * d[0] + kx * d[1])) - 1)
                            for d, v in cf["wd"].items()
                        )
                        L = cf["l"] * (
                            sum(np.exp(1j * (ky * d[0] + kx * d[1])) for d in DIRS[:4])
                            / 4
                            - 1
                        )
                        A[i, i] += 1 + S + cf["b"] + L + cf["c"]
                        B[i, i] += cf["b"] + L
                        for o, (kind, wv) in cf["coup"].items():
                            if mode == "no coupling" or (
                                mode == "vertical coupling only" and kind != "vertical"
                            ):
                                continue
                            A[i, pos[o]] += wv
                            B[i, pos[o]] += wv
                    M = np.block([[A, -B], [np.eye(n), np.zeros((n, n))]])
                    ev = np.linalg.eigvals(M)
                    j = int(np.argmax(np.abs(ev)))
                    if abs(ev[j]) > best[0]:
                        # which channels carry the leading eigenvector (current-time half)
                        w_, v_ = np.linalg.eig(M)
                        jj = int(np.argmax(np.abs(w_)))
                        vec = np.abs(v_[:n, jj]) ** 2
                        vec /= vec.sum()
                        top = sorted(
                            ((float(vec[pos[c]]), c) for c in chans), reverse=True
                        )[:6]
                        best = (
                            float(abs(ev[j])),
                            {"ky": float(ky), "kx": float(kx), "top_channels": top},
                        )
            res[mode] = {"max_growth": best[0], **best[1]}
            print(
                f"band {band} {mode:24s}: max growth {best[0]:.4f} at {best[1]}",
                flush=True,
            )
        rep[f"band_{band}"] = res
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "coupled.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
