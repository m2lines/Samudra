# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Neighbour correlation of an external model's one-step errors, and the joint-coding gap it implies.

    python error_corr.py CACHE PRED_DIR OUT.csv

The same estimate run.py makes for the ladder's `lpc` errors: rho is the
correlation between each wet cell's error and its eastern neighbour's (periodic
in longitude), pooled over test targets, and the gap is -1/2 log2(1 - rho^2)
bits per value. Done three ways per channel:

* `model`: errors of the forecasts in PRED_DIR (from one_step.py);
* `model_debiased`: the same, minus each cell's time-mean error. A fixed bias
  correlates neighbours, but it is fixed more cheaply than with a joint loss;
* `persistence`: errors of "no change", for scale.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def gap(rho: float) -> float:
    return float(-0.5 * np.log2(max(1 - rho**2, 1e-12)))


def zonal_rho(E: np.ndarray) -> float:
    e = np.roll(E, -1, axis=-1)
    ok = np.isfinite(E) & np.isfinite(e)
    return float(np.corrcoef(E[ok], e[ok])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("pred", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    meta = json.loads((a.cache / "meta.json").read_text())
    pm = json.loads((a.pred / "test" / "meta.json").read_text())
    if not pm["complete"]:
        raise SystemExit("test forecasts are incomplete")
    P = np.load(a.pred / "test" / "pred.npy", mmap_mode="r")
    masks = np.load(a.cache / "static.npz")["masks"]
    stamps = meta["splits"]["test"]
    js = [stamps.index(t) for t in pm["target_times"]]
    rows = []
    for i, c in enumerate(pm["channels"]):
        A = np.load(a.cache / "test" / f"{c}.npy", mmap_mode="r")
        wet = masks[meta["levels"][c]]
        y = np.asarray(A[js], np.float64)
        x = np.asarray(A[[j - 1 for j in js]], np.float64)
        f = np.asarray(P[:, i], np.float64)
        y[:, ~wet] = np.nan
        Em, Ep = y - f, y - x
        Ed = Em - np.nanmean(Em, axis=0, keepdims=True)
        r = {"channel": c}
        for k, E in (("model", Em), ("model_debiased", Ed), ("persistence", Ep)):
            rho = zonal_rho(E)
            r[f"{k}_rho"] = rho
            r[f"{k}_gap_bits"] = gap(rho)
        rows.append(r)
        print(
            f"{c:10s} "
            + " ".join(f"{k}={v:.3f}" for k, v in r.items() if k != "channel"),
            flush=True,
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
