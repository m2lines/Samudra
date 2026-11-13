# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Score external one-step forecasts (e.g. Samudra 2) next to the ladder.

    python score_external.py CACHE LADDER_RESULTS PRED_DIR OUT [--name samudra2]

CACHE is the ladder's cache of the *same* store the forecasts were made on, and
LADDER_RESULTS its results.json (for the bin tolerance). PRED_DIR holds
one_step.py output: val/ and test/, each with pred.npy and meta.json.

For every channel, over all wet cells and the forecasts' target times:
- RMSE of the forecast and of persistence on the identical cells and times, and
  the forecast's RMSE skill against persistence;
- bits per value at the ladder's tolerance. The forecast becomes a distribution
  by minimal EMOS: a Laplace centred on the forecast, with one spread per channel
  set to the mean absolute error of its *validation* forecasts. Persistence uses
  the Laplace spread fit on the cache's training tendencies, as the ladder does.
  Both use the ladder's floor (FLOOR, REF_MULT).

The ladder's own numbers (from LADDER_RESULTS) are copied beside them. Its `mix`
is scored on 8,000 area-sampled rows over every test step, the forecast here on
every wet cell of every other test step: the same population, different samples.
As a check on the time and bin alignment, persistence bits computed here are
compared with the ladder's full-field persistence bits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FLOOR, REF_MULT = 1e-9, 10.0
VARS = ["thetao", "so", "uo", "vo", "zos"]


def var_of(c: str) -> str:
    return "zos" if c == "zos" else c.rsplit("_", 1)[0]


def laplace_interval(lo, hi, mu, b):
    """P(lo < Y <= hi), Y ~ Laplace(mu, b), accurate in both tails."""
    lo, hi = (lo - mu) / b, (hi - mu) / b
    right = 0.5 * (np.exp(-np.maximum(lo, 0)) - np.exp(-np.maximum(hi, 0)))
    left = 0.5 * (np.exp(np.minimum(hi, 0)) - np.exp(np.minimum(lo, 0)))
    mid = 1 - 0.5 * np.exp(-np.maximum(hi, 0)) - 0.5 * np.exp(np.minimum(lo, 0))
    return np.where(lo >= 0, right, np.where(hi <= 0, left, mid))


def rows(cache: Path, meta: dict, split: str, c: str, pred: np.ndarray, times: list):
    """Truth now (x), truth next (y_next), forecast, area, over wet cells."""
    A = np.load(cache / split / f"{c}.npy", mmap_mode="r")
    stamps = meta["splits"][split]
    wet = np.load(cache / "static.npz")["masks"][meta["levels"][c]]
    area = np.load(cache / "static.npz")["area"]
    xs, ys, fs, ws = [], [], [], []
    for k, t in enumerate(times):
        if t not in stamps:
            raise ValueError(f"{c}: forecast time {t} not in cached {split} split")
        j = stamps.index(t)
        if j == 0:
            continue  # no true state at t in this split
        x, y, f = np.asarray(A[j - 1]), np.asarray(A[j]), pred[k]
        ok = wet & np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
        xs.append(x[ok]), ys.append(y[ok]), fs.append(f[ok]), ws.append(area[ok])
    return tuple(np.concatenate(z).astype(np.float64) for z in (xs, ys, fs, ws))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("ladder", type=Path)
    ap.add_argument("pred", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--name", default="samudra2")
    a = ap.parse_args()
    meta = json.loads((a.cache / "meta.json").read_text())
    L = json.loads(a.ladder.read_text())
    lad = {c["channel"]: c for c in L["channels"]}
    P = {}
    for split in ("val", "test"):
        m = json.loads((a.pred / split / "meta.json").read_text())
        if not m["complete"]:
            raise SystemExit(f"{split} forecasts are incomplete (smoke run?)")
        P[split] = (np.load(a.pred / split / "pred.npy", mmap_mode="r"), m)
    names = P["test"][1]["channels"]
    res = {}
    for c in lad:
        i = names.index(c)
        w = lad[c]["bin_width"]
        sig = L["meta"]["sigma_delta"][var_of(c)]
        # EMOS spread from validation forecasts
        xv, yv, fv, _ = rows(
            a.cache, meta, "val", c, P["val"][0][:, i], P["val"][1]["target_times"]
        )
        b_model = float(np.mean(np.abs(yv - fv)))
        # persistence spread from training tendencies, as the ladder fits it
        T = np.load(a.cache / "train" / f"{c}.npy", mmap_mode="r")
        wet = np.load(a.cache / "static.npz")["masks"][meta["levels"][c]]
        d = (np.asarray(T[1:]) - np.asarray(T[:-1]))[:, wet]
        b_pers = float(np.nanmean(np.abs(d)))
        x, y, f, area = rows(
            a.cache, meta, "test", c, P["test"][0][:, i], P["test"][1]["target_times"]
        )
        idx = np.floor(y / w)
        lo, hi = idx * w - x, (idx + 1) * w - x  # true bin, as a tendency interval
        q = laplace_interval(lo, hi, 0.0, REF_MULT * sig)

        def bits(p, q=q):
            return -np.log2((1 - FLOOR) * p + FLOOR * q)

        b_m = bits(laplace_interval(lo, hi, f - x, b_model))
        b_p = bits(laplace_interval(lo, hi, 0.0, b_pers))
        rm = float(np.sqrt(np.average((f - y) ** 2, weights=area)))
        rp = float(np.sqrt(np.average((x - y) ** 2, weights=area)))
        res[c] = {
            "depth": lad[c]["depth"],
            "n_rows": int(len(y)),
            "emos_laplace_b": b_model,
            "bits_model": float(np.average(b_m, weights=area)),
            "bits_persistence": float(np.average(b_p, weights=area)),
            "rmse_model": rm,
            "rmse_persistence": rp,
            "rmse_skill_model": 1 - rm / rp,
            "ladder_bits_mix": lad[c]["bits"]["mix"],
            "ladder_bits_persistence_full_field": lad[c]["bits_full_field"][
                "persistence"
            ],
            "ladder_rmse_skill_mix": 1
            - lad[c]["rmse"]["mix"] / lad[c]["rmse"]["persistence"],
        }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.name}_vs_ladder.json").write_text(json.dumps(res, indent=1))
    wa = L["meta"]["wet_area"]
    nm = a.name
    lines = [
        f"One-step forecasts on the same truth. Per variable: levels weighted by wet area, surface in brackets. "
        f"`{nm}` bits use minimal EMOS (one Laplace spread per channel from validation errors).",
        "",
        f"| variable | persistence bits | ladder mix bits | {nm} bits | RMSE skill vs persistence, ladder mix | RMSE skill vs persistence, {nm} |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in VARS:
        cs = sorted([c for c in res if var_of(c) == v], key=lambda c: res[c]["depth"])
        if not cs:
            continue
        wts = [wa[c] for c in cs]

        def agg(key, nd=2, cs=cs, wts=wts):
            vals = [res[c][key] for c in cs]
            return f"{np.average(vals, weights=wts):.{nd}f} [{vals[0]:.{nd}f}]"

        lines.append(
            f"| {v} | {agg('bits_persistence')} | {agg('ladder_bits_mix')} | {agg('bits_model')} "
            f"| {agg('ladder_rmse_skill_mix', 3)} | {agg('rmse_skill_model', 3)} |"
        )
    lines += [
        "",
        f"| channel | depth (m) | persistence bits (here) | persistence bits (ladder, full field) | ladder mix bits | {nm} bits | RMSE {nm} | RMSE persistence | RMSE skill, ladder mix | RMSE skill, {nm} |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in sorted(res, key=lambda c: (var_of(c), res[c]["depth"])):
        r = res[c]
        lines.append(
            f"| {c} | {r['depth']:g} | {r['bits_persistence']:.2f} | {r['ladder_bits_persistence_full_field']:.2f} "
            f"| {r['ladder_bits_mix']:.2f} | {r['bits_model']:.2f} | {r['rmse_model']:.4g} | {r['rmse_persistence']:.4g} "
            f"| {r['ladder_rmse_skill_mix']:.3f} | {r['rmse_skill_model']:.3f} |"
        )
    (a.out / f"{nm}_vs_ladder.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
