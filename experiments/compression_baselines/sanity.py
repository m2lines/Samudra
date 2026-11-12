# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Sanity checks on a compression-baselines run (protocol v2: tolerance bins).

Independent of the experiment's code (no imports from it): recomputes the
per-variable spread sigma_delta, the bin widths and persistence bits from the
cache with plain numpy, and audits results.json.

usage: python sanity.py CACHE RESULTS_JSON OUT_JSON   (writes OUT_JSON; prints a summary)
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import Blosc  # type: ignore[import-untyped]

cache, res_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
R = json.loads(res_path.read_text())
meta = json.loads((cache / "meta.json").read_text())
st = np.load(cache / "static.npz")
masks, area = st["masks"], st["area"]
EPS = R["args"]["eps"]
FLOOR, REF_MULT = 1e-9, 10.0
RUNGS = ["persistence", "ar1", "lpc", "lpc_ctx", "analog", "lpc_analog"]
out: dict[str, Any] = {}


def var_of(c):
    return "zos" if c == "zos" else c.rsplit("_", 1)[0]


def laplace_interval(lo, hi, b):
    """P(lo < Y <= hi), Y ~ Laplace(0, b), written per case to stay accurate in tails."""
    lo, hi = lo / b, hi / b
    right = 0.5 * (np.exp(-np.maximum(lo, 0)) - np.exp(-np.maximum(hi, 0)))
    left = 0.5 * (np.exp(np.minimum(hi, 0)) - np.exp(np.minimum(lo, 0)))
    straddle = 1 - 0.5 * np.exp(-np.maximum(hi, 0)) - 0.5 * np.exp(np.minimum(lo, 0))
    return np.where(lo >= 0, right, np.where(hi <= 0, left, straddle))


def triples(split, c):
    """(x_t, tendency) at every wet interior (t-1, t, t+1) with all three finite."""
    A = np.load(cache / split / f"{c}.npy")
    wet = masks[meta["levels"][c]]
    prev, cur, nxt = A[:-2][:, wet], A[1:-1][:, wet], A[2:][:, wet]
    ok = np.isfinite(prev) & np.isfinite(cur) & np.isfinite(nxt)
    w = np.broadcast_to(area[wet], cur.shape)
    return cur[ok].astype(np.float64), (nxt - cur)[ok].astype(np.float64), w[ok]


# ---- (a) sigma_delta per variable and bin widths, recomputed
chans = [ch["channel"] for ch in R["channels"]]
sig = {}
for v in dict.fromkeys(map(var_of, chans)):
    ds, ws = [], []
    for c in meta["channels"]:
        if var_of(c) != v:
            continue
        A = np.load(cache / "train" / f"{c}.npy")
        wet = masks[meta["levels"][c]]
        d = np.diff(A, axis=0)[:, wet].astype(np.float64)
        ok = np.isfinite(d)
        ds.append(d[ok])
        ws.append(np.broadcast_to(area[wet], d.shape)[ok])
    d, w = np.concatenate(ds), np.concatenate(ws)
    mu = np.average(d, weights=w)
    sig[v] = float(np.sqrt(np.average((d - mu) ** 2, weights=w)))
bins_audit = {}
for ch in R["channels"]:
    c = ch["channel"]
    width = EPS * sig[var_of(c)]
    _, y, _ = triples("test", c)
    bins_audit[c] = {
        "bin_width_run": ch["bin_width"],
        "bin_width_recomputed": width,
        "sigma_delta_recomputed": sig[var_of(c)],
        "test_tendency_mad_over_bin_width": float(np.mean(np.abs(y)) / width),
    }
out["bins"] = bins_audit

# ---- (b) mix vs its best component on test, in bits and RMSE
viol_bits, viol_rmse, margins = [], [], {}
for ch in R["channels"]:
    b, r = ch["bits"], ch["rmse"]
    best = min(RUNGS, key=lambda m: b[m])
    margins[ch["channel"]] = {
        "best_component": best,
        "mix_minus_best_bits": b["mix"] - b[best],
    }
    if b["mix"] > b[best]:
        viol_bits.append((ch["channel"], best, b["mix"] - b[best]))
    bestr = min(RUNGS, key=lambda m: r[m])
    if r["mix"] > r[bestr]:
        viol_rmse.append((ch["channel"], bestr, r["mix"] / r[bestr] - 1))
out["mix_vs_best"] = {
    "bits_violations": viol_bits,
    "rmse_violations": viol_rmse,
    "margins": margins,
}

# ---- (c) sampled (8000 rows) vs full-field bits for the cheap rungs
out["sample_vs_full"] = {
    ch["channel"]: {
        m: ch["bits"][m] - ch["bits_full_field"][m] for m in ch["bits_full_field"]
    }
    for ch in R["channels"]
}

# ---- (d) hand-recomputed persistence bits (all wet test cells, area-weighted)
spot = {}
for c in [c for c in ("thetao_0", "so_10", "uo_18", "zos") if c in bins_audit]:
    width = bins_audit[c]["bin_width_recomputed"]
    ref = REF_MULT * sig[var_of(c)]
    _, ytr, _ = triples("train", c)
    b = float(np.mean(np.abs(ytr)))  # full train set; the run may use a subset
    x, y, w = triples("test", c)
    idx = np.floor((x + y) / width)
    lo, hi = idx * width - x, (idx + 1) * width - x
    p = laplace_interval(lo, hi, b)
    q = laplace_interval(lo, hi, ref)
    bits = -np.log2((1 - FLOOR) * p + FLOOR * q)
    run = next(ch for ch in R["channels"] if ch["channel"] == c)
    rows = [
        {
            "x": float(x[i]),
            "next": float(x[i] + y[i]),
            "bin": int(idx[i]),
            "p": float(p[i]),
            "bits": float(bits[i]),
        }
        for i in np.random.default_rng(0).choice(len(x), 4, replace=False)
    ]
    spot[c] = {
        "laplace_b_full_train": b,
        "hand_bits_area_weighted": float(np.average(bits, weights=w)),
        "run_bits_full_field": run["bits_full_field"]["persistence"],
        "run_bits_sampled": run["bits"]["persistence"],
        "n_rows": int(len(x)),
        "example_rows": rows,
    }
out["persistence_spot_check"] = spot

# ---- (e) blosc land overhead: the mask alone, per wet value, per level
bl = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
over = {}
for k in range(masks.shape[0]):
    q = np.zeros(masks[k].shape, np.int32)
    q[masks[k]] = 4096
    over[k] = {
        "wet_cells": int(masks[k].sum()),
        "wet_fraction": float(masks[k].mean()),
        "mask_only_bits_per_wet_value": 8.0 * len(bl.encode(q)) / masks[k].sum(),
    }
out["blosc_mask_overhead_by_level"] = over

# ---- (f) blosc residual bound vs mix, raw and with the mask-only cost removed
gap = {}
for ch in R["channels"]:
    lvl = meta["levels"][ch["channel"]]
    raw = ch["blosc_bits_per_value"]["residual_vs_persistence"] - ch["bits"]["mix"]
    gap[ch["channel"]] = {
        "blosc_bound_minus_mix": raw,
        "blosc_bound_minus_mask_overhead_minus_mix": raw
        - over[lvl]["mask_only_bits_per_wet_value"],
    }
out["blosc_bound_vs_mix"] = gap

# ---- summaries
out["summary"] = {
    "n_channels": len(R["channels"]),
    "mix_bits_worse_than_best_component": len(viol_bits),
    "mix_rmse_worse_than_best_component": len(viol_rmse),
    "max_mix_rmse_excess_rel": max((v[2] for v in viol_rmse), default=0.0),
    "max_abs_sampled_minus_full_bits": max(
        abs(x) for d in out["sample_vs_full"].values() for x in d.values()
    ),
    "max_rel_bin_width_mismatch": max(
        abs(v["bin_width_run"] / v["bin_width_recomputed"] - 1)
        for v in bins_audit.values()
    ),
    "coarse_regime_channels": sorted(
        c for c, v in bins_audit.items() if v["test_tendency_mad_over_bin_width"] < 1
    ),
    "min_blosc_bound_minus_mask_overhead_minus_mix": min(
        v["blosc_bound_minus_mask_overhead_minus_mix"] for v in gap.values()
    ),
    "top_mix_weight_counts": dict(
        Counter(max(c["mix_weights"], key=c["mix_weights"].get) for c in R["channels"])
    ),
    "largest_step_among_ar1_lpc_lpc_ctx_counts": dict(
        Counter(
            max(
                {
                    "ar1": c["bits"]["persistence"] - c["bits"]["ar1"],
                    "lpc": c["bits"]["ar1"] - c["bits"]["lpc"],
                    "lpc_ctx": c["bits"]["lpc"] - c["bits"]["lpc_ctx"],
                }.items(),
                key=lambda kv: kv[1],
            )[0]
            for c in R["channels"]
        )
    ),
    # thetao below 1400 m and so below 375 m: where tendencies are ~1 bin or less
    "max_abs_persistence_hand_minus_run_full_field": max(
        abs(s["hand_bits_area_weighted"] - s["run_bits_full_field"])
        for s in spot.values()
    ),
}

Path(out_path).write_text(json.dumps(out, indent=1))
print(json.dumps(out["summary"], indent=1))
print("bits violations:", viol_bits)
print("rmse violations:", viol_rmse)
for c, s in spot.items():
    print(
        c,
        {
            k: s[k]
            for k in (
                "laplace_b_full_train",
                "hand_bits_area_weighted",
                "run_bits_full_field",
                "run_bits_sampled",
            )
        },
    )
print(
    "max |bin_width run - recomputed| rel:",
    max(
        abs(v["bin_width_run"] / v["bin_width_recomputed"] - 1)
        for v in bins_audit.values()
    ),
)
