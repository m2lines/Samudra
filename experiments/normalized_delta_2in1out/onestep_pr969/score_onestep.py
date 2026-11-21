# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Score one-step forecasts from #980's one_step.py against persistence.

    python score_onestep.py OM4.zarr OUT.json NAME=PRED_DIR [NAME=PRED_DIR ...]

All models are scored on the target dates they share, so a 2-output model
(forecasts every other step) and a 1-output model compare on the same dates.
Persistence is the true state one step before each target. Means over space
are cos(latitude)-weighted over wet cells (cells finite in the truth and
every forecast).
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

truth_path, out_path, *run_args = sys.argv[1:]
runs = dict(r.split("=", 1) for r in run_args)
data = xr.open_zarr(truth_path)
metas = {k: json.loads((Path(v) / "meta.json").read_text()) for k, v in runs.items()}
preds = {k: np.load(Path(v) / "pred.npy", mmap_mode="r") for k, v in runs.items()}
common = sorted(set.intersection(*(set(m["target_times"]) for m in metas.values())))
idx = {k: [m["target_times"].index(t) for t in common] for k, m in metas.items()}
channels = metas[next(iter(metas))]["channels"]
for m in metas.values():
    assert m["channels"] == channels, "channel order differs between runs"

times = data.time.values
time_str = np.array([str(t) for t in times])
pos = {t: i for i, t in enumerate(time_str)}
target_i = np.array([pos[t] for t in common])
lat = np.asarray(data.lat.values, dtype=np.float64)
coslat = np.cos(np.deg2rad(lat))
depths = data.lev.values if "lev" in data else None


def wmean(x, w):
    return float(np.nansum(x * w) / np.nansum(np.where(np.isfinite(x), w, 0)))


result = {
    "n_dates": len(common),
    "first": common[0],
    "last": common[-1],
    "depths_m": None if depths is None else [float(d) for d in depths],
    "channels": {},
}
for c in channels:
    truth = data[c].isel(time=target_i).values.astype(np.float64)
    persist = data[c].isel(time=target_i - 1).values.astype(np.float64)
    f = {
        k: np.asarray(preds[k][idx[k], channels.index(c)], dtype=np.float64)
        for k in runs
    }
    wet = np.isfinite(truth).all(axis=0) & np.isfinite(persist).all(axis=0)
    for v in f.values():
        wet &= np.isfinite(v).all(axis=0)
    w = np.where(wet, coslat, 0.0)
    row: dict[str, Any] = {}
    err_p = persist - truth
    mse_p_cell = np.mean(err_p**2, axis=0)
    row["persistence_rmse"] = float(
        np.sqrt(wmean(np.where(wet, mse_p_cell, np.nan), w))
    )
    for k, v in f.items():
        err = v - truth
        mse_cell = np.mean(err**2, axis=0)
        bias = np.mean(err, axis=0)
        r = {
            "rmse": float(np.sqrt(wmean(np.where(wet, mse_cell, np.nan), w))),
            "mean_abs_bias": wmean(np.where(wet, np.abs(bias), np.nan), w),
            "area_worse_than_persistence": wmean(
                np.where(wet, (mse_cell > mse_p_cell).astype(float), np.nan), w
            ),
        }
        band = wet & (np.abs(lat) < 5)
        r["equatorial_mean_bias"] = wmean(
            np.where(band, bias, np.nan), np.where(band, coslat, 0)
        )
        r["skill_vs_persistence"] = 1 - r["rmse"] ** 2 / row["persistence_rmse"] ** 2
        row[k] = r
    result["channels"][c] = row
    print(
        c,
        {k: round(row[k]["rmse"] / row["persistence_rmse"], 3) for k in runs},
        flush=True,
    )
Path(out_path).write_text(json.dumps(result, indent=1))
print("wrote", out_path)
