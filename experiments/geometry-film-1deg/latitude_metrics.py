# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Physical-unit error and stability by lead, channel and Gaussian latitude band."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr

from samudra.config import TrainConfig
from samudra.utils.data import extract_wet_mask

p = argparse.ArgumentParser()
p.add_argument("config")
a = p.parse_args()
cfg = TrainConfig.from_yaml(a.config)
root = Path(str(cfg.experiment.resolved_data_root))
src = cfg.data.sources[0]
data, _, _, layout = src.canonicalize_datasets(
    xr.open_zarr(root / "OM4.zarr"),
    xr.open_zarr(root / "OM4_means.zarr"),
    xr.open_zarr(root / "OM4_stds.zarr"),
)
masks = extract_wet_mask(
    data, layout.prognostic_var_names, layout.boundary_var_names
).prognostic.numpy()
# Exact Gaussian quadrature areas; common longitude factors cancel in means.
lat = np.asarray(data.lat.values)
area = np.broadcast_to(
    np.polynomial.legendre.leggauss(len(lat))[1][:, None], masks.shape[-2:]
)
bands = [("global", -90, 90)] + [
    (f"{lo}:{lo + 30}", lo, lo + 30) for lo in range(-90, 90, 30)
]
stores = sorted((cfg.experiment.output_dir / "evals").rglob("predictions.zarr"))
if len(stores) != 2:
    raise RuntimeError(f"Expected raw epoch-70 and EMA stores, found {stores}")
for store in stores:
    pred = xr.open_zarr(store)
    pred_lat = pred.lat.values
    if pred_lat.ndim == 2:
        pred_lat = pred_lat[:, 0]
    np.testing.assert_allclose(pred_lat, lat, atol=1e-5)
    truth = data.sel(time=pred.time)
    times = pred.time.values
    if len(times) != 598:
        raise RuntimeError(f"Expected 598 forecast times, found {len(times)}")
    # First prediction is five days after the final state in the initial history.
    leads = np.array([(t - times[0]).total_seconds() / 86400 + 5 for t in times])
    total_nonfinite = 0
    with (store.parent / "latitude-errors.csv").open("w") as f:
        out = csv.writer(f)
        out.writerow(
            [
                "channel",
                "latitude_band",
                "time",
                "lead_days",
                "rmse",
                "bias",
                "prediction_rms",
                "prediction_absmax",
                "nonfinite_wet_cells",
            ]
        )
        for ci, name in enumerate(layout.prognostic_var_names):
            base, sep, level = name.rpartition("_")
            field = (
                pred[base].isel(lev=int(level), drop=True)
                if sep and level.isdigit()
                else pred[name]
            )
            values = field.transpose("time", "y", "x").values.astype(np.float64)
            target = (
                truth[name].transpose("time", "lat", "lon").values.astype(np.float64)
            )
            wet = masks[ci]
            if not np.isfinite(target[:, wet]).all():
                raise RuntimeError(f"Nonfinite ground truth in wet cells: {name}")
            for band, lo, hi in bands:
                mask = wet & ((lat >= lo) & (lat < hi))[:, None]
                weights = np.where(mask, area, 0)
                if not weights.sum():
                    continue
                weights /= weights.sum()
                error = np.where(mask, values - target, 0)
                selected = np.where(mask, values, 0)
                rmse = np.sqrt(np.sum(error**2 * weights, axis=(1, 2)))
                bias = np.sum(error * weights, axis=(1, 2))
                rms = np.sqrt(np.sum(selected**2 * weights, axis=(1, 2)))
                maximum = np.max(np.abs(selected), axis=(1, 2))
                invalid = (~np.isfinite(values[:, mask])).sum(axis=1)
                if band == "global":
                    total_nonfinite += int(invalid.sum())
                out.writerows(
                    zip(
                        [name] * len(times),
                        [band] * len(times),
                        map(str, times),
                        leads,
                        rmse,
                        bias,
                        rms,
                        maximum,
                        invalid,
                    )
                )
    (store.parent / "stability.json").write_text(
        json.dumps(
            dict(
                forecast_times=len(times),
                nonfinite_wet_cells=total_nonfinite,
                latitude_csv="latitude-errors.csv",
                weights="Gaussian quadrature weights, wet cells only",
                checkpoint=str(pred.attrs.get("model_path", "")),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        store,
        "latitude/stability metrics complete; nonfinite wet cells:",
        total_nonfinite,
        flush=True,
    )
