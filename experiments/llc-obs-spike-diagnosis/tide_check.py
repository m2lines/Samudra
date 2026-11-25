# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Step 0: how much of the coarsened LLC surface signal is sub-daily?

LLC4320 is tidally forced and the coarsened store keeps instantaneous hourly
snapshots. DUACS ADT is detided and has high-frequency atmospheric effects
removed, and OISST is a daily analysis. If much of LLC's SSH (or velocity) is
sub-daily, the six-hour LLC targets ask the model to predict signals that the
inputs (no tidal potential) cannot determine, and LLC "SSH" is a different
quantity from observed SSH.

Two outputs, from several 30-day hourly segments of the training period:

1. Spectral partition of variance per wet cell, for Eta and level-0 Theta, U
   and V: low (< 0.5 cpd), diurnal (0.8-1.2 cpd), semidiurnal (1.7-2.3 cpd) and
   other high frequencies, plus the share of six-hourly variance that a daily
   mean removes (what the 6 h training targets contain beyond a daily state).
   Wet-area-weighted global means and 10-degree latitude bands.
2. LLC Eta against DUACS ADT for each training date in the segments: RMSE of the
   midnight snapshot (what training uses) and of the daily mean, raw, after
   removing each date's global mean difference, and after also removing the
   time-mean difference map (a datum or geoid-like offset).

Run on a CPU node with ~64 GB of memory. Writes JSON and an .npz of maps.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

BANDS = {
    "low": (0.0, 0.5),
    "diurnal": (0.8, 1.2),
    "semidiurnal": (1.7, 2.3),
}


def partition(series: np.ndarray, hours: float = 1.0) -> dict[str, np.ndarray]:
    """Fraction of detrended variance in each band, per cell. series: [time, cell]."""
    t = np.arange(series.shape[0], dtype=np.float64)
    design = np.stack([np.ones_like(t), t], 1)
    coefficients, *_ = np.linalg.lstsq(design, series, rcond=None)
    anomaly = series - design @ coefficients
    window = np.hanning(series.shape[0])[:, None]
    power = np.abs(np.fft.rfft(anomaly * window, axis=0)) ** 2
    power[0] = 0
    frequency = np.fft.rfftfreq(series.shape[0], d=hours / 24)  # cycles per day
    total = power.sum(0)
    result = {}
    for name, (lo, hi) in BANDS.items():
        result[name] = power[(frequency >= lo) & (frequency < hi)].sum(0) / total
    result["other_high"] = 1 - sum(result.values())
    six_hourly = anomaly[::6]
    days = six_hourly[: len(six_hourly) // 4 * 4].reshape(-1, 4, series.shape[1])
    within_day = days - days.mean(1, keepdims=True)
    result["six_hour_sub_daily"] = within_day.var((0, 1)) / days.reshape(
        -1, series.shape[1]
    ).var(0)
    result["std"] = anomaly.std(0)
    return result


def weighted(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values) & (weights > 0)
    return float((values[ok] * weights[ok]).sum() / weights[ok].sum())


def rmse(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    return weighted((a - b) ** 2, w) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llc", type=Path, required=True, help="LLC4320_samudra_1deg.zarr"
    )
    parser.add_argument(
        "--daily", type=Path, required=True, help="Prepared daily obs root"
    )
    parser.add_argument(
        "--starts",
        nargs="+",
        default=["2011-10-01", "2012-01-01", "2012-04-01", "2012-07-01"],
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ds = xr.open_zarr(args.llc)
    wet = ds["wet_area"].isel(k=0).values
    lat = np.broadcast_to(ds["lat"].values[:, None], wet.shape)
    cells = wet > 0
    weights = wet[cells]
    variables = {
        "Eta": ds["Eta"],
        "Theta_0": ds["Theta"].isel(k=0),
        "U_0": ds["U"].isel(k=0),
        "V_0": ds["V"].isel(k=0),
    }
    summary: dict = {"segments": [], "partition": {}, "eta_vs_adt": {}}
    maps: dict[str, Any] = {}
    differences: dict[str, list[np.ndarray]] = {"snapshot": [], "daily_mean": []}
    for start in args.starts:
        begin = pd.Timestamp(start)
        end = begin + pd.Timedelta(days=args.days)
        segment = slice(begin, end - pd.Timedelta(hours=1))
        hours = pd.DatetimeIndex(ds.time.sel(time=segment).values)
        if (
            len(hours) != args.days * 24
            or (
                np.diff(hours.values).astype("timedelta64[s]")
                != np.timedelta64(3600, "s")
            ).any()
        ):
            raise ValueError(
                f"Segment {start} is not {args.days} contiguous hourly days"
            )
        summary["segments"].append([str(hours[0]), str(hours[-1])])
        for name, variable in variables.items():
            values = variable.sel(time=segment).values.astype(np.float64)
            result = partition(values[:, cells])
            for key, value in result.items():
                maps.setdefault(f"{name}/{key}", []).append(value)
        eta = variables["Eta"].sel(time=segment).values
        for day in pd.date_range(begin, end - pd.Timedelta(days=1), freq="D"):
            path = args.daily / "adt" / str(day.year) / f"{day:%Y-%m-%d}.npz"
            with np.load(path) as saved:
                adt = saved["values"][0]
            index = (day - begin) // pd.Timedelta(hours=1)
            differences["snapshot"].append(eta[index] - adt)
            differences["daily_mean"].append(eta[index : index + 24].mean(0) - adt)
    for key, values in maps.items():
        mean = np.mean(values, 0)
        name, statistic = key.split("/")
        entry = summary["partition"].setdefault(name, {})
        entry[statistic] = {
            "global": weighted(mean, weights),
            "lat_bands": {
                f"{lo}..{lo + 10}": weighted(
                    np.where((lat[cells] >= lo) & (lat[cells] < lo + 10), mean, np.nan),
                    weights,
                )
                for lo in range(-80, 90, 10)
            },
        }
        full = np.full(wet.shape, np.nan)
        full[cells] = mean
        maps[key] = full
    for kind, items in differences.items():
        stack = np.stack(items)  # [date, y, x]
        support = np.isfinite(stack) & cells
        w = np.where(support, wet, 0)
        raw = np.sqrt(np.nansum(np.where(support, stack**2, 0) * w) / w.sum())
        offsets = np.nansum(np.where(support, stack, 0) * w, (1, 2)) / w.sum((1, 2))
        centered = stack - offsets[:, None, None]
        global_removed = np.sqrt(
            np.nansum(np.where(support, centered**2, 0) * w) / w.sum()
        )
        time_mean = np.nanmean(np.where(support, centered, np.nan), 0)
        anomaly = centered - time_mean
        datum_removed = np.sqrt(
            np.nansum(np.where(support, anomaly**2, 0) * w) / w.sum()
        )
        summary["eta_vs_adt"][kind] = {
            "dates": len(items),
            "mean_offset_m": float(offsets.mean()),
            "rmse_m": float(raw),
            "rmse_global_offset_removed_m": float(global_removed),
            "rmse_time_mean_map_removed_m": float(datum_removed),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tide-check.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        args.output / "tide-check-maps.npz",
        lat=ds["lat"].values,
        lon=ds["lon"].values,
        **{k.replace("/", "__"): v for k, v in maps.items()},
    )
    print(json.dumps(summary["eta_vs_adt"], indent=2))
    for name, entry in summary["partition"].items():
        print(name, {k: round(v["global"], 4) for k, v in entry.items()})


if __name__ == "__main__":
    main()
