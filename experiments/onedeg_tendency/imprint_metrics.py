# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Paired imprinting diagnostics on the exact same OM4 grid and timestamps.

These associations diagnose imprinting; they are not causal attribution.
Report physical RMSE and retained variance alongside velocity-error coupling
so a model cannot pass merely by becoming quiescent. The observation suite
independently checks surface and integrated-ocean skill.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from samudra.constants import build_om4_layout
from samudra.metrics.kernels import detrended_deseasonalized
from samudra.metrics.run import analysis_ready


def open_data(path):
    data = analysis_ready(xr.open_zarr(path, chunks={}), build_om4_layout())
    if not np.issubdtype(data.time.dtype, np.datetime64):
        data = data.assign_coords(
            time=np.array([str(t) for t in data.time.values], dtype="datetime64[ns]")
        )
    return data


def mean(value, area):
    """Area-weighted spatial mean, followed by equal weighting of time."""
    return value.weighted(area.fillna(0)).mean(("y", "x")).mean()


def correlation(a, b, weights):
    valid = np.isfinite(a) & np.isfinite(b)
    weights = weights.where(valid, 0).broadcast_like(a)
    total = weights.sum()
    if float(total) <= 0:
        return float("nan")
    a = a.fillna(0)
    b = b.fillna(0)
    a = a - (a * weights).sum() / total
    b = b - (b * weights).sum() / total
    covariance = (a * b * weights).sum()
    denominator = np.sqrt((a**2 * weights).sum() * (b**2 * weights).sum())
    return float(covariance / denominator) if float(denominator) > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--run", action="append", required=True, help="label=store")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    truth = open_data(args.truth)
    runs = {
        label: open_data(path) for label, path in (r.split("=", 1) for r in args.run)
    }
    # Require complete paired coverage: no nearest-neighbour temporal matching.
    times = next(iter(runs.values())).time.sel(time=slice("2015-01-01", "2022-12-31"))
    if len(times) < 500:
        raise ValueError("Expected an eight-year, five-day rollout")
    truth = truth.sel(time=times)
    runs = {label: data.sel(time=times) for label, data in runs.items()}
    for label, data in runs.items():
        for coord in ("time", "y", "x", "lev"):
            xr.testing.assert_equal(data[coord], truth[coord])
    base_area = xr.DataArray(
        np.cos(np.deg2rad(truth.y)), dims="y", coords={"y": truth.y}
    ).broadcast_like(truth.wetmask.isel(lev=0, drop=True))
    rows = []
    maps = {}
    trajectories = {}
    layout = build_om4_layout()
    for level, depth in enumerate(layout.depth_levels):
        wet = truth.wetmask.isel(lev=level).astype(bool)
        area = base_area.where(wet, 0)
        for variable in ("thetao", "so", "uo", "vo"):
            reference = (
                truth[variable].isel(lev=level).where(wet).astype("float64").load()
            )
            reference_residual = detrended_deseasonalized(reference)
            reference_var = reference_residual.var("time")
            for label, data in runs.items():
                prediction = (
                    data[variable].isel(lev=level).where(wet).astype("float64").load()
                )
                if bool((~np.isfinite(prediction.where(wet, 0))).any()):
                    raise ValueError(
                        f"Nonfinite wet prediction: {label}/{variable}/{level}"
                    )
                residual = detrended_deseasonalized(prediction)
                error = prediction - reference
                error_residual = residual - reference_residual
                variance = residual.var("time")
                row = {
                    "run": label,
                    "variable": variable,
                    "level": level,
                    "depth_m": depth,
                    "n_times": len(times),
                    "rmse": float(np.sqrt(mean(error**2, area))),
                    "bias": float(mean(error, area)),
                    "anomaly_rmse": float(np.sqrt(mean(error_residual**2, area))),
                    "anomaly_r2": float(
                        1
                        - mean(error_residual**2, area)
                        / mean(reference_residual**2, area)
                    ),
                    "anomaly_variance_ratio": float(
                        mean(variance, area) / mean(reference_var, area)
                    ),
                    "error_increment_rms": float(
                        np.sqrt(mean(error.diff("time") ** 2, area))
                    ),
                    "error_increment_rms_over_truth": float(
                        np.sqrt(
                            mean(error.diff("time") ** 2, area)
                            / mean(reference.diff("time") ** 2, area)
                        )
                    ),
                    "variance_pattern_correlation": correlation(
                        variance, reference_var, area
                    ),
                }
                if variable in ("thetao", "so"):
                    for velocity in ("uo", "vo"):
                        fast = (
                            truth[velocity]
                            .isel(lev=level)
                            .where(wet)
                            .astype("float64")
                            .load()
                        )
                        row[f"error_increment_{velocity}_increment_corr"] = correlation(
                            error_residual.diff("time"),
                            detrended_deseasonalized(fast).diff("time"),
                            area,
                        )
                        fast_var = detrended_deseasonalized(fast).var("time")
                        row[f"excess_variance_{velocity}_variance_corr"] = correlation(
                            variance - reference_var, fast_var, area
                        )
                    # Independent maps and trajectories at every depth make
                    # banding, over-smoothing, and high-frequency noise reviewable.
                    maps[f"{label}_{variable}_{level}"] = variance.drop_vars(
                        "lev", errors="ignore"
                    )
                    maps[f"truth_{variable}_{level}"] = reference_var.drop_vars(
                        "lev", errors="ignore"
                    )
                    trajectories[f"{label}_{variable}_{level}"] = (
                        prediction.weighted(area)
                        .mean(("y", "x"))
                        .drop_vars("lev", errors="ignore")
                    )
                    trajectories[f"truth_{variable}_{level}"] = (
                        reference.weighted(area)
                        .mean(("y", "x"))
                        .drop_vars("lev", errors="ignore")
                    )
                rows.append(row)
                print(json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(args.output / "depth_metrics.csv", index=False)
    xr.Dataset(maps).to_zarr(args.output / "variance_maps.zarr", mode="w")
    xr.Dataset(trajectories).to_zarr(
        args.output / "global_mean_trajectories.zarr", mode="w"
    )
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharey=True)
    for i, variable in enumerate(("thetao", "so")):
        for j, metric in enumerate(
            ("anomaly_r2", "anomaly_variance_ratio", "error_increment_rms_over_truth")
        ):
            for label in runs:
                subset = frame[(frame.run == label) & (frame.variable == variable)]
                axes[i, j].plot(subset[metric], subset.depth_m, label=label)
            axes[i, j].set_title(f"{variable}: {metric}")
            axes[i, j].set_ylabel("Depth (m)")
            axes[i, j].grid(alpha=0.3)
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(args.output / "imprinting_depth_summary.png", dpi=160)
    plt.close(fig)
    metadata = {
        "truth": args.truth,
        "runs": args.run,
        "start": str(times.values[0]),
        "end": str(times.values[-1]),
        "spatial_weights": "cos(latitude), wet cells only",
        "anomalies": "samudra.metrics.kernels detrended_deseasonalized: pentad climatology then linear trend removal",
        "interpretation": "Single-seed joint intervention; correlations are diagnostics, not evidence of causal direction. Compare variance retention and velocity skill with imprinting indicators.",
    }
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
