# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only annual bias/anomaly diagnostics with training-only climatology."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from samudra.experiments.observation_pilot import atomic_json, digest


def weighted_summary(prediction, reference, climatology, area):
    valid = np.isfinite(reference) & (area > 0)
    if not valid.any() or not np.isfinite(prediction[valid]).all():
        raise ValueError("Missing prediction on observed support")
    if not np.isfinite(climatology[valid]).all():
        raise ValueError("Missing training climatology")
    w = np.where(valid, area, 0)
    w = w / w.sum()

    def average(a):
        return float(np.sum(np.where(valid, a, 0) * w))

    error = prediction - reference
    bias = average(error)
    p, r = prediction - climatology, reference - climatology
    pc, rc = p - average(p), r - average(r)
    denominator = np.sqrt(average(pc**2) * average(rc**2))
    return {
        "rmse": np.sqrt(average(error**2)),
        "spatial_mean_bias": bias,
        "bias_removed_rmse": np.sqrt(average((error - bias) ** 2)),
        "climatology_rmse": np.sqrt(average(r**2)),
        "prediction_anomaly_rms": np.sqrt(average(p**2)),
        "reference_anomaly_rms": np.sqrt(average(r**2)),
        "centered_anomaly_correlation": average(pc * rc) / denominator
        if denominator > 0
        else None,
    }


def temporal_decomposition(prediction, reference, area):
    valid = np.isfinite(reference) & (area > 0)
    if not valid.any() or not np.isfinite(prediction[valid]).all():
        raise ValueError("Missing prediction on observed support")
    error = np.where(valid, prediction - reference, 0)
    count = valid.sum(0)
    bias = error.sum(0) / np.maximum(count, 1)
    weight = valid * area
    total = weight.sum()
    mse = float((error**2 * weight).sum() / total)
    mean_mse = float((bias**2 * count * area).sum() / total)
    anomaly_mse = float((np.where(valid, error - bias, 0) ** 2 * weight).sum() / total)
    if not np.isclose(mse, mean_mse + anomaly_mse, rtol=1e-6, atol=1e-10):
        raise ValueError("Temporal error decomposition does not close")
    return {
        "mse": mse,
        "within_year_temporal_mean_bias_mse": mean_mse,
        "within_year_temporal_anomaly_error_mse": anomaly_mse,
        "temporal_mean_bias_fraction": mean_mse / mse if mse > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluation, data = Path(args.evaluation), Path(args.data)
    complete = json.loads((evaluation / "COMPLETE.json").read_text())
    grid, stats = np.load(data / "grid.npz"), np.load(data / "statistics.npz")
    area = np.cos(np.deg2rad(grid["lat"]))[:, None] * grid["mask"][0]
    area *= np.abs(grid["lat"][:, None]) <= 60
    result = {
        "evaluation_completion_sha256": digest(evaluation / "COMPLETE.json"),
        "statistics_sha256": digest(data / "statistics.npz"),
        "scope": "Read-only diagnostic; no selection changes. Climatology from training statistics. Within-year bias decomposition is a descriptive identity, not a fitted forecast.",
        "origins": {},
    }
    for origin, record in complete["origins"].items():
        arrays = np.load(evaluation / record["arrays"])
        prediction, reference = arrays["surface"], arrays["reference"][:, :2]
        dates = pd.date_range(
            pd.Timestamp(origin) + pd.Timedelta(days=2.5), periods=73, freq="5D"
        )
        climate = stats["surface_climatology"][dates.month.to_numpy() - 1]
        values = {}
        for channel, name in enumerate(["sst", "adt"]):
            values[name] = {
                "temporal_decomposition": temporal_decomposition(
                    prediction[:, channel], reference[:, channel], area
                ),
                "leads": {
                    str(day): weighted_summary(
                        prediction[day // 5 - 1, channel],
                        reference[day // 5 - 1, channel],
                        climate[day // 5 - 1, channel],
                        area,
                    )
                    for day in [5, 15, 30, 90, 180, 365]
                },
            }
        result["origins"][origin] = values
    atomic_json(result, Path(args.output))


if __name__ == "__main__":
    main()
