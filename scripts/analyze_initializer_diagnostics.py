#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize precision and field-fitting runs without selecting on test data."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def weighted_mean(value, weights):
    return float(np.sum(value.astype(np.float64) * weights) / weights.sum())


def summarize(root, output):
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        for table in directory.glob("*.csv"):
            if table.name == "utilization.csv":
                continue
            rows = pd.read_csv(table)
            if "field_mse" not in rows:
                continue
            maps = table.with_name(table.stem + "-maps.npz")
            with np.load(maps) as data:
                std = float(data["std"])
            a2 = rows.true_anomaly_m2.mean()
            b2 = rows.pred_anomaly_m2.mean()
            cross = rows.cross_anomaly.mean()
            # Correlation within each spatial field, then averaged across dates.
            covariance = (
                rows.cross_anomaly - rows.pred_anomaly_mean * rows.true_anomaly_mean
            )
            variance_product = (rows.pred_anomaly_m2 - rows.pred_anomaly_mean**2) * (
                rows.true_anomaly_m2 - rows.true_anomaly_mean**2
            )
            corr = covariance / np.sqrt(variance_product.clip(lower=0))
            summaries.append(
                dict(
                    run=directory.name,
                    split=rows.split.iloc[0],
                    precision=rows.precision.iloc[0],
                    origins=len(rows),
                    physical_rmse=np.sqrt(rows.current_field_mse.mean()) * std,
                    climatology_rmse=np.sqrt(rows.climatology_mse.mean()) * std,
                    mse_skill=1
                    - rows.current_field_mse.mean() / rows.climatology_mse.mean(),
                    spatial_correlation=corr.mean(),
                    anomaly_rms_ratio=np.sqrt(b2 / a2),
                    anomaly_cosine=cross / np.sqrt(a2 * b2),
                    physical_bias=rows.bias.mean() * std,
                    both_times_field_mse=rows.field_mse.mean(),
                    ts_rmse=np.sqrt(rows.ts_mse.mean()),
                )
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    precision = []
    folder = root / "precision"
    if folder.is_dir():
        for split in ["train", "validation"]:
            with (
                np.load(folder / f"{split}-bf16-maps.npz") as bf,
                np.load(folder / f"{split}-fp32-maps.npz") as fp,
            ):
                weights = bf["mask"] * np.cos(np.deg2rad(bf["latitude"]))[:, None]
                std = float(bf["std"])
                for key in bf.files:
                    if not key.endswith("_prediction"):
                        continue
                    idx = key.split("_")[0]
                    np.testing.assert_array_equal(
                        bf[idx + "_truth"], fp[idx + "_truth"]
                    )
                    b, f, t = bf[key], fp[key], bf[idx + "_truth"]
                    precision.append(
                        dict(
                            split=split,
                            date=str(bf[idx + "_date"]),
                            paired_output_rmse=np.sqrt(
                                weighted_mean((b - f) ** 2, weights)
                            )
                            * std,
                            bf16_rmse=np.sqrt(weighted_mean((b - t) ** 2, weights))
                            * std,
                            fp32_rmse=np.sqrt(weighted_mean((f - t) ** 2, weights))
                            * std,
                        )
                    )
        pd.DataFrame(precision).to_csv(output / "precision-pairs.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    progress: list[dict] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name == "smoke":
            continue
        path = directory / "progress.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        checks = [
            r
            for r in records
            if r.get("event") in ["baseline", "validation"] and "selection" in r
        ]
        if not checks:
            continue
        protocol = json.loads((directory / "diagnostic-protocol.json").read_text())
        ax = axes[0 if len(protocol["population"]) <= 16 else 1]
        x = [r.get("step", 0) for r in checks]
        y = [np.sqrt(r["selection"]["field_mse"]) for r in checks]
        ax.plot(x, y, marker="o", label=directory.name)
        progress.extend(
            {"run": directory.name, "step": i, "selection_rmse": v}
            for i, v in zip(x, y, strict=True)
        )
    for ax, title in zip(
        axes,
        ["Fitting-set error (memorization)", "Validation error (generalization)"],
        strict=True,
    ):
        ax.set(
            title=title,
            xlabel="Optimizer updates",
            ylabel="Normalized so_9 RMSE, both times",
            yscale="log",
        )
        if ax.lines:
            ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output / "learning.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(progress).to_csv(output / "learning.csv", index=False)
    print(summary.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    summarize(args.raw, args.output)


if __name__ == "__main__":
    main()
