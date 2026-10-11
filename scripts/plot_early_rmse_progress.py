#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot saved validation events; never run inference or select on held-out data."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPONENTS = [
    "sst_rmse",
    "velocity_rmse",
    "eke_rmse",
    "ohc_0_700_rmse",
    "ohc_700_2000_rmse",
]
DYNAMICS = {
    "sst_dynamics/day30/bias_celsius": "Day-30 SST bias (°C)",
    "sst_dynamics/day30/anomaly_correlation": "Day-30 SST anomaly correlation",
    "sst_dynamics/day30/anomaly_rms_ratio": "Day-30 SST anomaly RMS ratio",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.root / "paths.json").read_text())
    assert config["selection_metric"] == "rmse-only"
    control = json.loads((args.root / "selection-reference.json").read_text())[
        "control"
    ]["metrics"]
    rows = []
    for name, run in config["runs"].items():
        for path in sorted((args.root / name).glob("validation-*.json")):
            event = json.loads(path.read_text())
            m = event["metrics"]["metrics"]
            ratios = [m[k] / control[k] for k in COMPONENTS]
            if not np.isfinite(ratios).all() or not np.isclose(
                np.mean(ratios), event["score"], rtol=0, atol=1e-12
            ):
                raise ValueError(
                    "Saved event differs from fixed RMSE selection: " + str(path)
                )
            rows.append(
                dict(
                    model=name,
                    seed=run["seed"],
                    update=event["global_step"],
                    observation_updates=event["task_counts"]["observation"],
                    om4_updates=event["task_counts"]["om4"],
                    score=event["score"],
                    **{k: m[k] for k in COMPONENTS},
                    **{k: m[k] for k in DYNAMICS},
                    om4_retention_ts_mse=event["om4_retention"]["ts_mse"],
                )
            )
    if not rows:
        raise ValueError("No completed validation events yet")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "validation.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, axis, label in zip(
        axes,
        ["update", "observation_updates"],
        ["Total optimizer updates", "Observation updates"],
        strict=True,
    ):
        for name in config["runs"]:
            series = [r for r in rows if r["model"] == name]
            if series:
                ax.plot(
                    [r[axis] for r in series],
                    [r["score"] for r in series],
                    label=name,
                    marker="o",
                    markersize=3,
                    lw=1.3,
                )
        ax.set(xlabel=label, ylabel="Validation RMSE blend (lower is better)")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / ("validation-blend." + ext), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    diagnostics = dict(DYNAMICS, om4_retention_ts_mse="OM4 held-out T/S MSE")
    for ax, (key, label) in zip(axes.flat, diagnostics.items(), strict=True):
        for name in config["runs"]:
            series = [r for r in rows if r["model"] == name]
            if series:
                ax.plot(
                    [r["update"] for r in series],
                    [r[key] for r in series],
                    label=name,
                    lw=1.2,
                )
        if key.endswith("bias_celsius"):
            ax.axhline(0, color="gray", ls=":", lw=1)
        elif key.endswith("anomaly_rms_ratio"):
            ax.axhline(1, color="gray", ls=":", lw=1)
        ax.set(xlabel="Total optimizer updates", ylabel=label)
        ax.grid(alpha=0.25)
    axes.flat[-1].legend(fontsize=7)
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / ("validation-dynamics." + ext), dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, component in zip(axes.flat, COMPONENTS, strict=False):
        for name in config["runs"]:
            series = [r for r in rows if r["model"] == name]
            if series:
                ax.plot(
                    [r["update"] for r in series],
                    [r[component] / control[component] for r in series],
                    label=name,
                    marker="o",
                    markersize=3,
                    lw=1.2,
                )
        ax.set(
            title=component,
            xlabel="Total optimizer updates",
            ylabel="RMSE / fixed climatology RMSE",
        )
        ax.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        *axes.flat[0].get_legend_handles_labels(), loc="center", fontsize=8
    )
    for ext in ["png", "pdf"]:
        fig.savefig(args.output / ("validation-components." + ext), dpi=160)
    plt.close(fig)
    print(f"Plotted {len(rows)} verified validation events")


if __name__ == "__main__":
    main()
