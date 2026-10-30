# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot CPU toy mechanism checks; all plotted scores are synthetic."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.root / "toy/learning-curves.csv")
    names = list(dict.fromkeys(row["objective"] for row in rows))
    labels = [
        "Member MSE",
        "MAE",
        "Fair CRPS",
        "Fair CRPS + mean MSE",
        "Almost-fair (alpha .95)",
    ]
    colors = plt.get_cmap("tab10")(np.arange(10))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    summary = []
    for index, (name, label) in enumerate(zip(names, labels, strict=True)):
        selected = [row for row in rows if row["objective"] == name]
        steps = sorted(set(int(row["updates"]) for row in selected))
        for ax, metric in zip(axes[0], ["jet_mean_rmse", "expected_crps"], strict=True):
            values = np.array(
                [
                    [
                        float(row[metric])
                        for row in selected
                        if int(row["updates"]) == step
                    ]
                    for step in steps
                ]
            )
            ax.plot(steps, values.mean(1), marker="o", color=colors[index], label=label)
            ax.fill_between(
                steps, values.min(1), values.max(1), color=colors[index], alpha=0.12
            )
        fields = read(args.root / "toy" / f"fields-{name}-11.csv")
        x = [float(row["x"]) for row in fields]
        for ax, metric in zip(
            axes[1], ["predicted_mean", "predicted_std"], strict=True
        ):
            ax.plot(
                x,
                [float(row[metric]) for row in fields],
                color=colors[index],
                label=label,
            )
        final = [row for row in selected if int(row["updates"]) == 2000]
        entry = dict(objective=name)
        for metric in [
            "mean_rmse",
            "jet_mean_rmse",
            "background_mean_rmse",
            "spread_rms",
            "expected_crps",
            "peak_mean",
        ]:
            final_values = [float(row[metric]) for row in final]
            entry[metric] = dict(
                mean=float(np.mean(final_values)),
                minimum=min(final_values),
                maximum=max(final_values),
            )
        summary.append(entry)
    for ax, metric in zip(axes[1], ["true_mean", "true_std"], strict=True):
        ax.plot(
            x,
            [float(row[metric]) for row in fields],
            color="black",
            linewidth=2,
            label="Truth",
        )
        ax.set_xlim(0.6, 0.9)
        ax.set_xlabel("Synthetic coordinate")
    axes[0, 0].set_title("Jet-region conditional-mean RMSE")
    axes[0, 1].set_title("Expected population CRPS (analytic Gaussian)")
    axes[1, 0].set_title("Conditional mean after 2k updates; seed 11")
    axes[1, 1].set_title("Conditional standard deviation; seed 11")
    for ax in axes[0]:
        ax.set_xlabel("Updates")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(
        "Synthetic narrow-current task: four training members, three seeds\nCurves show seed means and ranges; this is not an ocean model"
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(args.root / f"toy-loss-comparison.{suffix}", dpi=150)
    (args.root / "toy/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    long_rows = read(args.root / "toy-long/learning-curves.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for name, color in [("mae", colors[1]), ("fair-crps", colors[2])]:
        selected = [row for row in long_rows if row["objective"] == name]
        steps = sorted(set(int(row["updates"]) for row in selected))
        for ax, metric in zip(
            axes, ["jet_mean_rmse", "spread_rms", "expected_crps"], strict=True
        ):
            values = np.array(
                [
                    [
                        float(row[metric])
                        for row in selected
                        if int(row["updates"]) == step
                    ]
                    for step in steps
                ]
            )
            ax.plot(steps, values.mean(1), marker="o", color=color, label=name)
            ax.fill_between(
                steps, values.min(1), values.max(1), color=color, alpha=0.12
            )
    axes[0].set_title("Jet conditional-mean RMSE")
    axes[1].set_title("Predicted conditional spread RMS")
    axes[1].axhline(
        float(long_rows[0]["true_spread_rms"]), color="black", label="True spread"
    )
    axes[2].set_title("Expected population CRPS")
    for ax in axes:
        ax.set_xlabel("Updates")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Longer training on the same synthetic task; three-seed means and ranges"
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(args.root / f"toy-longer-training.{suffix}", dpi=150)


if __name__ == "__main__":
    main()
