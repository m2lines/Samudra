# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare corrected and ordinary sample-mean MSE in the synthetic task."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with (args.root / "learning-curves.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    names = ["fair-plus-mean-mse", "fair-plus-ustat-mean-mse"]
    labels = ["CRPS + sample-mean MSE", "CRPS + corrected mean MSE"]
    keys = [
        "jet_mean_rmse",
        "expected_crps",
        "expected_90_coverage",
        "conditional_90_coverage_mae",
    ]
    titles = [
        "Jet conditional-mean RMSE",
        "Expected population CRPS",
        "Expected coverage of central 90% interval",
        "Mean absolute conditional coverage error",
    ]
    for name, label in zip(names, labels, strict=True):
        selected = [r for r in rows if r["objective"] == name]
        steps = sorted(set(int(r["updates"]) for r in selected))
        for ax, key in zip(axes.flat, keys, strict=True):
            values = np.array(
                [
                    [float(r[key]) for r in selected if int(r["updates"]) == step]
                    for step in steps
                ]
            )
            line = ax.plot(steps, values.mean(1), marker="o", label=label)[0]
            ax.fill_between(
                steps, values.min(1), values.max(1), alpha=0.15, color=line.get_color()
            )
    axes[1, 0].axhline(0.9, color="black", linewidth=1, label="Nominal 90%")
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set_title(title)
        ax.set_xlabel("Updates")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(
        "Synthetic task: four members, coefficient 1, three-seed means and ranges"
    )
    for suffix in ["png", "pdf"]:
        path = args.root / f"comparison.{suffix}"
        fig.savefig(path, dpi=150)
        Path(str(path) + ".license").write_text(
            "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
        )


if __name__ == "__main__":
    main()
