#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot fixed-checkpoint annual RMSE results and matched surface maps."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.export_presentation_annual_maps import GROUPS
from scripts.plot_observation_rollout_maps import plot as plot_maps


def plot(folder, output):
    output.mkdir(parents=True, exist_ok=True)
    result = json.loads((folder / "results.json").read_text())
    frame = pd.read_csv(folder / "rmse-scores.csv")
    pooled = frame[frame.origin == "pooled"]
    names = list(dict.fromkeys(n for group in GROUPS.values() for n in group))
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, day in zip(axes, (30, 365), strict=True):
        values = pooled[pooled.lead_days == day].set_index("model").rmse_score
        prediction = values.loc[names].to_numpy()
        persistence = values.loc[
            [n + " / initialized persistence" for n in names]
        ].to_numpy()
        y = np.arange(len(names))
        ax.hlines(y, prediction, persistence, color="0.7", lw=1)
        ax.scatter(prediction, y, label="Evolved forecast", color="C0", zorder=3)
        ax.scatter(
            persistence, y, marker="|", s=65, color="0.4", label="Own persistence"
        )
        ax.axvline(1, color="black", ls=":", label="Training climatology")
        ax.set_title(f"Day {day}")
        ax.set_xlabel("Four-component RMSE score (lower is better)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_yticks(y, names, fontsize=8)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Same selected checkpoints · three January starts · matched RMSE controls"
    )
    fig.tight_layout()
    fig.savefig(output / "rmse-comparison.png", dpi=160)
    fig.savefig(output / "rmse-comparison.pdf")
    plt.close(fig)

    origins = result["configuration"]["origins"]
    leads = [5, 15, 30, 90, 180, 365]
    metrics = result["metrics"]
    for group, models in GROUPS.items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        labels = models + [
            "U-global / initialized persistence",
            "Training seasonal climatology",
        ]
        for label in labels:
            records = metrics[label]
            style = (
                {"color": "0.5", "ls": "--"}
                if "persistence" in label
                else {"color": "black", "ls": ":"}
                if "climatology" in label
                else {}
            )
            for ax, metric, unit in zip(
                axes,
                ("sst_rmse", "velocity_rmse", "adt_rmse"),
                ("SST RMSE (°C)", "Geostrophic velocity RMSE (m/s)", "ADT RMSE (m)"),
                strict=True,
            ):
                curve = [
                    np.sqrt(
                        np.mean(
                            [records[o]["leads"][str(d)][metric] ** 2 for o in origins]
                        )
                    )
                    for d in leads
                ]
                ax.plot(leads, curve, marker=".", label=label, **style)
                ax.set(xlabel="Lead (days)", ylabel=unit)
                ax.grid(alpha=0.2)
        handles, legend = axes[-1].get_legend_handles_labels()
        fig.legend(handles, legend, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle(f"{group}: equal-origin MSE pooled before square root")
        fig.tight_layout(rect=(0, 0.20, 1, 0.95))
        fig.savefig(output / f"lead-curves-{group}.png", dpi=160)
        fig.savefig(output / f"lead-curves-{group}.pdf")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for label in labels:
            records = metrics[label]
            climatology = "climatology" in label
            months = [1, 12] if climatology else list(range(1, 13))
            style = (
                {"color": "black", "ls": "none", "marker": "x"}
                if climatology
                else {"color": "0.5", "ls": "--", "marker": "."}
                if "persistence" in label
                else {"marker": "."}
            )
            for ax, layer in zip(axes, ("0_700", "700_2000"), strict=True):
                curve = [
                    np.sqrt(
                        np.mean(
                            [
                                records[o]["monthly_ohc"][f"{o[:4]}-{m:02d}"][layer]
                                ** 2
                                for o in origins
                            ]
                        )
                    )
                    / 1e8
                    for m in months
                ]
                ax.plot(months, curve, label=label, **style)
                ax.set(
                    xlabel="Forecast calendar month (January start)",
                    ylabel=f"OHC {layer.replace('_', '–')} m RMSE (10⁸ J/m²)",
                    xticks=[1, 3, 6, 9, 12],
                )
                ax.grid(alpha=0.2)
        handles, legend = axes[-1].get_legend_handles_labels()
        fig.legend(handles, legend, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle(f"{group}: full calendar-month means; equal-origin MSE pooling")
        fig.tight_layout(rect=(0, 0.20, 1, 0.95))
        fig.savefig(output / f"ohc-curves-{group}.png", dpi=160)
        fig.savefig(output / f"ohc-curves-{group}.pdf")
        plt.close(fig)

    for bundle in sorted((folder / "maps").glob("*.npz")):
        plot_maps(bundle, output / "maps" / bundle.stem, global_observations=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.input, args.output)
