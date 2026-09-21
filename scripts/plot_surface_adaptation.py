#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Render the adaptation report from audited tables and original validation logs."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from samudra.experiments.surface_adaptation_analysis import read_bytes, read_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "A": "#d99000",
    "B": "#008c77",
    "C": "#2467af",
    "D": "#799fd1",
    "E": "#b33a3a",
}
LABELS = {
    "A": "A: initializer only",
    "B": "B: evolution only",
    "C": "C: joint, 1e-5",
    "D": "D: joint, alternate order",
    "E": "E: joint, 1e-4",
}
LICENSE = "SPDX-FileCopyrightText: 2026 Samudra Authors\n\nSPDX-License-Identifier: CC-BY-4.0\n"


def save(fig, output, name):
    for suffix in ["png", "svg"]:
        path = output / (name + "." + suffix)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        Path(str(path) + ".license").write_text(LICENSE)
    plt.close(fig)


def plot(raw, analysis, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(analysis / "grouped_metrics.csv")
    comparisons = read_csv(analysis / "paired_comparisons.csv")
    arms = sorted({row["arm"] for row in rows})
    lookup = {
        (r["arm"], r["mode"], r["region"], int(r["lead_days"]), r["variable"]): float(
            r["normalized_rmse"]
        )
        for r in rows
    }
    leads = [5, 10, 15, 20, 25, 30]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, variable, title in zip(
        axes, ["thetao", "so"], ["Subsurface temperature", "Salinity"], strict=True
    ):
        for arm in arms:
            ax.plot(
                leads,
                [lookup[arm, "inferred", "global", lead, variable] for lead in leads],
                color=COLORS[arm],
                label=LABELS[arm],
                linestyle="--" if arm == "D" else "-",
                marker="o",
                markersize=3,
            )
        for mode, label, color, style in [
            ("frozen_pretrain", "Untuned pair", "#222222", "--"),
            ("wave1_joint", "Selected wave-1 joint", "#777777", ":"),
        ]:
            ax.plot(
                leads,
                [lookup["A", mode, "global", lead, variable] for lead in leads],
                label=label,
                color=color,
                linestyle=style,
            )
        ax.set(
            title=title, xlabel="Lead (days)", ylabel="Normalized RMSE", xticks=leads
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "OM4 held-out hindcasts: 99 origins, five-day states\nEqual-depth errors; this is not observational skill"
    )
    save(fig, output, "forecast_skill")

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True
    )
    for ax, mode, title in zip(
        axes,
        ["frozen_pretrain", "wave1_joint"],
        ["Versus untuned pair", "Versus selected wave-1 joint"],
        strict=True,
    ):
        for y, arm in enumerate(arms):
            row = next(
                r
                for r in comparisons
                if r["arm"] == arm
                and r["reference_arm"] == arm
                and r["reference_mode"] == mode
                and r["region"] == "global"
                and r["lead_days"] == "30"
                and r["variable"] == "ts"
            )
            value, low, high = [
                float(row[k])
                for k in ["rmse_reduction_pct", "ci95_low_pct", "ci95_high_pct"]
            ]
            ax.errorbar(
                value,
                y,
                xerr=[[value - low], [high - value]],
                fmt="o",
                color=COLORS[arm],
                capsize=3,
            )
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.set(
            title=title,
            xlabel="30-day T/S RMSE reduction (%)",
            yticks=range(len(arms)),
            yticklabels=[LABELS[a] for a in arms],
        )
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Paired calendar-year bootstrap, 95% descriptive intervals\nNine partly sampled years; intervals condition on these trained models",
        fontsize=11,
    )
    save(fig, output, "paired_improvement")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fields = [
        ("ts_mse", "Forecast T/S (selection score)"),
        ("forecast_full_mse", "Full-variable forecast"),
        ("reconstruction_ts_mse", "Initial reconstruction T/S"),
        ("reconstruction_full_mse", "Full interior reconstruction"),
    ]
    for arm in arms:
        progress = [
            json.loads(line)
            for line in read_bytes(raw / arm / "progress.jsonl").decode().splitlines()
        ]
        validation = [
            r for r in progress if r.get("phase") == "adapt" and "ts_mse" in r
        ]
        chosen = min(validation, key=lambda r: r["ts_mse"])
        for ax, (field, title) in zip(axes.flat, fields, strict=True):
            ax.plot(
                [r.get("elapsed_seconds", 0) / 3600 for r in validation],
                [r[field] for r in validation],
                color=COLORS[arm],
                label=LABELS[arm],
                linestyle="--" if arm == "D" else "-",
                marker=".",
                markersize=4,
            )
            ax.scatter(
                [chosen.get("elapsed_seconds", 0) / 3600],
                [chosen[field]],
                color=COLORS[arm],
                marker="*",
                s=65,
                zorder=3,
            )
            ax.set(
                title=title,
                xlabel="Training-phase elapsed hours",
                ylabel="Normalized MSE",
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Validation over 11 origins; stars mark the T/S-selected checkpoint\nElapsed phase time includes cache preparation and checkpoint work",
        fontsize=11,
    )
    save(fig, output, "validation_curves")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for arm in arms:
        manifest = json.loads((raw / arm / "manifest.json").read_text())
        channels = read_csv(raw / arm / "heldout_metrics.csv")
        channel_lookup = {
            (r["mode"], r["region"], int(r["lead_days"]), r["channel"]): float(
                r["normalized_rmse"]
            )
            for r in channels
        }
        for ax, variable, title in zip(
            axes, ["thetao", "so"], ["Subsurface temperature", "Salinity"], strict=True
        ):
            names = [
                n
                for n in manifest["channels"]
                if n.startswith(variable + "_") and n != "thetao_0"
            ]
            depths = [manifest["depths_m"][int(n.split("_")[1])] for n in names]
            improvement = [
                100
                * (
                    1
                    - channel_lookup["inferred", "global", 30, n]
                    / channel_lookup["frozen_pretrain", "global", 30, n]
                )
                if channel_lookup["frozen_pretrain", "global", 30, n] > 0
                else np.nan
                for n in names
            ]
            ax.plot(
                improvement,
                depths,
                color=COLORS[arm],
                label=LABELS[arm],
                linestyle="--" if arm == "D" else "-",
                marker=".",
                markersize=4,
            )
            ax.set(
                title=title,
                xlabel="30-day RMSE reduction vs untuned pair (%)",
                ylabel="Depth (m)",
                yscale="log",
            )
    for ax in axes:
        ax.invert_yaxis()
        ax.axvline(0, color="#777777", linewidth=0.8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Depth diagnostic: wet-cell area-weighted error at each level\nDeep levels cover fewer wet cells; the selection score is not volume-weighted",
        fontsize=11,
    )
    save(fig, output, "depth_skill")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.raw, args.analysis, args.output)


if __name__ == "__main__":
    main()
