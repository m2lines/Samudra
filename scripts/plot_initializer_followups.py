#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot independent seeds and matched joint adaptation with persistence controls."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LICENSE = "SPDX-FileCopyrightText: 2026 Samudra Authors\n\nSPDX-License-Identifier: CC-BY-4.0\n"


def curve(ax, data, run, mode, variable, label, color, marker, style="-", hollow=False):
    rows = data[
        (data.run == run)
        & (data["mode"] == mode)
        & (data.variable == variable)
        & (data.region == "global")
    ].sort_values("lead_days")
    if rows.lead_days.tolist() != list(range(0, 31, 5)):
        raise ValueError(f"Missing or duplicated curve: {run}/{mode}/{variable}")
    ax.plot(
        rows.lead_days,
        rows.normalized_rmse,
        label=label,
        color=color,
        marker=marker,
        linestyle=style,
        markerfacecolor="none" if hollow else color,
        markersize=4,
    )


def finish(fig, axes, output, name, title):
    for ax in axes:
        ax.set(
            xlabel="Lead (days)",
            ylabel="Normalized T/S RMSE",
            xticks=list(range(0, 31, 5)),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, handlelength=4)
    fig.suptitle(title)
    path = output / (name + ".png")
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)
    if path.stat().st_size > 240 * 1024:
        raise ValueError(f"Oversized figure: {path}")
    Path(str(path) + ".license").write_text(LICENSE)


def plot(summary, output):
    data = pd.read_csv(summary)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, arm, title in zip(
        axes,
        "AD",
        ["A: 30M U-Net, short history", "D: 121M U-Net, expanded history"],
        strict=True,
    ):
        ax.set_title(title)
        curve(
            ax,
            data,
            arm,
            "inferred",
            "ts",
            "Seed 1729 + fixed evolution",
            "#2166ac",
            "o",
        )
        curve(
            ax,
            data,
            arm + "-s2718",
            "inferred",
            "ts",
            "Seed 2718 + fixed evolution",
            "#b33a3a",
            "s",
            "--",
            True,
        )
        curve(
            ax,
            data,
            arm,
            "true",
            "ts",
            "True interior + fixed evolution",
            "#333333",
            "^",
        )
        curve(
            ax,
            data,
            arm,
            "inferred_persistence",
            "ts",
            "Seed 1729 interior, hold fixed",
            "#b8860b",
            "x",
            ":",
        )
    finish(
        fig,
        axes,
        output,
        "replicas",
        "Independent initializer seeds: same 99 held-out origins",
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, arm in zip(axes, "AD", strict=True):
        ax.set_title("Arm " + arm)
        curve(
            ax,
            data,
            arm,
            "inferred",
            "ts",
            "Initializer + fixed evolution",
            "#2166ac",
            "o",
        )
        curve(
            ax,
            data,
            arm + "-joint",
            "inferred",
            "ts",
            "Joint initializer + adapted evolution",
            "#b33a3a",
            "s",
            "--",
            True,
        )
        curve(
            ax,
            data,
            arm + "-joint",
            "inferred_persistence",
            "ts",
            "Joint initializer, hold fixed",
            "#b8860b",
            "x",
            ":",
        )
        curve(
            ax,
            data,
            arm,
            "true",
            "ts",
            "True interior + fixed evolution",
            "#333333",
            "^",
        )
        curve(
            ax,
            data,
            arm + "-joint",
            "true",
            "ts",
            "True interior + adapted evolution",
            "#258444",
            "D",
            "--",
            True,
        )
        curve(
            ax,
            data,
            arm,
            "true_persistence",
            "ts",
            "True interior, hold fixed",
            "#777777",
            "+",
            "-.",
        )
    finish(
        fig,
        axes,
        output,
        "joint-and-controls",
        "Matched three-hour joint adaptation: independent 0–30-day windows",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    plot(a.summary, a.output)


if __name__ == "__main__":
    main()
