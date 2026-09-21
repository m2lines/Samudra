#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Render primary initializer comparisons from audited numerical tables."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "A": "30M U-Net, short",
    "B": "30M U-Net, expanded",
    "C": "121M U-Net, short",
    "D": "121M U-Net, expanded",
    "E": "122M attention, short",
    "F": "122M attention, expanded",
}
COLORS = {
    "A": "#2166ac",
    "B": "#2166ac",
    "C": "#b33a3a",
    "D": "#b33a3a",
    "E": "#258444",
    "F": "#258444",
}


def plot(analysis, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(Path(analysis) / "summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, variable, title in zip(
        axes,
        ["thetao", "so", "ts"],
        ["Subsurface temperature", "Salinity", "Combined T/S"],
        strict=True,
    ):
        selected = data[(data["region"] == "global") & (data["variable"] == variable)]
        for arm, label in LABELS.items():
            rows = selected[
                (selected["run"] == arm) & (selected["mode"] == "inferred")
            ].sort_values("lead_days")
            if rows.empty:
                continue
            expanded = arm in "BDF"
            ax.plot(
                rows["lead_days"],
                rows["normalized_rmse"],
                label=label,
                color=COLORS[arm],
                linestyle="--" if expanded else "-",
                marker="s" if expanded else "o",
                markerfacecolor="none" if expanded else COLORS[arm],
                markersize=4,
            )
        for mode, label, style, marker in [
            ("true", "True interior + fixed evolution", ":", "^"),
            ("true_persistence", "True interior, hold fixed", "-.", "x"),
        ]:
            rows = selected[
                (selected["run"] == "A") & (selected["mode"] == mode)
            ].sort_values("lead_days")
            if not rows.empty:
                ax.plot(
                    rows["lead_days"],
                    rows["normalized_rmse"],
                    label=label,
                    color="#333333",
                    linestyle=style,
                    marker=marker,
                    markersize=4,
                )
        ax.set(
            title=title,
            xlabel="Lead (days)",
            ylabel="Normalized RMSE",
            xticks=[0, 5, 10, 15, 20, 25, 30],
        )
        ax.grid(alpha=0.2)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        fontsize=8,
        handlelength=4,
        markerscale=1.3,
    )
    fig.suptitle(
        "Initializer capacity and historical inputs: fixed pretrained evolution\nOM4-only, matching held-out origins; lead zero measures reconstruction"
    )
    for suffix in ["png", "svg"]:
        path = output / f"fixed-dynamics.{suffix}"
        fig.savefig(path, dpi=115, bbox_inches="tight")
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
        Path(str(path) + ".license").write_text(
            "SPDX-FileCopyrightText: 2026 Samudra Authors\n\nSPDX-License-Identifier: CC-BY-4.0\n"
        )
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    plot(args.analysis, args.output)


if __name__ == "__main__":
    main()
