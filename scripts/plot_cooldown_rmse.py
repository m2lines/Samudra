# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Rescore archived cooldown validation metrics and compare with original parents."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPONENTS = [
    "sst_rmse",
    "velocity_rmse",
    "eke_rmse",
    "ohc_0_700_rmse",
    "ohc_700_2000_rmse",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(gzip.decompress(args.source.read_bytes()))
    parents = list(csv.DictReader(args.parents.open()))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    rows = []
    common_reference = None
    for ax, (name, run) in zip(axes.flat, source.items()):
        reference = {key: run["reference"][key] for key in COMPONENTS}
        if common_reference is not None and reference != common_reference:
            raise ValueError("Different climatology references")
        common_reference = reference
        original = [
            r for r in parents if r["model"] == name and r["wave"] == "original 4k"
        ]
        if not original:
            raise ValueError("Missing original parent: " + name)
        for row in original:
            recalculated = (
                sum(float(row[key]) / reference[key] for key in COMPONENTS) / 5
            )
            if not math.isclose(
                recalculated, float(row["rmse_blend"]), rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError("Parent reference differs")
        points = sorted(run["events"], key=lambda r: r["update"])
        if points[-1]["update"] != 4000 or len({p["update"] for p in points}) != len(
            points
        ):
            raise ValueError("Incomplete or duplicate cooldown validation")
        for point in points:
            score = (
                sum(point["metrics"][key] / reference[key] for key in COMPONENTS) / 5
            )
            if not math.isfinite(score):
                raise ValueError("Nonfinite RMSE")
            rows.append(
                dict(model=name + "-cooldown", update=point["update"], rmse_blend=score)
            )
        cooled = [r for r in rows if r["model"] == name + "-cooldown"]
        ax.plot(
            [int(r["update"]) for r in original],
            [float(r["rmse_blend"]) for r in original],
            color="0.4",
            marker="o",
            markersize=3,
            label="Original constant LR = 1e-4",
        )
        ax.plot(
            [r["update"] for r in cooled],
            [r["rmse_blend"] for r in cooled],
            color="#0072B2",
            marker="o",
            markersize=3,
            label="Resumed run with cooldown",
        )
        ax.axvline(2667, color="0.3", linestyle=":", label="Checkpoint fork: 2,667")
        ax.axvline(3000, color="0.3", linestyle="--", label="LR decay begins: 3,000")
        ax.axvspan(3000, 4000, color="#0072B2", alpha=0.05)
        visible = [
            float(r["rmse_blend"]) for r in original if int(r["update"]) >= 2500
        ] + [r["rmse_blend"] for r in cooled]
        ax.set(
            title=name,
            xlabel="Total optimizer updates",
            ylabel="RMSE-only validation blend ↓",
            xlim=(2500, 4000),
            ylim=(min(visible) - 0.025, max(visible) + 0.025),
        )
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="upper left", frameon=False)
    axes.flat[-1].text(
        0.02,
        0.48,
        "All runs: 4k total updates, seed 1729\nLinear decay: 1e-4 → 1e-6 by update 4,000\n\nRaw events; no smoothing or reselection.\nEach panel has its own vertical scale.\n\nResumed trajectories diverge before decay;\nU-global also changes runtime/hardware.\nDifferences are not solely an LR effect.",
        transform=axes.flat[-1].transAxes,
        fontsize=10,
        va="top",
    )
    fig.suptitle("Cooldown versus original parent · RMSE-only validation", fontsize=17)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / ("rmse-validation-cooldown." + extension), dpi=100)
    with (args.output / "rmse-validation-cooldown.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "update", "rmse_blend"])
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Verified and plotted {len(rows)} cooldown validation events across {len(source)} models."
    )


if __name__ == "__main__":
    main()
