#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot training and held-out depth diagnostics from verified saved results."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEPTHS = [2.5, 10, 22.5, 40, 65, 105, 165, 250, 375, 550, 775, 1050, 1400, 1850]


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(source, output):
    result = json.loads(source.read_text())
    assert result["paths"]["selection_metric"] == "rmse-only"
    output.mkdir(parents=True, exist_ok=True)
    training, profiles = [], []
    for name, arm in result["arms"].items():
        bins: dict[tuple[str, int], list[dict]] = {}
        previous_early = 0
        for event in arm["training_events"]:
            task = event["task"]
            if task == "om4":
                task = (
                    "early OM4" if event["om4_early"] > previous_early else "recent OM4"
                )
            previous_early = event["om4_early"]
            block = (event["global_step"] - 1) // 100
            bins.setdefault((task, block), []).append(event)
        for (task, _), events in bins.items():
            training.append(
                dict(
                    model=name,
                    task=task,
                    mean_update=statistics.mean(e["global_step"] for e in events),
                    mean_loss=statistics.mean(e["loss"] for e in events),
                    samples=len(events),
                )
            )
        for forecast, filename in [
            ("evolved", "selected.json"),
            ("initialized persistence", "selected-inferred-persistence.json"),
        ]:
            data = arm["files"][filename]["data"]
            if forecast == "evolved":
                data = data["metrics"]
            ts = data["thermohaline"]
            for channel, rmse, bias in zip(
                ts["channels"], ts["rmse"], ts["bias"], strict=True
            ):
                field, index = channel.split("_")
                profiles.append(
                    dict(
                        model=name,
                        forecast=forecast,
                        field=field,
                        depth_m=DEPTHS[int(index)],
                        rmse=rmse,
                        bias=bias,
                    )
                )
    write_csv(output / "training-loss.csv", training)
    write_csv(output / "heldout-depth-errors.csv", profiles)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, task in zip(axes, ["recent OM4", "early OM4", "observation"], strict=True):
        for name in result["arms"]:
            rows = [r for r in training if r["model"] == name and r["task"] == task]
            if rows:
                ax.plot(
                    [r["mean_update"] for r in rows],
                    [r["mean_loss"] for r in rows],
                    label=name,
                )
        ax.set(
            title=task,
            xlabel="Total updates",
            ylabel="Training objective",
            yscale="log",
        )
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=6)
    fig.suptitle(
        "Means within 100-update bins; early fine and coarse objectives differ"
    )
    fig.savefig(output / "training-loss.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(
        1, 4, figsize=(14, 5), sharey=True, constrained_layout=True
    )
    for ax, (field, measure) in zip(
        axes,
        [("thetao", "rmse"), ("thetao", "bias"), ("so", "rmse"), ("so", "bias")],
        strict=True,
    ):
        for name in result["arms"]:
            rows = [
                r
                for r in profiles
                if r["model"] == name
                and r["field"] == field
                and r["forecast"] == "evolved"
            ]
            ax.plot(
                [r[measure] for r in rows], [r["depth_m"] for r in rows], label=name
            )
        if measure == "bias":
            ax.axvline(0, color="gray", ls=":", lw=1)
        unit = "°C" if field == "thetao" else "practical salinity"
        ax.set(xlabel=f"{field} {measure} ({unit})")
        ax.grid(alpha=0.2)
    axes[0].set(ylabel="Depth (m)", ylim=(1900, 0))
    axes[-1].legend(fontsize=6)
    fig.suptitle("Selected-model monthly T/S errors over 96 held-out origins")
    fig.savefig(output / "heldout-depth-errors.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.source, args.output)
