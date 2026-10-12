#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Report the accessory and physical objectives from verified training events."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads(args.source.read_text())
    if not bundle["complete_wave"]:
        raise ValueError("Training summary requires the complete reproduction wave")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    rows = []
    for name, arm in bundle["arms"].items():
        coefficient = arm["auxiliary_targets"]["coefficient"]
        events = [e for e in arm["training_events"] if e["task"] == "om4"]
        budget = arm["training_complete"]["global_step"]
        if len(events) != budget // 2:
            raise ValueError("Missing OM4 events: " + name)
        blocks: dict[int, list[dict]] = {}
        for event in events:
            blocks.setdefault((event["global_step"] - 1) // 100, []).append(event)
        x, auxiliary, physical = [], [], []
        for block, values in sorted(blocks.items()):
            x.append(100 * (block + 1))
            auxiliary.append(statistics.mean(e["accessory_mse"] for e in values))
            physical.append(statistics.mean(e["physical_objective"] for e in values))
        color = "tab:orange" if "anomaly" in name else "tab:blue"
        style = "--" if budget == 6000 else "-"
        curves = [
            auxiliary,
            physical,
            [coefficient * a / p for a, p in zip(auxiliary, physical, strict=True)],
        ]
        for ax, values in zip(axes, curves, strict=True):
            ax.plot(x, values, label=name, color=color, ls=style)
        for period, values in [
            ("first quarter", [e for e in events if e["global_step"] <= budget // 4]),
            ("last quarter", [e for e in events if e["global_step"] > 3 * budget // 4]),
        ]:
            accessory = statistics.mean(e["accessory_mse"] for e in values)
            main_loss = statistics.mean(e["physical_objective"] for e in values)
            rows.append(
                dict(
                    model=name,
                    period=period,
                    om4_updates=len(values),
                    accessory_mse=accessory,
                    physical_objective=main_loss,
                    weighted_accessory=coefficient * accessory,
                    weighted_ratio=coefficient * accessory / main_loss,
                )
            )
    for ax, title in zip(
        axes,
        [
            "Accessory target MSE",
            "OM4 physical objective",
            "Weighted accessory / physical",
        ],
        strict=True,
    ):
        ax.set(title=title, xlabel="Total optimizer updates", yscale="log")
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        "OM4 updates only; averages within 100 schedule slots; ratios are not gradient contributions"
    )
    for suffix in ["png", "pdf"]:
        fig.savefig(out / ("accessory-training." + suffix), dpi=100)
    plt.close(fig)
    with (out / "accessory-training.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
