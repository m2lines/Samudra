#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plot validation and matched training-probe curves without reading held-out scores."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from samudra.experiments.surface_adaptation_analysis import read_bytes

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LICENSE = "SPDX-FileCopyrightText: 2026 Samudra Authors\n\nSPDX-License-Identifier: CC-BY-4.0\n"


def plot(raw, output, title):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for run in sorted(Path(raw).iterdir()):
        if not run.is_dir():
            continue
        manifest = json.loads(read_bytes(run / "manifest.json"))
        args = manifest["arguments"]
        if args["evaluate_only"] or args["max_steps"]:
            continue
        events = [
            json.loads(line)
            for line in read_bytes(run / "progress.jsonl").decode().splitlines()
        ]
        for event in events:
            if event.get("event") not in ["baseline", "validation"]:
                continue
            common = {
                "run": run.name,
                "arm": args["arm"],
                "phase": args["phase"],
                "learning_rate": args["learning_rate"],
                "step": event.get("step", 0),
                "elapsed_hours": event.get("elapsed_seconds", 0) / 3600,
            }
            records.append(
                {**common, "split": "validation", "ts_rmse": np.sqrt(event["ts_mse"])}
            )
            if "train_probe" in event:
                records.append(
                    {
                        **common,
                        "split": "train probe",
                        "ts_rmse": np.sqrt(event["train_probe"]["ts_mse"]),
                    }
                )
    if not records:
        raise ValueError("No production/pilot validation curves")
    data = pd.DataFrame(records)
    data.to_csv(output / "learning-curves.csv", index=False)
    arms = sorted(data["arm"].unique())
    for x, label, suffix in [
        ("elapsed_hours", "Training elapsed time (hours)", "time"),
        ("step", "Optimizer updates", "updates"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        for ax, arm in zip(axes.flat, "ABCDEF", strict=True):
            ax.set(
                title="Arm " + arm,
                xlabel=label,
                ylabel="Normalized T/S RMSE",
                yscale="log",
            )
            ax.grid(alpha=0.2)
            selected = data[data["arm"] == arm]
            for color, (run_name, rows) in zip(
                plt.rcParams["axes.prop_cycle"].by_key()["color"],
                selected.groupby("run"),
            ):
                rate = rows["learning_rate"].iloc[0]
                for split, marker, style in [
                    ("validation", "s", "-"),
                    ("train probe", "o", ":"),
                ]:
                    curve = rows[rows["split"] == split].sort_values("step")
                    ax.plot(
                        curve[x],
                        curve["ts_rmse"],
                        color=color,
                        marker=marker,
                        markerfacecolor="none" if split == "train probe" else color,
                        linestyle=style,
                        markersize=3,
                        label=f"{run_name} ({rate:g}), {split}",
                    )
            if arm in arms:
                ax.legend(fontsize=6, handlelength=3.5)
        fig.suptitle(
            title
            + "\nValidation selects checkpoints; training probes use the same metric and evaluation mode"
        )
        path = output / f"learning-{suffix}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        Path(str(path) + ".license").write_text(LICENSE)
    Path(str(output / "learning-curves.csv") + ".license").write_text(LICENSE)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--title", default="Initializer wave: training and validation")
    args = p.parse_args()
    plot(args.raw, args.output, args.title)


if __name__ == "__main__":
    main()
