# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export the raw-validation qualification decision and observational EMA results."""

import argparse
import csv
import gzip
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from .report import write_csv

VIEWS = ("raw", "0.99", "0.999")


def task(config):
    return "copy" if config["main_weight"] else config["auxiliary"]


def key(config):
    return (
        task(config),
        config["lr"],
        config["schedule"],
        config["aux_loss_scale"] != 1,
    )


def metrics(record, view, step=None):
    if step is None:
        result = record["result"]
        return result["test"] if view == "raw" else result["ema_test"][view]
    row = next(row for row in record["history"] if row["step"] == step)
    return row if view == "raw" else row["ema"][view]


def export(records, output):
    groups = defaultdict(list)
    curves, runs, conditions = [], [], []
    for record in records:
        c = record["result"]["config"]
        groups[key(c)].append(record)
        identity = dict(
            id=record["result"]["id"],
            seed=c["seed"],
            task=task(c),
            lr=c["lr"],
            schedule=c["schedule"],
            normalized=c["aux_loss_scale"] != 1,
        )
        for view in VIEWS:
            test = metrics(record, view)
            runs.append(
                {
                    **identity,
                    "view": view,
                    "step": record["result"]["step"],
                    "training_seconds": record["result"]["training_seconds"],
                    **{k: v for k, v in test.items() if not isinstance(v, dict)},
                }
            )
            for row in record["history"]:
                value = row if view == "raw" else row["ema"][view]
                curves.append(
                    {
                        **identity,
                        "view": view,
                        "step": row["step"],
                        "training_seconds": row["training_seconds"],
                        **{
                            k: value[k]
                            for k in (
                                "bits",
                                "rmse",
                                "bias",
                                "auxiliary_normalized_rmse",
                                "auxiliary_rmse",
                            )
                        },
                    }
                )
    for recipe, group in groups.items():
        assert sorted(r["result"]["config"]["seed"] for r in group) == [11, 22, 33]
        for step in (3000, 12000, None):
            row = dict(
                task=recipe[0],
                lr=recipe[1],
                schedule=recipe[2],
                normalized=recipe[3],
                split="test" if step is None else "validation",
                step=12000 if step is None else step,
            )
            for view in VIEWS:
                data = [metrics(r, view, step) for r in group]
                for metric in (
                    "bits",
                    "rmse",
                    "bias",
                    "auxiliary_normalized_rmse",
                    "auxiliary_rmse",
                    "validation_bias_corrected_rmse",
                ):
                    if metric not in data[0]:
                        continue
                    values = [x[metric] for x in data]
                    row[f"{view}_{metric}_median"] = median(values)
                    row[f"{view}_{metric}_min"] = min(values)
                    row[f"{view}_{metric}_max"] = max(values)
                if step is None:
                    corrected_bits = [
                        -math.log2(
                            x["validation_bias_corrected_rmse"] / x["reference_scale"]
                        )
                        for x in data
                    ]
                    row[f"{view}_bias_corrected_bits_median"] = median(corrected_bits)
                metric = "bits" if recipe[0] == "copy" else "auxiliary_normalized_rmse"
                last = [
                    min(metrics(r, view, s)[metric] for s in (11800, 11900, 12000))
                    if recipe[0] == "copy"
                    else max(metrics(r, view, s)[metric] for s in (11800, 11900, 12000))
                    for r in group
                ]
                row[f"{view}_worst_seed_last_three"] = (
                    min(last) if recipe[0] == "copy" else max(last)
                )
            conditions.append(row)
    write_csv(output / "runs.csv", runs)
    write_csv(output / "conditions.csv", conditions)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
    writer.writeheader()
    writer.writerows(curves)
    (output / "curves.csv.gz").write_bytes(
        gzip.compress(stream.getvalue().encode(), mtime=0)
    )
    (output / "records.json.gz").write_bytes(
        gzip.compress(json.dumps(records, allow_nan=False).encode(), mtime=0)
    )
    return groups


def plot(groups, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def line(ax, group, view, metric, label, color, style="-"):
        x = [row["step"] for row in group[0]["history"]]
        values = np.array(
            [[metrics(r, view, step)[metric] for step in x] for r in group]
        )
        ax.plot(x, np.median(values, axis=0), label=label, color=color, linestyle=style)
        ax.fill_between(
            x, values.min(axis=0), values.max(axis=0), color=color, alpha=0.11
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    recipes = [
        (3e-4, "constant"),
        (3e-4, "cosine"),
        (3e-5, "constant"),
        (3e-5, "cosine"),
    ]
    for ax, (lr, schedule) in zip(axes.flat, recipes, strict=True):
        for i, view in enumerate(VIEWS):
            line(
                ax,
                groups[("copy", lr, schedule, False)],
                view,
                "bits",
                "Raw" if view == "raw" else f"EMA {view}",
                f"C{i}",
            )
        ax.axhline(6, color="black", linestyle=":", label="Raw qualification threshold")
        ax.axvline(3000, color="grey", linestyle=":", linewidth=0.8)
        ax.set(
            title=f"Copy: LR {lr:g}, {schedule}",
            xlabel="Updates",
            ylabel="Validation RMS bits (higher is better)",
            ylim=(2, 10),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Three-seed medians and ranges; EMAs never used for selection")
    fig.savefig(output / "copy-learning.png", dpi=90)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (name, schedule) in zip(
        axes.flat,
        [(t, s) for t in ("transport", "evolution") for s in ("constant", "cosine")],
        strict=True,
    ):
        for i, view in enumerate(VIEWS):
            line(
                ax,
                groups[(name, 3e-4, schedule, False)],
                view,
                "auxiliary_normalized_rmse",
                "Raw" if view == "raw" else f"EMA {view}",
                f"C{i}",
            )
        line(
            ax,
            groups[(name, 3e-4, schedule, True)],
            "raw",
            "auxiliary_normalized_rmse",
            "Raw, normalized loss",
            "black",
            "--",
        )
        ax.axhline(1, color="grey", linestyle=":", label="Zero predictor")
        ax.axhline(0.8, color="grey", linestyle="-.", label="Transport gate")
        ax.set(
            title=f"{name.capitalize()}: LR 0.0003, {schedule}",
            xlabel="Updates",
            ylabel="Validation auxiliary RMSE / target RMS",
            yscale="log",
            ylim=(0.02, 5),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Three-seed medians and ranges; initial errors above axis limits omitted"
    )
    fig.savefig(output / "auxiliary-learning.png", dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.source:
        entries = json.loads(args.manifest.read_text())
        records = [
            {
                "entry": e,
                "result": json.loads(
                    (args.source / e["id"] / "result.json").read_text()
                ),
                "history": json.loads(
                    (args.source / e["id"] / "history.json").read_text()
                ),
            }
            for e in entries
        ]
    else:
        records = json.loads(
            gzip.decompress((args.output / "records.json.gz").read_bytes())
        )
    groups = export(records, args.output)
    plot(groups, args.output)


if __name__ == "__main__":
    main()
