# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Reproduce paired-seed summaries and figures from the compact first-wave record."""

import argparse
import gzip
import json
import math
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median

import numpy as np

from .benchmark import Config
from .report import write_csv


def grouped(records):
    groups = defaultdict(list)
    for record in records:
        config = Config(**record["entry"]["config"])
        groups[replace(config, seed=11).run_id].append(record)
    return groups


def summarize(records, output):
    lookup = {r["result"]["id"]: r for r in records}
    rows = []
    for group in grouped(records).values():
        config = Config(**group[0]["entry"]["config"])
        row = {k: v for k, v in asdict(config).items() if k != "seed"}
        row.update(
            seeds=len(group), parameters=group[0]["result"]["provenance"]["parameters"]
        )
        metrics = defaultdict(list)
        for record in group:
            result = record["result"]
            c = Config(**record["entry"]["config"])
            for key, value in result["test"].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[key].append(value)
            metrics["training_seconds"].append(result["training_seconds"])
            metrics["bias_mse_fraction"].append(
                result["test"]["bias"] ** 2 / result["test"]["rmse"] ** 2
            )
            metrics["high_frequency_error_rms"].append(
                result["spectrum"]["high_frequency_error_rms"]
            )
            solo = lookup.get(
                replace(
                    c, auxiliary_weight=0.0, relation="independent", auxiliary_start=0
                ).run_id
            )
            if c.main_weight and c.auxiliary_weight and solo:
                assert solo["result"]["step"] == result["step"]
                metrics["test_bits_lost"].append(
                    math.log2(result["test"]["rmse"] / solo["result"]["test"]["rmse"])
                )
                a = {h["step"]: h for h in solo["history"]}
                for h in record["history"]:
                    if h["step"] in (1000, 2000, 3000) and h["step"] in a:
                        metrics[f"validation_bits_lost_{h['step']}"].append(
                            math.log2(h["rmse"] / a[h["step"]]["rmse"])
                        )
        for key, values in metrics.items():
            row.update(
                {
                    f"{key}_median": median(values),
                    f"{key}_min": min(values),
                    f"{key}_max": max(values),
                }
            )
        rows.append(row)
    write_csv(output / "conditions.csv", rows)
    return lookup


def figures(records, lookup, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def get(config):
        return [lookup[replace(config, seed=s).run_id] for s in (11, 22, 33)]

    def range_line(ax, x, y, label, **kwargs):
        y = np.array(y)
        ax.plot(x, np.median(y, axis=0), label=label, **kwargs)
        ax.fill_between(x, y.min(axis=0), y.max(axis=0), alpha=0.13)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for weight, relation, label in [
        (0.0, "independent", "Solo copy"),
        (1.0, "independent", "Independent joint"),
        (1.0, "dependent", "Dependent joint"),
    ]:
        group = get(replace(Config(), auxiliary_weight=weight, relation=relation))
        steps = [h["step"] for h in group[0]["history"]]
        range_line(
            axes[0], steps, [[h["bits"] for h in r["history"]] for r in group], label
        )
    for weight, label in [(0.0, "Solo copy"), (1.0, "Auxiliary alone")]:
        c = replace(Config(), auxiliary_weight=weight, main_weight=1.0 - weight)
        group = get(c)
        key = "normalized_rmse" if weight == 0 else "auxiliary_normalized_rmse"
        range_line(
            axes[1], steps, [[h[key] for h in r["history"]] for r in group], label
        )
    axes[0].set(
        ylabel="Effective RMS bits (higher is better)", xlabel="Optimizer updates"
    )
    axes[1].set(
        ylabel="RMSE / task reference scale", xlabel="Optimizer updates", yscale="log"
    )
    axes[1].axhline(1, color="black", linestyle=":", linewidth=1)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Base model: median and full range across three seeds")
    fig.savefig(output / "paired-learning.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    settings = [
        (
            [(w, 4, h) for w, h in [(128, 2), (256, 4), (512, 8)]],
            "Width / heads, head width 64",
        ),
        ([(256, d, 4) for d in [2, 4, 8, 16]], "Depth, width 256 / 4 heads"),
        ([(256, 4, h) for h in [2, 4, 8]], "Heads, width 256 / depth 4"),
    ]
    for ax, (configs, title) in zip(axes, settings, strict=True):
        for offset, relation in [(-0.12, "independent"), (0.12, "dependent")]:
            for i, (width, depth, heads) in enumerate(configs):
                c = replace(Config(), width=width, depth=depth, heads=heads)
                solo, joint = (
                    get(c),
                    get(replace(c, auxiliary_weight=1.0, relation=relation)),
                )
                values = [
                    math.log2(b["result"]["test"]["rmse"] / a["result"]["test"]["rmse"])
                    for a, b in zip(solo, joint, strict=True)
                ]
                color = "C0" if relation == "independent" else "C1"
                ax.plot([i + offset] * 3, values, ".", color=color, alpha=0.65)
                ax.errorbar(
                    i + offset,
                    median(values),
                    yerr=[
                        [median(values) - min(values)],
                        [max(values) - median(values)],
                    ],
                    fmt="_",
                    markersize=12,
                    capsize=3,
                    color=color,
                    label=relation if i == 0 else None,
                )
        ax.axhline(0, color="black", linewidth=0.8)
        labels = [
            f"{w}/{h}"
            if title.startswith("Width")
            else str(d if title.startswith("Depth") else h)
            for w, d, h in configs
        ]
        ax.set(
            xticks=range(len(configs)),
            xticklabels=labels,
            title=title,
            ylabel="Test bits lost to joint training",
        )
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("3,000 updates; seed-paired medians, ranges and individual points")
    fig.savefig(output / "capacity.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    bins = [16, 64, 256, 1024, 4096]
    for weight, label in [(0.0, "Solo CE"), (1.0, "Joint CE")]:
        curve_values = [
            [
                get(replace(Config(), objective="ce", bins=b, auxiliary_weight=weight))[
                    s
                ]["result"]["test"]["bits"]
                for b in bins
            ]
            for s in range(3)
        ]
        range_line(axes[0], bins, curve_values, label, marker="o")
    oracle = [
        -math.log2(
            get(replace(Config(), objective="ce", bins=b))[0]["result"]["test"][
                "quantization_rmse"
            ]
            / get(Config())[0]["result"]["test"]["reference_scale"]
        )
        for b in bins
    ]
    axes[0].plot(bins, oracle, "k--", label="Oracle nearest-center quantization")
    axes[0].set(xscale="log", xlabel="Output bins", ylabel="Effective RMS bits")
    for weight, relation, label in [
        (0.0, "independent", "Solo"),
        (1.0, "independent", "Independent joint"),
        (1.0, "dependent", "Dependent joint"),
    ]:
        alphas = [0.001, 0.01, 0.1]
        curve_values = [
            [
                get(
                    replace(
                        Config(),
                        main="forecast",
                        auxiliary="evolution",
                        alpha=a,
                        auxiliary_weight=weight,
                        relation=relation,
                    )
                )[s]["result"]["test"]["persistence_ratio"]
                for a in alphas
            ]
            for s in range(3)
        ]
        range_line(axes[1], alphas, curve_values, label, marker="o")
    axes[1].axhline(1, color="black", linestyle=":", label="Persistence")
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Nominal forecast alpha",
        ylabel="RMSE / persistence RMSE (lower is better)",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("3,000 updates; medians and full ranges across three seeds")
    fig.savefig(output / "bins-and-persistence.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    records = json.load(gzip.open(args.results / "records.json.gz", "rt"))
    lookup = summarize(records, args.results)
    figures(records, lookup, args.results)


if __name__ == "__main__":
    main()
