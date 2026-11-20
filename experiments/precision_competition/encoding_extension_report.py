# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched extension contrasts against original scalar-encoding controls."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median

from .encoding_report import LABELS, bits, identity, penalty, test_rows, write_csv

PAIRS = [
    ("cont_shrink_normalized", "cont_shrink_mse"),
    ("cont_fixed_mse", "cont_shrink_mse"),
    ("cont_shrink_normalized", "mse"),
    ("cont_fixed_mse", "mse"),
    ("ar_overlap_scaled", "ar_overlap"),
    ("ar_twohot_hard_scaled", "ar_twohot_hard"),
    ("ar_twohot_soft_scaled", "ar_twohot_soft"),
]


def paired_rows(rows):
    index = {}
    for row in rows:
        key = tuple(row[k] for k in ("kind", "lr", "seed", "phase", "view"))
        if key in index:
            raise ValueError("Duplicate comparison key")
        index[key] = row
    result = []
    for variant, reference in PAIRS:
        for key, row in index.items():
            kind, lr, seed, phase, view = key
            if kind != variant:
                continue
            other = index.get((reference, lr, seed, phase, view))
            if other is None:
                continue
            item = {
                "variant": variant,
                "reference": reference,
                "lr": lr,
                "seed": seed,
                "phase": phase,
                "view": view,
                "variant_bits": bits(row["rmse"]),
                "reference_bits": bits(other["rmse"]),
                "copy_bits_gain": penalty(other["rmse"], row["rmse"]),
                "variant_control_bits_lost": row.get("control_bits_lost"),
                "reference_control_bits_lost": other.get("control_bits_lost"),
            }
            a, b = row.get("control_bits_lost"), other.get("control_bits_lost")
            item["interference_penalty_reduction"] = (
                b - a if a is not None and b is not None else None
            )
            result.append(item)
    return result


def plot_weighting(bundles, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    for r, base in enumerate(("ar_overlap", "ar_twohot_hard", "ar_twohot_soft")):
        for c, lr in enumerate((1e-4, 1e-5)):
            ax = axes[r, c]
            for kind, color, label in (
                (base, "#2166ac", "Equal stage weights"),
                (base + "_scaled", "#b2182b", "Radius-squared weights"),
            ):
                selected = [
                    b
                    for b in bundles
                    if identity(b)["kind"] == kind and identity(b)["lr"] == lr
                ]
                for view, style in (("raw", "-"), ("ema", "--")):
                    histories = [b["phases"]["solo"]["curves"] for b in selected]
                    steps = [row["step"] for row in histories[0]]
                    values = [
                        [bits(h[i][view]["rmse"]) for h in histories]
                        for i in range(len(steps))
                    ]
                    ax.plot(
                        steps,
                        [median(v) for v in values],
                        style,
                        color=color,
                        label=label + (" raw" if view == "raw" else " EMA"),
                    )
                    if view == "raw":
                        ax.fill_between(
                            steps,
                            [min(v) for v in values],
                            [max(v) for v in values],
                            color=color,
                            alpha=0.12,
                        )
            ax.axvline(1000, color="gray", linewidth=0.8, linestyle=":")
            ax.set(
                title=f"{LABELS[base]} / LR {lr:g}",
                ylabel="RMS bits (higher is better)",
            )
            ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Solo optimizer updates")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Stage weighting: paired seeds; median curves, raw seed ranges; dotted line ends correct-prefix warm-up"
    )
    fig.tight_layout()
    fig.savefig(output / "stage-weighting-comparison.png", dpi=150)
    plt.close(fig)


def plot_continuous_stages(rows, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for r, view in enumerate(("raw", "ema")):
        for c, lr in enumerate((1e-4, 1e-5)):
            ax = axes[r, c]
            for kind, color in (
                ("cont_shrink_mse", "#2166ac"),
                ("cont_shrink_normalized", "#b2182b"),
                ("cont_fixed_mse", "#4d9221"),
            ):
                selected = [
                    x
                    for x in rows
                    if x["kind"] == kind
                    and x["lr"] == lr
                    and x["view"] == view
                    and x["phase"] == "solo"
                ]
                values = [[x[f"stage_rmse_{k}"] for x in selected] for k in range(1, 5)]
                ax.plot(
                    range(1, 5),
                    [median(v) for v in values],
                    "o-",
                    color=color,
                    label=LABELS[kind],
                )
                ax.fill_between(
                    range(1, 5),
                    [min(v) for v in values],
                    [max(v) for v in values],
                    color=color,
                    alpha=0.12,
                )
            baseline = [
                x["rmse"]
                for x in rows
                if x["kind"] == "mse"
                and x["lr"] == lr
                and x["view"] == view
                and x["phase"] == "solo"
            ]
            ax.axhline(
                median(baseline),
                color="black",
                linestyle="--",
                label="One-shot MSE final output",
            )
            ax.set(
                title=f"LR {lr:g}, {view}",
                xlabel="Refinement stage",
                ylabel="Normalized RMSE",
                yscale="log",
                xticks=range(1, 5),
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Continuous refinement: final solo test, median and range across three seeds"
    )
    fig.tight_layout()
    fig.savefig(output / "continuous-stage-comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extension = json.loads(gzip.decompress(args.extension.read_bytes()))
    original = json.loads(gzip.decompress(args.original.read_bytes()))
    controls = [
        b
        for b in original
        if identity(b)["kind"]
        in {"mse", "ar_overlap", "ar_twohot_hard", "ar_twohot_soft"}
    ]
    bundles = extension + controls
    rows = [r for b in bundles for r in test_rows(b)]
    write_csv(args.output / "matched-contrasts.csv", paired_rows(rows))
    write_csv(
        args.output / "original-control-test-metrics.csv",
        [r for b in controls for r in test_rows(b)],
    )
    plot_weighting(bundles, args.output)
    plot_continuous_stages(rows, args.output)


if __name__ == "__main__":
    main()
