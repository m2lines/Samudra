#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize frozen-model denoising controls from saved metrics."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(root, name):
    p = root / f"{name}.json"
    return json.loads(
        p.read_text()
        if p.exists()
        else gzip.decompress(p.with_suffix(".json.gz").read_bytes())
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("grid", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    read(args.root, "COMPLETE")
    controls = read(args.root, "correction")
    solver = read(args.root, "solver")
    examples = read(args.root, "examples")
    std = np.load(args.grid)["std"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for stage in ["initial", "day5", "monthly"]:
        rows = [
            q
            for q in solver
            if q["checkpoint"] == "2k" and q["stage"] == stage and q["seed"] == 905001
        ]
        ax[0, 0].plot(
            [q["steps"] for q in rows],
            [q["rmse"][38] * std[38] for q in rows],
            marker="o",
            label=stage,
        )
    ax[0, 0].set(
        title="More solver steps barely help",
        xlabel="Heun steps",
        ylabel="SST RMSE (°C)",
    )
    for condition in ["correct", "zero", "rolled"]:
        rows = [
            q
            for q in controls
            if q["checkpoint"] == "2k"
            and q["stage"] == "initial"
            and q["source"] == "gold"
            and q["condition"] == condition
        ]
        ax[0, 1].plot(
            [q["call"] / 2 + 1 for q in rows],
            [q["prediction_rmse"][38] * std[38] for q in rows],
            label=condition,
        )
    ax[0, 1].set(
        title="Latent conditioning is used",
        xlabel="Diffusion step",
        ylabel="Corrupted-gold SST RMSE (°C)",
    )
    rows = [
        q
        for q in controls
        if q["checkpoint"] == "2k"
        and q["stage"] == "initial"
        and q["source"] == "sampler"
        and q["condition"] == "correct"
    ]
    for key, label in [
        ("correction_cosine", "Alignment cosine"),
        ("correction_projection", "Fraction along needed correction"),
    ]:
        ax[0, 2].plot(
            [q["call"] / 2 + 1 for q in rows], [q[key][38] for q in rows], label=label
        )
    ax[0, 2].set(
        title="Correction weakens along sampler path",
        xlabel="Diffusion step",
        ylabel="SST correction diagnostic",
        ylim=(0, 1.05),
    )
    names = ["2000-07", "2010-01", "2015-01", "2018-01", "2021-07"]
    for col, (title, group, scale) in enumerate(
        [
            ("SST", slice(38, 39), float(std[38])),
            ("All temperature depths", slice(38, 57), 1),
            ("U velocity", slice(0, 19), 1),
        ]
    ):
        for label in ["2k", "4k"]:
            values = []
            for name in names:
                if name == "2015-01":
                    matches = [
                        q
                        for q in solver
                        if q["checkpoint"] == label
                        and q["stage"] == "initial"
                        and q["steps"] == 32
                        and q["seed"] == 905001
                    ]
                    value = matches[0]["rmse"]
                else:
                    value = next(
                        q["final_rmse"]
                        for q in examples
                        if q["checkpoint"] == label and q["name"] == name
                    )
                values.append(
                    float(np.sqrt(np.mean(np.array(value)[group] ** 2))) * scale
                )
            ax[1, col].plot(range(len(names)), values, marker="o", label=label)
        ax[1, col].set_xticks(
            range(len(names)),
            [
                n + ("\ntrain window" if i < 2 else "\nheld-out")
                for i, n in enumerate(names)
            ],
            fontsize=8,
        )
        ax[1, col].set(
            title=f"Longer main training: {title}",
            ylabel="RMSE (°C)" if col == 0 else "Normalized RMSE",
            ylim=(0, None),
        )
    for a in ax.flat:
        a.legend(fontsize=8)
        a.grid(alpha=0.2)
    fig.suptitle(
        "Original A · frozen checkpoints · supervised OM4\n2k versus continued 4k before cooldown; same local noise for paired sampling comparisons",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.output.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(args.output, dpi=110)


if __name__ == "__main__":
    main()
