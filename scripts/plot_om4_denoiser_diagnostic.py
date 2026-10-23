#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot the matched corrupted-gold versus sampler-state denoising diagnostic."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    assert (args.results / "COMPLETE.json").is_file()
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharex=True)
    groups = [
        ("U", slice(0, 19)),
        ("V", slice(19, 38)),
        ("Temperature", slice(38, 57)),
        ("Salinity", slice(57, 76)),
        ("SSH", slice(76, 77)),
    ]
    for row, stage in enumerate(["initial", "day5", "monthly"]):
        path = args.results / f"{stage}.json"
        records = json.loads(
            path.read_text()
            if path.exists()
            else gzip.decompress(path.with_suffix(".json.gz").read_bytes())
        )
        steps = np.array([r["call"] for r in records]) / 2 + 1
        sampler = np.array([r["local_replay_rmse"] for r in records])
        gold = np.array([r["corrupted_gold_rmse"] for r in records])
        for col, (name, group) in enumerate(groups):
            ax = axes[row, col]
            y = np.sqrt((sampler[:, group] ** 2).mean(-1))
            draws = np.sqrt((gold[:, :, group] ** 2).mean(-1))
            ax.plot(steps, y, color="#dc2626", label="Sampler input")
            ax.plot(
                steps, draws.mean(-1), color="#2563eb", label="Corrupted gold input"
            )
            ax.fill_between(
                steps, draws.min(-1), draws.max(-1), color="#2563eb", alpha=0.2
            )
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(stage + "\nNormalized RMSE")
            if row == 2:
                ax.set_xlabel("Diffusion step (predictor call)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Original A 2k EMA · supervised OM4 · same latent, gold, support and noise level\nJanuary 2015; corrupted-gold band spans three fixed noise draws; no retraining",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=110)


if __name__ == "__main__":
    main()
