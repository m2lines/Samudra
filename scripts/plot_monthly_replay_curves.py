#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Replay-only views of the verified six-model annual comparison."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.assets / "annual-results.json").read_text())
    assert result["verification"]["annual_cases"] == 18
    models = {name: result["models"][name] for name in ("A-replay", "E-replay")}
    colors = {"A-replay": "#0072B2", "E-replay": "#D55E00"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    keys = ("sst_rmse", "adt_rmse", "velocity_rmse", "eke_rmse")
    titles = (
        "SST RMSE (°C)",
        "ADT RMSE (m)",
        "Velocity RMSE (m/s)",
        "EKE RMSE (m²/s²)",
    )
    for ax, key, title in zip(axes.flat, keys, titles, strict=True):
        for name, model in models.items():
            ax.plot(
                [v["day"] for v in model["leads"]],
                [v[key] for v in model["leads"]],
                label=name,
                color=colors[name],
            )
        ax.plot(
            [v["day"] for v in result["climatology_leads"]],
            [v[key] for v in result["climatology_leads"]],
            "k:",
            label="Seasonal climatology",
        )
        ax.set(title=title, xlabel="Forecast lead (days)")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Replay models · three held-out starts (2015, 2018, 2021) · fixed cooled EMA endpoints",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.assets / "annual-replay-only-surface.png", dpi=120)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name, model in models.items():
        for ax, key in zip(axes[:2], ("0_700", "700_2000"), strict=True):
            ax.plot(
                range(1, 13),
                [v[key] / 1e9 for v in model["monthly_ohc"]],
                label=name,
                color=colors[name],
            )
        axes[2].plot(
            np.arange(74) * 5, model["latent_rms_mean"], label=name, color=colors[name]
        )
    for ax, key, title in zip(
        axes[:2], ("0_700", "700_2000"), ("OHC 0–700 m", "OHC 700–2,000 m"), strict=True
    ):
        ax.plot(
            range(1, 13),
            [
                v[key + "_climatology"] / 1e9
                for v in next(iter(models.values()))["monthly_ohc"]
            ],
            "k:",
            label="Seasonal climatology",
        )
        ax.set(title=title + " RMSE (GJ/m²)", xlabel="Calendar month")
    axes[2].set(title="Mean latent RMS", xlabel="Forecast lead (days)")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Replay models · three held-out starts · monthly readouts and continuous latent rollout",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.assets / "annual-replay-only-ohc-latent.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
