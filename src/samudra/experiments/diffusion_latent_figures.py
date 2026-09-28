# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the verified three-system comparison without pooling independent seeds."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.diffusion_scratch_figures import DEPTHS, annual, profiles

STYLES = (
    ("baseline", "#444444", "^", "Deterministic baseline"),
    ("B-1729", "#286cb0", "o", "Physical diffusion 1729"),
    ("B-1730", "#286cb0", "s", "Physical diffusion 1730"),
    ("D-1729", "#c84638", "D", "Latent + diffusion readout 1729"),
    ("D-1730", "#c84638", "X", "Latent + diffusion readout 1730"),
)


def calibration(summary, output, styles=STYLES[1:]):
    """Raw finite-ensemble coverage, spread/error profiles and nine-bin ranks."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharey=True)
    rows = (len(styles) + 1) // 2
    rank_fig, rank_axes = plt.subplots(
        rows, 2, figsize=(11, rows * 3.5), sharex=True, sharey=True, squeeze=False
    )
    for rank_ax, (name, color, marker, label) in zip(
        rank_axes.flat, styles, strict=True
    ):
        group = summary["runs"][name]["probabilistic"]["interior"]
        profiles = group["profiles"]
        supported = group["supported_channels"]
        coverage, ratio = np.full(28, np.nan), np.full(28, np.nan)
        coverage[supported] = profiles["coverage_90"]
        mse = np.asarray(profiles["mean_squared_error"])
        spread = np.asarray(profiles["ensemble_variance"])
        ratio[supported] = np.sqrt(
            np.divide(spread, mse, out=np.full_like(mse, np.nan), where=mse > 0)
        )
        for column in range(2):
            select = slice(column * 14, (column + 1) * 14)
            axes[0, column].plot(
                coverage[select], DEPTHS[:14], color=color, marker=marker, label=label
            )
            axes[1, column].plot(
                ratio[select], DEPTHS[:14], color=color, marker=marker, label=label
            )
        ranks = np.asarray(group["scores"]["rank_weights"])
        np.testing.assert_allclose(ranks.sum(), 1)
        rank_ax.bar(range(9), ranks, color=color)
        rank_ax.axhline(1 / 9, color="black", linewidth=0.8)
        rank_ax.set_title(label, fontsize=10)
        rank_ax.set_xticks(range(9))
    for column, title in enumerate(("Temperature", "Salinity")):
        axes[0, column].set_title(title)
        axes[0, column].set_xlabel("Raw 5–95% sample-quantile coverage")
        axes[0, column].set_xlim(0, 1)
        axes[1, column].set_xlabel("RMS ensemble spread / ensemble-mean RMSE")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Depth (m)")
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Monthly interior calibration: 96 origins, eight members, ±60°\nRaw eight-member quantiles do not have nominal 90% coverage"
    )
    fig.tight_layout()
    save_png(fig, output / "interior-calibration.png", dpi=135)
    plt.close(fig)
    for ax in rank_axes[-1]:
        ax.set_xlabel("Observation rank among eight members")
    for ax in rank_axes[:, 0]:
        ax.set_ylabel("Wet-area / channel averaged fraction")
    rank_fig.suptitle(
        "Monthly interior rank histograms; horizontal line = uniform 1/9\nLatent monthly members combine independently sampled readouts"
    )
    rank_fig.tight_layout()
    save_png(rank_fig, output / "interior-ranks.png", dpi=135)
    plt.close(rank_fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.summary, "rt") as stream:
        summary = json.load(stream)
    runs = {}
    for family in ("physical", "latent"):
        runs.update(summary["families"][family])
    expected = {name for name, *_ in STYLES if name != "baseline"}
    if set(runs) != expected:
        raise ValueError("Expected both completed seeds from both model families")
    shared = {"baseline": summary["baseline"], "runs": runs}
    args.output.mkdir(parents=True, exist_ok=True)
    profiles(shared, args.output, styles=STYLES)
    annual(shared, args.output, styles=STYLES)
    calibration(shared, args.output)


if __name__ == "__main__":
    main()
