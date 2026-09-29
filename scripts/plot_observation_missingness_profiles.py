#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot initialized state profiles and held-out regional spectra from audit bundles."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summarize_observation_missingness import NAMES  # type: ignore[import-not-found]

from samudra.experiments.observation_prepare import DEPTHS


def profiles(files, arms, output):
    variables = [
        (38, "Temperature (°C)"),
        (57, "Salinity (psu)"),
        (0, "Zonal velocity (m/s)"),
        (19, "Meridional velocity (m/s)"),
    ]
    for statistic in ("initial_mean", "initial_rms"):
        fig, axes = plt.subplots(
            1, 4, figsize=(13, 5), sharey=True, layout="constrained"
        )
        for arm in arms:
            records = files[arm + "-state-diagnostics/best.json"]["profiles"]
            values = np.array([r[statistic] for r in records]).mean(0)
            for ax, (offset, label) in zip(axes, variables, strict=True):
                ax.plot(values[offset : offset + 14], DEPTHS, label=NAMES[arm])
                ax.set_xlabel(label)
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Depth (m)")
        axes[0].legend(fontsize=6)
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.suptitle(
            f"Selected initializer: {statistic.replace('_', ' ')} over wet cells, averaged over 9 validation origins"
        )
        fig.savefig(output / (statistic + "-profiles.png"), dpi=160)
        plt.close(fig)
    for arm in arms:
        fig, axes = plt.subplots(
            1, 4, figsize=(13, 5), sharey=True, layout="constrained"
        )
        for checkpoint in (
            "joint-00000",
            "joint-00010",
            "joint-00100",
            "joint-01000",
            "joint-04000",
            "before-observation-finish",
            "joint-08000",
            "joint-16000",
            "best",
        ):
            key = f"{arm}-state-diagnostics/{checkpoint}.json"
            if key not in files:
                continue
            records = files[key]["profiles"]
            values = np.array([r["initial_mean"] for r in records]).mean(0)
            for ax, (offset, label) in zip(axes, variables, strict=True):
                ax.plot(values[offset : offset + 14], DEPTHS, label=checkpoint)
                ax.set_xlabel(label)
                ax.grid(alpha=0.2)
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Depth (m)")
        axes[0].legend(fontsize=6)
        fig.suptitle(
            NAMES[arm]
            + ": initialized wet-cell mean profiles; joint-N denotes observation updates"
        )
        fig.savefig(output / (arm + "-profile-evolution.png"), dpi=160)
        plt.close(fig)


def spectra(files, arms, output):
    for lead in (5, 30):
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout="constrained")
        for row, variable in enumerate(("sst", "adt", "eke")):
            for col, region in enumerate(("North Pacific", "Gulf Stream", "Agulhas")):
                ax = axes[row, col]
                reference = None
                for arm in arms:
                    record = files[arm + "-selected-monthly/selected.json"]["metrics"][
                        "spectra"
                    ][f"{variable}/{region}/day{lead}"]
                    if reference is None:
                        reference = record
                        ax.loglog(
                            record["k_rad_km"],
                            record["reference_power"],
                            color="black",
                            linestyle="--",
                            label="Observations",
                        )
                    else:
                        np.testing.assert_allclose(
                            record["reference_power"],
                            reference["reference_power"],
                            rtol=1e-5,
                        )
                    ax.loglog(
                        record["k_rad_km"],
                        record["prediction_power"],
                        marker="o",
                        markersize=3,
                        label=NAMES[arm],
                    )
                ax.set_title(f"{variable.upper()} / {region}")
                ax.set_xlabel("Wavenumber (rad/km)")
                ax.set_ylabel("Power (metric convention)")
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=6)
        fig.suptitle(
            f"Selected models: day-{lead} regional spectra, 96 held-out monthly origins"
        )
        fig.savefig(output / f"spectra-day{lead}.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(gzip.decompress(args.bundle.read_bytes()))
    arms = [a for a in NAMES if data["status"][a] == "verified complete"]
    args.output.mkdir(parents=True, exist_ok=True)
    profiles(data["files"], arms, args.output)
    spectra(data["files"], arms, args.output)


if __name__ == "__main__":
    main()
