#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Render fixed-member pilot maps and finite-ensemble diagnostic plots."""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from samudra.experiments.surface_adaptation_analysis import read_bytes

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATES = [
    "2014-11-04",
    "2015-02-02",
    "2015-05-03",
    "2015-08-01",
    "2018-11-14",
    "2022-11-24",
]


def four_up(path, output):
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("plot_initializer_diagnostic_maps.py")),
            "--maps",
            str(path),
            "--output",
            str(output),
            "--dates",
            *DATES,
            "--panels",
        ],
        check=True,
    )


def render(raw, analysis, output):
    output.mkdir(parents=True, exist_ok=True)
    for run in ["diffusion", "deterministic"]:
        path = raw / run / "heldout-maps.npz"
        four_up(path, output)
    with np.load(io.BytesIO(read_bytes(raw / "diffusion/heldout-maps.npz"))) as archive:
        metadata = {
            k: archive[k] for k in ["latitude", "longitude", "mask", "mean", "std"]
        }
        keys = {
            str(archive[k]).split()[0]: k.removesuffix("_date")
            for k in archive.files
            if k.endswith("_date")
        }
        for member in [0, 1]:
            fields = dict(metadata)
            for day in DATES:
                idx = keys[day]
                with np.load(
                    io.BytesIO(
                        read_bytes(raw / "diffusion" / f"heldout-members-{idx}.npz")
                    )
                ) as data:
                    for name in ["truth", "climatology", "date"]:
                        fields[f"{idx}_{name}"] = data[name]
                    fields[f"{idx}_prediction"] = data["samples"][member]
            directory = output / f"diffusion-member{member}"
            directory.mkdir(exist_ok=True)
            path = directory / "maps.npz"
            np.savez_compressed(path, **fields)
            four_up(path, output)
    with np.load(io.BytesIO(read_bytes(raw / "diffusion/heldout-maps.npz"))) as archive:
        mask = archive["mask"].astype(bool)
        spread_limit = float(
            np.percentile(
                np.concatenate([archive[keys[day] + "_spread"][mask] for day in DATES]),
                99,
            )
            * float(archive["std"])
        )
        for day in DATES:
            field = np.where(
                mask, archive[keys[day] + "_spread"] * float(archive["std"]), np.nan
            )
            fig = plt.figure(figsize=(8.4, 5), dpi=100)
            ax = fig.add_axes((65 / 840, 80 / 500, 720 / 840, 360 / 500))
            im = ax.imshow(
                field,
                origin="lower",
                interpolation="nearest",
                resample=False,
                cmap="viridis",
                vmin=0,
                vmax=spread_limit,
            )
            ax.set_facecolor("#ddd")
            ax.set_title(
                day + " | Ensemble spread (salinity standard deviation)", fontsize=10
            )
            ax.set_xticks(
                [-0.5, 89.5, 179.5, 269.5, 359.5], ["0", "90", "180", "270", "360"]
            )
            latitude = archive["latitude"]
            ax.set_yticks(
                [int(np.argmin(abs(latitude - v))) for v in [-60, -30, 0, 30, 60]],
                ["−60", "−30", "0", "30", "60"],
            )
            ax.set(xlabel="Longitude", ylabel="Latitude")
            cax = fig.add_axes((65 / 840, 20 / 500, 720 / 840, 10 / 500))
            fig.colorbar(im, cax=cax, orientation="horizontal")
            fig.canvas.draw()
            assert (
                abs(ax.get_window_extent().width - 720) < 1e-6
                and abs(ax.get_window_extent().height - 360) < 1e-6
            )
            fig.savefig(output / f"spread-{day}-2x.png", dpi=100)
            plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for run, color, marker in [
        ("diffusion", "#377eb8", "o"),
        ("deterministic", "#ff7f00", "D"),
    ]:
        events = [
            json.loads(line)
            for line in (raw / run / "progress.jsonl").read_text().splitlines()
        ]
        events = [e for e in events if e.get("event") == "validation_selection"]
        if events:
            steps = [e["step"] for e in events]
            axes[0].plot(
                steps,
                [e["scores"]["crps"] for e in events],
                label=run,
                color=color,
                marker=marker,
                markersize=3,
            )
            axes[1].plot(
                steps,
                [np.sqrt(e["scores"]["mean_mse"]) for e in events],
                label=run,
                color=color,
                marker=marker,
                markersize=3,
            )
    axes[0].set(
        xlabel="Optimizer updates",
        ylabel="Validation CRPS",
        title="Eight-member diffusion / point control",
    )
    axes[1].set(
        xlabel="Optimizer updates",
        ylabel="Validation RMSE",
        title="Ensemble mean / point control",
    )
    for ax in axes:
        ax.legend()
    fig.savefig(output / "learning.png", dpi=130)
    plt.close(fig)
    ranks = pd.read_csv(analysis / "rank-histogram.csv").groupby("rank").weight.mean()
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    ax.bar(ranks.index, ranks.to_numpy(), color="#377eb8")
    ax.axhline(
        1 / len(ranks), color="black", linestyle="--", label="Exchangeable reference"
    )
    ax.set(
        xlabel="Truth rank among 16 members (0–16)",
        ylabel="Area-weighted frequency",
        title="Held-out rank histogram; uniform is calibrated",
    )
    ax.legend()
    fig.savefig(output / "rank-histogram.png", dpi=130)
    plt.close(fig)
    spectra = pd.read_csv(analysis / "zonal-spectrum.csv")
    styles = {
        "truth": ("black", "x"),
        "D": ("#999999", "^"),
        "specialist": ("#ff7f00", "D"),
        "mean": ("#377eb8", "o"),
        "sample0": ("#4daf4a", "s"),
    }
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for name, (color, marker) in styles.items():
        g = spectra[
            (spectra.run == "diffusion")
            & (spectra.field == name)
            & (spectra.zonal_wavenumber > 0)
        ]
        ax.loglog(
            g.zonal_wavenumber,
            g.power,
            label=name,
            color=color,
            marker=marker,
            markevery=20,
            markersize=4,
        )
    ax.set(
        xlabel="Grid-space zonal wavenumber",
        ylabel="Salinity anomaly power",
        title="Common-mask zonal spectra (land edges affect power)",
    )
    ax.legend()
    fig.savefig(output / "zonal-spectrum.png", dpi=130)
    plt.close(fig)
    covariance = pd.read_csv(analysis / "spatial-covariance.csv")
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for name, (color, marker) in styles.items():
        lagged = (
            covariance[(covariance.run == "diffusion") & (covariance.field == name)]
            .groupby("lag_cells")
            .structure_function.mean()
        )
        ax.plot(lagged.index, lagged.to_numpy(), label=name, color=color, marker=marker)
    ax.set(
        xlabel="Zonal lag (grid cells)",
        ylabel="Mean squared salinity difference",
        title="Spatial structure on pairs of wet cells",
    )
    ax.legend()
    fig.savefig(output / "spatial-structure.png", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    render(args.raw, args.analysis, args.output)


if __name__ == "__main__":
    main()
