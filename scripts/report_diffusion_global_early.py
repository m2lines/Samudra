#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Member maps, regional spectra and pooled calibration for early global checkpoints."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_correlated_noise_pilot import (  # type: ignore[import-not-found]
    patch,
    spectrum,
)

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.diffusion_latent_maps import grid
from samudra.experiments.observation_pilot import digest


def rank_probabilities(records, task):
    count = records[0][task]["members"]
    counts = sum(
        np.array(row[task]["rank_weights"]).reshape(count + 1, -1).sum(1)
        for row in records
    )
    return counts / counts.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-root", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((args.root / "COMPLETE.json").read_text())
    calibration = json.loads((args.root / "calibration.json").read_text())
    comparison = None
    if args.compare_root:
        comparison = json.loads((args.compare_root / "COMPLETE.json").read_text())
        for key in ("checkpoint_sha256", "training_contract", "split"):
            if receipt[key] != comparison[key]:
                raise ValueError(f"Sampling comparison changed {key}")
        if receipt["files"].keys() != comparison["files"].keys():
            raise ValueError("Sampling comparison changed evaluation origins")
    spectra = []
    for index, (filename, sha) in enumerate(sorted(receipt["files"].items())):
        path = args.root / filename
        if digest(path) != sha:
            raise ValueError(f"Changed member export: {filename}")
        with np.load(path) as data:
            for channel, name in enumerate(data["channel_names"]):
                # Surface references are five-day bins; interiors are monthly.
                monthly_index = 9 if channel == 2 else 23
                members = (
                    data["day30"][:, channel]
                    if channel < 2
                    else data["monthly"][:, monthly_index]
                )
                target = (
                    data["day30_surface_reference"][channel]
                    if channel < 2
                    else data["monthly_reference"][monthly_index]
                )
                mask = data["mask"][channel].astype(bool)
                support = mask & np.isfinite(target)
                y, x = patch(support, data["lat"], data["lon"])
                spectra.append(
                    dict(
                        origin=filename,
                        channel=str(name),
                        patch_y=y,
                        patch_x=x,
                        reference=spectrum(target, y, x).tolist(),
                        members=spectrum(members, y, x).tolist(),
                        mean=spectrum(members.mean(0), y, x).tolist(),
                    )
                )
                if index in (0, 4, 8):
                    temporal = "day-30 five-day bin" if channel < 2 else "monthly"
                    grid(
                        [target, members.mean(0), *members[:4]],
                        ["Observed reference", "Eight-member mean"]
                        + [f"Member {i + 1}" for i in range(4)],
                        mask,
                        data["lat"],
                        args.output / f"{path.stem}-{name}.png",
                        f"{path.stem}; {temporal}; {receipt['counts']} updates",
                        str(name),
                        show_loss_boundary=False,
                        distinguish_missing=True,
                    )
                if index == 0 and comparison is not None:
                    other_path = args.compare_root / filename
                    if digest(other_path) != comparison["files"][filename]:
                        raise ValueError(f"Changed comparison export: {filename}")
                    with np.load(other_path) as other:
                        for key in (
                            "mask",
                            "lat",
                            "lon",
                            "channel_names",
                            "monthly_reference",
                            "day30_surface_reference",
                        ):
                            np.testing.assert_array_equal(data[key], other[key])
                        other_members = (
                            other["day30"][:, channel]
                            if channel < 2
                            else other["monthly"][:, monthly_index]
                        )
                    fields, titles = [], []
                    for member_index in (None, 0, 1):
                        for values, steps in (
                            (other_members, comparison["sampling_steps"]),
                            (members, receipt["sampling_steps"]),
                        ):
                            fields.append(
                                values.mean(0)
                                if member_index is None
                                else values[member_index]
                            )
                            label = (
                                "Eight-member mean"
                                if member_index is None
                                else f"Member {member_index + 1}"
                            )
                            titles.append(f"{steps} steps: {label}")
                    grid(
                        fields,
                        titles,
                        mask,
                        data["lat"],
                        args.output / f"sampling-{path.stem}-{name}.png",
                        f"{path.stem}; matched checkpoint and initial noise; "
                        + (
                            "five-day surface bin"
                            if channel < 2
                            else "monthly interior"
                        ),
                        str(name),
                        show_loss_boundary=False,
                        distinguish_missing=True,
                    )
    (args.output / "spectra.json").write_text(json.dumps(spectra, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    for ax, name in zip(
        axes.flat, ("thetao_0", "zos", "thetao_9", "so_9"), strict=True
    ):
        records = [r for r in spectra if r["channel"] == name]
        k = np.arange(1, 17) / 32
        for key, label, color, marker in (
            ("reference", "Reference", "black", "x"),
            ("members", "Individual-member power (averaged)", "#0072b2", "o"),
            ("mean", "Power of ensemble mean", "#d55e00", "s"),
        ):
            spectral_values = np.array([r[key] for r in records])
            power = spectral_values.mean(tuple(range(spectral_values.ndim - 1)))
            ax.loglog(k, power, label=label, color=color, marker=marker, markersize=3)
        ax.set(title=name, xlabel="cycles/grid cell", ylabel="Window-normalized power")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    save_png(fig, args.output / "member-spectra.png", dpi=130)
    plt.close(fig)
    pooled = {}
    for task in ("surface", "interior"):
        weights = sum(np.array(row[task]["weight"]).sum() for row in calibration)
        values = {
            key: float(
                sum(np.array(row[task][key]).sum() for row in calibration) / weights
            )
            for key in (
                "mean_squared_error",
                "ensemble_variance",
                "fair_crps",
                "empirical_crps",
                "coverage_80",
                "coverage_90",
            )
        }
        values["rmse"] = values["mean_squared_error"] ** 0.5
        values["spread"] = values["ensemble_variance"] ** 0.5
        values["spread_rmse"] = values["spread"] / values["rmse"]
        ranks = rank_probabilities(calibration, task)
        values["rank_probabilities"] = ranks.tolist()
        values["outside_ensemble_fraction"] = float(ranks[0] + ranks[-1])
        pooled[task] = values
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, task in zip(axes, ("surface", "interior"), strict=True):
        ranks = np.array(pooled[task]["rank_probabilities"])
        positions = np.arange(len(ranks))
        offset = 0.18 if comparison else 0
        ax.bar(
            positions + offset,
            ranks,
            width=0.36 if comparison else 0.7,
            label=f"{receipt['sampling_steps']} steps",
            color="#d55e00",
        )
        if comparison:
            records = json.loads((args.compare_root / "calibration.json").read_text())
            ax.bar(
                positions - offset,
                rank_probabilities(records, task),
                width=0.36,
                label=f"{comparison['sampling_steps']} steps",
                color="#0072b2",
            )
        ax.axhline(1 / len(ranks), color="black", label="Exchangeable reference")
        ax.set(
            title=task,
            xlabel="Observation rank (0 / 8: outside ensemble)",
            ylabel="Wet-area weighted fraction",
            xticks=positions,
        )
        ax.legend(fontsize=8)
    save_png(fig, args.output / "rank-histograms.png", dpi=130)
    plt.close(fig)
    summary = dict(
        counts=receipt["counts"],
        score=receipt["score"],
        calibration=pooled,
        calibration_units="Observation-standardized, pooled finite wet-area support",
        spectra="32x32 entirely observed Pacific patch per field/date; not a global spectrum",
        checkpoint_sha256=receipt["checkpoint_sha256"],
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
