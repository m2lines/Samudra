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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((args.root / "COMPLETE.json").read_text())
    calibration = json.loads((args.root / "calibration.json").read_text())
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
        pooled[task] = values
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
