# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare native temporal variability before and after extended observation fitting."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.diffusion_scratch_figures import DEPTHS, VARIABLES


def ratios(summary):
    """RMS increment magnitudes relative to truth, not increment prediction errors."""
    temporal = summary["temporal"]
    truth = np.asarray(temporal["truth_increment_mse"], dtype=float)
    if (
        truth.shape != (len(summary["errors"]["channels"]),)
        or not np.isfinite(truth).all()
        or (truth < 0).any()
    ):
        raise ValueError("Invalid native temporal reference")
    result = {}
    for name, key in (
        ("member", "member_increment_mse"),
        ("mean", "mean_increment_mse"),
    ):
        value = np.asarray(temporal[key], dtype=float)
        if (
            value.shape != truth.shape
            or not np.isfinite(value).all()
            or (value < 0).any()
        ):
            raise ValueError("Invalid predicted increment magnitude")
        result[name] = np.sqrt(
            np.divide(value, truth, out=np.full_like(truth, np.nan), where=truth > 0)
        )
    result["correlation"] = np.asarray(
        temporal["adjacent_member_correlation"], dtype=float
    )
    if (
        result["correlation"].shape != truth.shape
        or not np.isfinite(result["correlation"]).all()
    ):
        raise ValueError("Invalid adjacent-member correlation")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-runs", type=Path, required=True)
    parser.add_argument("--extended-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(14, 11), sharey=True)
    reference = None
    rows = []
    for phase, root, color, markers in (
        ("Short", args.short_runs, "#c84638", ("D", "X")),
        ("Extended", args.extended_runs, "#268449", ("P", "v")),
    ):
        for seed, marker in zip((1729, 1730), markers, strict=True):
            summary = json.loads(
                (root / f"D-{seed}/native-observation-summary.json").read_text()
            )
            contract = {
                k: summary["protocol"][k]
                for k in ("origins", "leads", "channels", "members")
            }
            if reference is None:
                reference = summary
            else:
                if contract != {k: reference["protocol"][k] for k in contract}:
                    raise ValueError("Temporal comparison cohorts differ")
                np.testing.assert_array_equal(
                    summary["temporal"]["truth_increment_mse"],
                    reference["temporal"]["truth_increment_mse"],
                )
            values = ratios(summary)
            names = summary["errors"]["channels"]
            for index, channel in enumerate(names):
                rows.append(
                    dict(
                        phase=phase,
                        seed=seed,
                        channel=channel,
                        unit={
                            "thetao": "degC",
                            "so": "practical_salinity",
                            "uo": "m/s",
                            "vo": "m/s",
                            "zos": "m",
                        }[channel.split("_")[0]],
                        member_ratio=float(values["member"][index]),
                        mean_ratio=float(values["mean"][index]),
                        correlation=float(values["correlation"][index]),
                        **{
                            f"{kind}_rms_change": float(
                                np.sqrt(
                                    summary["temporal"][f"{kind}_increment_mse"][index]
                                )
                            )
                            for kind in ("member", "mean", "truth")
                        },
                    )
                )
            for column, variable in enumerate(VARIABLES):
                select = [names.index(f"{variable}_{i}") for i in range(len(DEPTHS))]
                for row, metric in enumerate(("member", "mean", "correlation")):
                    axes[row, column].plot(
                        values[metric][select],
                        DEPTHS,
                        color=color,
                        marker=marker,
                        markersize=4,
                        label=f"{phase} {seed}",
                    )
    for column, variable in enumerate(VARIABLES):
        axes[0, column].set_title(variable)
        for row in (0, 1):
            axes[row, column].set_xscale("log")
            axes[row, column].axvline(1, color="black", linewidth=0.8)
        axes[0, column].set_xlabel("Member RMS change / OM4 RMS change")
        axes[1, column].set_xlabel("Mean RMS change / OM4 RMS change")
        axes[2, column].set_xlabel("Adjacent member-deviation correlation")
        axes[2, column].axvline(0, color="black", linewidth=0.8)
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Depth (m)")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Native OM4 temporal diagnostics: 24 origins, five-day changes through day 30, ±60°\nMagnitude ratios do not measure increment error; independent readouts are not coherent trajectories"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png(fig, args.output / "temporal-variability.png", dpi=110)
    plt.close(fig)
    with (args.output / "temporal-variability.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
