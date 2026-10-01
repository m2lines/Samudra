# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot native velocity controls before and after observation adaptation."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-runs", type=Path, required=True)
    parser.add_argument("--pretraining-runs", type=Path)
    parser.add_argument("--previous-latent-runs", type=Path)
    parser.add_argument("--physical-native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, layout="constrained")
    for family, prefix, markers in (
        ("physical", "B", ("o", "s")),
        ("latent", "D", ("D", "X")),
    ):
        for seed, marker in zip((1729, 1730), markers, strict=True):
            phases = [("om4", "tab:blue"), ("observation", "tab:red")]
            if family == "latent" and args.previous_latent_runs:
                phases.insert(1, ("short", "tab:orange"))
            for phase, color in phases:
                root = (
                    (
                        (args.pretraining_runs or args.latent_runs)
                        if phase == "om4"
                        else args.latent_runs
                    )
                    / f"D-{seed}"
                    if family == "latent"
                    else args.physical_native / f"B-{seed}"
                )
                if phase == "short":
                    root = args.previous_latent_runs / f"D-{seed}"
                source_phase = "observation" if phase == "short" else phase
                path = root / (
                    f"native-{source_phase}-summary.json"
                    if family == "latent"
                    else f"{phase}-summary.json"
                )
                data = json.loads(path.read_text())
                records = [
                    r for r in data["results"] if r["region"] == "scored_latitudes"
                ]
                points = []
                for r in records:
                    score = r["ensemble"]["scores"]
                    row = dict(
                        family=family,
                        seed=seed,
                        phase=phase,
                        lead_days=r["lead_days"],
                        rmse=np.sqrt(score["mean_squared_error"]),
                        spread=np.sqrt(score["ensemble_variance"]),
                        fair_crps=score["fair_crps"],
                        zero_rmse=np.sqrt(
                            r["zero_velocity"]["scores"]["mean_squared_error"]
                        ),
                    )
                    rows.append(row)
                    points.append(row)
                axis = axes[int(family == "latent")]
                axis.plot(
                    [r["lead_days"] for r in points],
                    [r["rmse"] for r in points],
                    marker=marker,
                    color=color,
                    label=f"{prefix}-{seed} "
                    + (
                        "stopped"
                        if family == "latent"
                        and args.previous_latent_runs
                        and phase == "observation"
                        else phase
                    ),
                )
                if seed == 1729 and phase == "om4":
                    axis.plot(
                        [r["lead_days"] for r in points],
                        [r["zero_rmse"] for r in points],
                        color="black",
                        marker="^",
                        label="Zero velocity",
                    )
    for axis, title in zip(
        axes, ("Physical-state diffusion", "Persistent latent state"), strict=True
    ):
        axis.set(
            title=title,
            xlabel="Lead (days)",
            ylabel="Velocity ensemble-mean RMSE (m/s)",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.savefig(args.output / "native-velocity.png", dpi=150)
    plt.close(fig)
    with (args.output / "native-velocity.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
