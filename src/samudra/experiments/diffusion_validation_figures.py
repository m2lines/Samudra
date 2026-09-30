# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot observed validation history and export the provisional plateau assessment."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.diffusion_plateau import assess, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-runs", type=Path, required=True)
    parser.add_argument("--extended-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    rows: list[dict[str, str | int | float]] = []
    assessments = {}
    for ax, seed in zip(axes, (1729, 1730), strict=True):
        short = sorted(read(args.short_runs / f"D-{seed}/observation"))
        extended = sorted(read(args.extended_runs / f"D-{seed}/observation"))
        if not short or not extended or short[-1][0] >= extended[0][0]:
            raise ValueError("Expected short training followed by continuation checks")
        assessment = assess(short, extended)
        assessments[str(seed)] = assessment
        for phase, records, color, marker in (
            ("Short", short, "#c84638", "D"),
            ("Extended", extended, "#268449", "P"),
        ):
            rows.extend(
                dict(seed=seed, phase=phase, step=step, composite=score)
                for step, score in records
            )
            ax.plot(
                [step for step, _ in records],
                [score for _, score in records],
                color=color,
                marker=marker,
                markersize=5,
                label=phase,
            )
        ax.axvline(short[-1][0], color="gray", linewidth=0.8, label="Budget extended")
        flag = (
            "provisional plateau"
            if assessment["provisional_plateau"]
            else "no plateau flag"
        )
        ax.set_title(f"Seed {seed}: {flag}")
        ax.set_xlabel("Cumulative observation-training updates")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Frozen validation composite (lower is better)")
    fig.suptitle("Validation history: eight-member ensemble-mean selection metric")
    fig.tight_layout()
    save_png(fig, args.output / "validation-history.png", dpi=110)
    plt.close(fig)
    with (args.output / "validation-history.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "plateau-assessment.json").write_text(
        json.dumps(assessments, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
