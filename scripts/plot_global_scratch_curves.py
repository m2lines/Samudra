#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare global mixed and scratch validation histories on two update axes."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    curves = json.loads(args.input.read_text())["curves"]
    assert curves["mixed"]["reference_sha256"] == curves["scratch"]["reference_sha256"]
    assert curves["mixed"]["complete"]["task_counts"] == {
        "om4": 8000,
        "observation": 8000,
    }
    assert curves["scratch"]["complete"]["task_counts"] == {
        "om4": 0,
        "observation": 16000,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    result: dict = {
        "reference_sha256": curves["mixed"]["reference_sha256"],
        "models": {},
        "definition": "Raw global v4 integrated-plus-spectral validation on identical frozen validation reference. Every logged validation is plotted; no smoothing. Star marks validation minimum and filled circles fixed observation milestones. These curves do not use test-normalized reporting scores.",
    }
    for key, name, color in [
        ("mixed", "Small conditioned mixed global", "#2167ad"),
        ("scratch", "Small conditioned scratch global", "#c87526"),
    ]:
        rows = curves[key]["validation"]
        best = min(rows, key=lambda r: r["validation/obs_score"])
        milestones = [50, 500, 2000, 8000] if key == "mixed" else [8000, 16000]
        fixed = {n: next(r for r in rows if r["observation"] == n) for n in milestones}
        result["models"][key] = {
            "name": name,
            "best_validation": best,
            "fixed_milestones": fixed,
            "best_through_8k": min(
                [r for r in rows if r["observation"] <= 8000],
                key=lambda r: r["validation/obs_score"],
            ),
        }
        for ax, field in zip(axes, ["global_step", "observation"], strict=True):
            ax.plot(
                [r[field] for r in rows],
                [r["validation/obs_score"] for r in rows],
                color=color,
                lw=1.25,
                label=name,
            )
            ax.scatter(
                [r[field] for r in fixed.values()],
                [r["validation/obs_score"] for r in fixed.values()],
                color=color,
                s=24,
                zorder=4,
            )
            ax.scatter(
                [best[field]],
                [best["validation/obs_score"]],
                color=color,
                marker="*",
                s=100,
                zorder=5,
            )
            ax.set_yscale("log")
            ax.grid(alpha=0.25)
            ax.set_xlim(0, 16500)
            ax.set_ylabel("Global integrated + spectral validation score (log axis)")
            ax.set_xlabel(
                "Total optimizer updates"
                if field == "global_step"
                else "Observation optimizer updates"
            )
            ax.legend(fontsize=8)
    fig.suptitle(
        "Mixed OM4 + observations versus observation-only; same validation reference"
    )
    fig.savefig(args.output / "global-validation-training-curves.png", dpi=140)
    plt.close(fig)
    (args.output / "validation-curve-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("GLOBAL_VALIDATION_CURVES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
