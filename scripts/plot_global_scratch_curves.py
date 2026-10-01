#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare global mixed and scratch validation histories on two update axes."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth_scores(rows, half_life):
    """Causal EMA in linear score space, with decay per total optimizer update."""
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError("EMA half-life must be positive and finite")
    smoothed = []
    previous_step = None
    value = None
    for row in rows:
        step = row["global_step"]
        score = row["validation/obs_score"]
        if not math.isfinite(score) or score <= 0:
            raise ValueError("Expected finite positive validation scores")
        if previous_step is not None:
            if step <= previous_step:
                raise ValueError("Validation steps must be strictly increasing")
            decay = 2 ** (-(step - previous_step) / half_life)
            value = decay * value + (1 - decay) * score
        else:
            value = score
        smoothed.append(value)
        previous_step = step
    if not smoothed:
        raise ValueError("Empty validation history")
    return smoothed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ema-half-life", type=float, default=500.0)
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
        "definition": "Global v4 integrated-plus-spectral validation on identical frozen validation reference. Every logged validation is plotted as a faint raw trace with a bold causal EMA overlay. Filled circles mark only the fixed mixed final and scratch 8k/16k report checkpoints. These curves do not use test-normalized reporting scores; EMA does not select checkpoints.",
        "smoothing": {
            "half_life_total_optimizer_updates": args.ema_half_life,
            "formula": "ema[0] = score[0]; decay = 2 ** (-delta_global_step / half_life); ema[i] = decay * ema[i-1] + (1-decay) * score[i]",
            "space": "linear validation score, before applying log display axis",
            "causal": True,
            "axes": "The same smoothed scores appear on both axes; only horizontal coordinates change.",
        },
    }
    for key, name, color in [
        ("mixed", "Small conditioned mixed global", "#2167ad"),
        ("scratch", "Small conditioned scratch global", "#c87526"),
    ]:
        rows = curves[key]["validation"]
        best = min(rows, key=lambda r: r["validation/obs_score"])
        milestones = [8000] if key == "mixed" else [8000, 16000]
        fixed = {n: next(r for r in rows if r["observation"] == n) for n in milestones}
        ema = smooth_scores(rows, args.ema_half_life)
        result["models"][key] = {
            "name": name,
            "raw_validation": rows,
            "ema_validation_scores": ema,
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
                lw=0.8,
                alpha=0.3,
                label=name + " (raw)",
            )
            ax.plot(
                [r[field] for r in rows],
                ema,
                color=color,
                lw=2.0,
                label=name + " (EMA)",
            )
            ax.scatter(
                [r[field] for r in fixed.values()],
                [r["validation/obs_score"] for r in fixed.values()],
                color=color,
                s=24,
                zorder=4,
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
        "Mixed OM4 + observations versus observation-only; same validation reference\n"
        f"Raw + EMA; half-life {args.ema_half_life:g} total optimizer updates"
    )
    fig.savefig(args.output / "global-validation-training-curves.png", dpi=140)
    plt.close(fig)
    (args.output / "validation-curve-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("GLOBAL_VALIDATION_CURVES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
