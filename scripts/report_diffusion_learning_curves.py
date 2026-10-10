#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched checkpoint learning curves and explicitly conditional budget scenarios."""

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_diffusion_interventions import (  # type: ignore[import-not-found]
    COMPONENTS,
    ORIGINS,
    endpoint,
    manifests,
    pool,
    read,
    sha,
    table,
)

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.diffusion_interventions import V2_ARMS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--presentation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.presentation, "rt") as f:
        presentation = json.load(f)
    expected = manifests(presentation["models"]["U-global"]["inputs"])
    climate = presentation["metrics"]["Training seasonal climatology"]
    rows, evidence = [], []
    lead_rows: list[dict[str, Any]] = []
    paths = [
        ("parent", 0, args.baseline / "final-annual-steps32"),
        ("deterministic", 0, args.baseline / "global-endpoint-annual"),
    ]
    paths += [
        (arm, step, args.root / "learning-curves" / arm / f"step-{step:04d}")
        for arm in V2_ARMS
        for step in (32, 64, 128)
    ]
    paths += [(arm, 256, args.root / "evaluation" / arm / "annual") for arm in V2_ARMS]
    for arm, updates, directory in paths:
        signature = read(directory / "input.json")
        complete = read(directory / "COMPLETE.json")
        assert manifests(signature) == expected
        assert complete["inputs"] == signature
        if updates:
            assert signature["counts"] == {
                "om4": 8000 + updates // 2,
                "observation": 8000 + updates // 2,
            }
            assert signature["step"] == 16000 + updates
            assert signature["sampling_steps"] == 32 and signature["members"] == 8
            assert signature["intervention"]["arm"] == arm
            assert signature["intervention"]["updates"] == 256
        records = {origin: read(directory / f"{origin}.json") for origin in ORIGINS}
        evidence.append(
            dict(
                arm=arm,
                updates=updates,
                inputs=signature,
                metric_sha256={o: sha(directory / f"{o}.json") for o in ORIGINS},
            )
        )
        for day in (30, 365):
            normal = pool([endpoint(climate[o], o, day) for o in ORIGINS])
            for origin in [*ORIGINS, "pooled"]:
                origins = ORIGINS if origin == "pooled" else [origin]
                values = pool([endpoint(records[o], o, day) for o in origins])
                rows.append(
                    dict(
                        arm=arm,
                        updates=updates,
                        added_om4_updates=updates // 2,
                        added_observation_updates=updates // 2,
                        origin=origin,
                        day=day,
                        score=sum(values[k] / normal[k] for k in COMPONENTS) / 4,
                        **values,
                    )
                )
        for lead in records[ORIGINS[0]]["leads"]:
            for key in ("sst_rmse", "velocity_rmse"):
                lead_rows.append(
                    dict(
                        arm=arm,
                        updates=updates,
                        day=int(lead),
                        metric=key,
                        rmse=float(
                            np.sqrt(
                                np.mean(
                                    [
                                        records[o]["leads"][lead][key] ** 2
                                        for o in ORIGINS
                                    ]
                                )
                            )
                        ),
                    )
                )
    table(args.output / "learning-curves.csv", rows)
    table(args.output / "error-versus-lead.csv", lead_rows)
    (args.output / "provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")

    def score(arm, step, day):
        return next(
            r["score"]
            for r in rows
            if r["arm"] == arm
            and r["updates"] == step
            and r["origin"] == "pooled"
            and r["day"] == day
        )

    scenarios = []
    for arm in V2_ARMS:
        ratio = score(arm, 256, 365) / score(arm, 128, 365)
        scenarios.append(
            dict(
                arm=arm,
                last_doubling_ratio=ratio,
                score256=score(arm, 256, 365),
                same_fractional_gain_at512=score(arm, 256, 365) * ratio,
                same_fractional_gain_at1024=score(arm, 256, 365) * ratio**2,
                no_further_gain=score(arm, 256, 365),
            )
        )
    table(args.output / "conditional-budget-scenarios.csv", scenarios)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, day in zip(axes, (30, 365), strict=True):
        for arm in V2_ARMS:
            ax.plot(
                [0, 32, 64, 128, 256],
                [score("parent", 0, day)]
                + [score(arm, s, day) for s in (32, 64, 128, 256)],
                marker="o",
                label=arm,
            )
        ax.axhline(
            score("deterministic", 0, day), color="black", label="deterministic 8k/8k"
        )
        ax.set(
            title=f"Day {day}",
            xlabel="Added updates (half OM4, half observations)",
            ylabel="Four-component climatology-normalized RMSE",
        )
        if day == 365:
            ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    save_png(fig, args.output / "learning-curves.png", dpi=140)
    fig.savefig(args.output / "learning-curves.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    for ax, arm in zip(axes.flat, V2_ARMS, strict=True):
        for model, step in [
            ("parent", 0),
            *((arm, s) for s in (32, 64, 128, 256)),
            ("deterministic", 0),
        ]:
            r = sorted(
                [
                    r
                    for r in lead_rows
                    if r["arm"] == model
                    and r["updates"] == step
                    and r["metric"] == "sst_rmse"
                ],
                key=lambda r: r["day"],
            )
            ax.plot(
                [x["day"] for x in r],
                [x["rmse"] for x in r],
                marker="o",
                label=str(step) if step else model,
            )
        ax.set(
            title=arm,
            xlabel="Forecast lead (days)",
            ylabel="SST RMSE (°C)",
            yscale="log",
        )
    axes.flat[0].legend(fontsize=7)
    save_png(fig, args.output / "error-versus-lead.png", dpi=120)
    plt.close(fig)
    (args.output / "interpretation.txt").write_text(
        "These are checkpoints along a single 256-update cosine-LR trajectory per arm, not separately optimized training budgets. Annual cases are exploratory and reused. The 512/1024 values are arithmetic scenarios assuming the last doubling's fractional gain repeats; they are not forecasts or confidence bounds. No-further-gain is an alternative scenario. Prefer physical component and lead curves to extrapolating only a blended score.\n"
    )


if __name__ == "__main__":
    main()
