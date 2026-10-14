# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Reduce fixed-validation cooldown sweeps and original EMA validation logs."""

import argparse
import ast
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_score import (  # type: ignore[import-not-found]  # noqa: E402
    save_csv,
    save_figure,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / "diagnostic-results"
    out.mkdir(exist_ok=True)
    scores, families, completions = [], [], []
    metrics = ["rmse_normalized", "crps_empirical_normalized", "coverage_member_range"]
    for arm, job in [("F", 19831109), ("M", 19831110)]:
        root = args.artifacts / f"eval-adaptation-diagnostic-{arm}-{job}"
        for epoch in [0, 2, 4, 6, 8, 10]:
            data = pd.read_csv(root / f"epoch-{epoch}/forecast/metrics-rank0.csv")
            assert len(data) == 539 and np.isfinite(data[metrics]).all().all()
            assert data.origin.str.startswith("2013-10-10").all()
            data["family"] = data.variable.str.split("_").str[0]
            grouped = data.groupby("lead_days")[metrics].mean().reset_index()
            scores.append(
                grouped.assign(
                    arm=arm, epoch=epoch, weights="initial EMA" if epoch == 0 else "raw"
                )
            )
            families.append(
                data.groupby(["lead_days", "family"])[metrics]
                .mean()
                .reset_index()
                .assign(arm=arm, epoch=epoch)
            )
            if epoch in [0, 4, 10]:
                c = pd.read_csv(root / f"epoch-{epoch}/completion/metrics.csv")
                assert len(c) == 1324 and np.isfinite(c[metrics]).all().all()
                c = c.groupby(["task", "origin"])[metrics].mean().reset_index()
                completions.append(
                    c.groupby("task")[metrics]
                    .mean()
                    .reset_index()
                    .assign(arm=arm, epoch=epoch)
                )
        text = (args.artifacts / f"cooldown-{arm}-experiment.log").read_text()
        for match in re.finditer(r"Annual validation epoch (\d+): (\{[^\n]+\})", text):
            epoch = int(match[1])
            values = ast.literal_eval(match[2])
            rows = []
            for lead in [10, 20, 40, 90, 180, 360]:
                rows.append(
                    dict(
                        arm=arm,
                        epoch=epoch,
                        weights="EMA",
                        lead_days=lead,
                        rmse_normalized=values[f"rollout_val/origin0/{lead}d/rmse"],
                        crps_empirical_normalized=values[
                            f"rollout_val/origin0/{lead}d/crps"
                        ],
                    )
                )
            scores.append(pd.DataFrame(rows))
    score = pd.concat(scores, ignore_index=True)
    family = pd.concat(families, ignore_index=True)
    completion = pd.concat(completions, ignore_index=True)
    save_csv(score, out / "forecast.csv")
    save_csv(family, out / "forecast-family.csv")
    save_csv(completion, out / "completion.csv")
    selection = (
        score[score.lead_days.isin([10, 20, 40, 90, 180, 360])]
        .groupby(["arm", "epoch", "weights"])[
            ["rmse_normalized", "crps_empirical_normalized"]
        ]
        .mean()
        .reset_index()
    )
    save_csv(selection, out / "selection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, arm in zip(axes, ["F", "M"], strict=True):
        for kind, color, marker in [("raw", "#d55e00", "s"), ("EMA", "#0072b2", "o")]:
            d = selection[
                (selection.arm == arm) & (selection.weights.isin([kind, "initial EMA"]))
            ].sort_values("epoch")
            ax.plot(
                d.epoch,
                d.crps_empirical_normalized,
                color=color,
                marker=marker,
                label=kind,
            )
        ax.set(
            title=f"{arm}: fixed validation origin",
            xlabel="Cooldown epoch",
            ylabel="Mean CRPS across selection leads",
        )
        ax.legend()
        ax.grid(alpha=0.2)
    save_figure(fig, out / "curves.jpg")
    print(selection.to_string(index=False))
    print(completion[completion.arm == "M"].to_string(index=False))


if __name__ == "__main__":
    main()
