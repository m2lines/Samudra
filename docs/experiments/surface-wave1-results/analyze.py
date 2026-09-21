# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize collected wave metrics; does not submit jobs or select checkpoints."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import matplotlib
from artifact_io import artifact_exists, open_artifact

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
modes = ("inferred", "true", "inferred_persistence", "climatology")
regions = ("global", "tropics", "extratropics")
leads = (5, 10, 15, 20, 25, 30)


class MetricSummary(TypedDict):
    model: str
    mode: str
    region: str
    lead_days: int
    variable: str
    normalized_rmse: float
    levels: int
    origins: int


summary: list[MetricSummary] = []
for task in ("ar", "direct"):
    path = root / task / "heldout_metrics.csv"
    if not artifact_exists(path):
        continue
    rows = list(csv.DictReader(open_artifact(path)))
    manifest = json.loads((root / task / "manifest.json").read_text())
    expected = {
        (mode, region, lead, ch)
        for mode in modes
        for region in regions
        for lead in leads
        for ch in manifest["channels"]
    }
    actual = {(r["mode"], r["region"], int(r["lead_days"]), r["channel"]) for r in rows}
    assert actual == expected and len(actual) == len(rows) == 5544
    assert {int(r["origins"]) for r in rows} == {99}
    for r in rows:
        for key in (
            "normalized_rmse",
            "physical_rmse",
            "prediction_second_moment",
            "target_second_moment",
        ):
            assert math.isfinite(float(r[key]))
    groups = defaultdict(list)
    for r in rows:
        variable = "sst" if r["channel"] == "thetao_0" else r["channel"].split("_")[0]
        groups[(r["mode"], r["region"], int(r["lead_days"]), variable)].append(
            float(r["normalized_rmse"]) ** 2
        )
    for (mode, region, lead, var), mse in sorted(groups.items()):
        summary.append(
            dict(
                model=task,
                mode=mode,
                region=region,
                lead_days=lead,
                variable=var,
                normalized_rmse=math.sqrt(sum(mse) / len(mse)),
                levels=len(mse),
                origins=99,
            )
        )
with (root / "grouped_metrics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
lookup = {
    (r["model"], r["mode"], r["region"], r["lead_days"], r["variable"]): r[
        "normalized_rmse"
    ]
    for r in summary
}
models = sorted({r["model"] for r in summary})
fig, axes = plt.subplots(
    len(models),
    2,
    figsize=(10, 4 * len(models)),
    squeeze=False,
    constrained_layout=True,
)
styles = {
    "inferred": ("#2166ac", "-", "Inferred interior + evolution"),
    "true": ("#1b9e77", "--", "True interior + evolution"),
    "inferred_persistence": ("#d95f02", ":", "Inferred-state persistence"),
    "climatology": ("#777777", "-.", "Monthly climatology"),
}
for i, task in enumerate(models):
    for j, (var, title) in enumerate(
        [("thetao", "Subsurface temperature"), ("so", "Salinity")]
    ):
        ax = axes[i, j]
        for mode in modes:
            color, style, label = styles[mode]
            ax.plot(
                leads,
                [lookup[(task, mode, "global", lead, var)] for lead in leads],
                color=color,
                linestyle=style,
                marker={
                    "inferred": "o",
                    "true": "^",
                    "inferred_persistence": "s",
                    "climatology": "D",
                }[mode],
                markersize=4,
                markerfacecolor="none" if mode != "inferred" else color,
                label=label,
            )
        ax.set_title(task.upper() + ": " + title)
        ax.set_xlabel("Lead (days)")
        ax.set_ylabel("Normalized RMSE")
        ax.set_xticks(leads)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, handlelength=4, markerscale=1.3)
fig.suptitle(
    "Wave 1: OM4-only held-out hindcasts, 99 start dates\nGlobal errors; equal weight per depth level; five-day data",
    fontsize=12,
)
fig.savefig(root / "heldout_ts_leads.png", dpi=160)
fig.savefig(root / "heldout_ts_leads.svg")
plt.close(fig)
lines = [
    (
        "# Collected wave 1 results"
        if set(models) == {"ar", "direct"}
        else "# Collected wave 1 results (interim)"
    ),
    "",
    "OM4-only hindcasts at five-day cadence; these do not measure observational skill, daily outputs, or continuous long rollouts.",
    "",
    (
        "Both forecast models completed full held-out evaluation."
        if set(models) == {"ar", "direct"}
        else "One forecast model is still pending. This comparison is interim."
    ),
    "",
    "Normalized RMSE uses equal weights across levels within each variable. Subsurface temperature excludes SST.",
    "",
    "| Model | Region | Lead | T RMSE vs persistence | S RMSE vs persistence | T RMSE vs climatology | S RMSE vs climatology |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
]
for task in models:
    for region in regions:
        for lead in (5, 15, 30):
            values = []
            for ref in ("inferred_persistence", "climatology"):
                for var in ("thetao", "so"):
                    value = lookup[(task, "inferred", region, lead, var)]
                    baseline = lookup[(task, ref, region, lead, var)]
                    values.append(100 * (1 - value / baseline))
            lines.append(
                f"| {task} | {region} | {lead} | "
                + " | ".join(f"{v:+.1f}%" for v in values)
                + " |"
            )
lines += [
    "",
    "Positive percentages mean lower RMSE. Negative values mean worse errors. These aggregate CSVs do not support uncertainty intervals over forecast origins.",
    "",
    "Direct pretraining and joint tuning were stopped early based on validation. Its selected joint checkpoint is the pre-joint starting model; joint tuning did not improve the selection metric.",
    "",
    "Training was stopped using validation. Realized compute differs substantially between models; this is not an equal-compute architecture ranking. See the final report and final-accounting.psv for all attempts.",
]
(root / "metrics-summary.md").write_text("\n".join(lines) + "\n")
print(
    "Verified metrics and wrote grouped_metrics.csv, heldout_ts_leads.{png,svg}, metrics-summary.md"
)
