# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize the pinned initial-checkpoint map assessment, without refitting."""

import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parent
report = json.loads(gzip.decompress((root / "report/summary.json.gz").read_bytes()))
rows = [r for r in report["assessment"] if r["region"] == "Global"]
std = np.array(report["receipt"]["source"]["std"])[38:57]
names = ["regression", "samudra", "persistence", "classifier_mode"]
metrics = {
    name: float(np.mean([(r[name + "_rmse_c"] / std[r["depth"]]) ** 2 for r in rows]))
    for name in names
}
detectors = {}
for score in sorted({r["score"] for r in report["detection"]}):
    selected = [
        r
        for r in report["detection"]
        if r["region"] == "Global" and r["score"] == score
    ]
    detectors[score] = {
        key: float(np.mean([r[key] for r in selected]))
        for key in [
            "auroc",
            "average_precision",
            "spearman",
            "top_decile_sse_fraction",
            "top_decile_selected_fraction",
            "large_error_prevalence",
        ]
    }
result = {
    "aggregation": "Arithmetic mean across 19 depths; within depth, global assessment dates have equal weight and locations use the pinned area-sampling protocol.",
    "mean_depth_nmse": metrics,
    "regression_nmse_reduction_vs_samudra_percent": 100
    * (1 - metrics["regression"] / metrics["samudra"]),
    "regression_nmse_increase_vs_persistence_percent": 100
    * (metrics["regression"] / metrics["persistence"] - 1),
    "regression_depths_beating_samudra": sum(
        r["regression_rmse_c"] < r["samudra_rmse_c"] for r in rows
    ),
    "regression_depths_beating_persistence": sum(
        r["regression_rmse_c"] < r["persistence_rmse_c"] for r in rows
    ),
    "mean_depth_detection": detectors,
    "mean_depth_coverage": {
        str(level): float(np.mean([r[f"gold_coverage_{level}"] for r in rows]))
        for level in [50, 80, 90, 95]
    },
    "mean_depth_gold_false_alarm_fraction": float(
        np.mean([r["gold_false_alarm_fraction"] for r in rows])
    ),
}
(root / "headline-metrics.json").write_text(json.dumps(result, indent=2) + "\n")

fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
depths = [r["depth_m"] for r in rows]
for name, label in [
    ("regression", "Regression"),
    ("samudra", "Samudra"),
    ("persistence", "Persistence"),
]:
    axes[0].plot([r[name + "_rmse_c"] for r in rows], depths, ".-", label=label)
axes[0].set(
    xscale="log",
    xlabel="Temperature RMSE (°C; lower is better)",
    ylabel="Depth (m)",
    title="Four held-out 2016 forecast dates",
)
axes[0].invert_yaxis()
axes[0].legend(fontsize=8)
for score, label in [
    ("samudra_density_surprise", "Classifier surprise"),
    ("samudra_regression_disagreement", "Samudra − regression"),
    ("samudra_persistence_disagreement", "Samudra − persistence"),
]:
    selected = [
        r
        for r in report["detection"]
        if r["region"] == "Global" and r["score"] == score
    ]
    axes[1].plot(
        [r["auroc"] for r in selected],
        [r["depth_m"] for r in selected],
        ".-",
        label=label,
    )
axes[1].set(
    xlim=(0.5, 1),
    xlabel="Large-error AUROC (higher is better)",
    title="Detecting Samudra errors vs OM4",
)
axes[1].invert_yaxis()
axes[1].legend(fontsize=8)
levels = np.array([50, 80, 90, 95])
axes[2].plot([0, 100], [0, 100], "--", color="0.6", label="Nominal coverage")
axes[2].plot(
    levels,
    [100 * result["mean_depth_coverage"][str(level)] for level in levels],
    "o-",
    label="Observed; mean across depths",
)
axes[2].set(
    xlim=(40, 100),
    ylim=(40, 100),
    xlabel="Nominal central interval (%)",
    ylabel="OM4 targets covered (%)",
    title="Classifier intervals are too wide",
)
axes[2].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=0.2)
fig.suptitle(
    "Initial raw checkpoints · +5-day temperature · OM4 model reference", fontsize=13
)
fig.savefig(root / "assessment-summary.png", dpi=150)
plt.close(fig)
