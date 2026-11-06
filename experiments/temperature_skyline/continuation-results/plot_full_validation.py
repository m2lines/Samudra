# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot recorded initial and continued validation histories at every depth."""

import csv
import gzip
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator

root = Path(__file__).parent
with gzip.open(root / "validation-full-history.csv.gz", "rt") as f:
    rows = list(csv.DictReader(f))
plt.rcParams.update(
    {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
for arm, metric, title, filename, offset in [
    (
        "regression",
        "rmse_c",
        "Regression validation RMSE — full recorded history",
        "validation-regression-full.png",
        18000,
    ),
    (
        "absolute",
        "bin_nll",
        "Classifier validation bin NLL — full recorded history",
        "validation-likelihood-full.png",
        20000,
    ),
]:
    fig, axes = plt.subplots(5, 4, figsize=(13, 12), layout="constrained")
    for depth, ax in enumerate(axes.flat[:19]):
        d = [r for r in rows if r["arm"] == arm and int(r["depth"]) == depth]
        initial = sorted(
            [r for r in d if r["phase"] == "initial" and int(r["step"]) <= offset],
            key=lambda r: int(r["step"]),
        )
        assert initial[0]["step"] == "2000"
        ax.plot(
            [int(r["step"]) / 1000 for r in initial],
            [float(r[metric]) for r in initial],
            color="#168c83",
            marker=".",
            ms=3,
            lw=1.3,
            label="Initial raw",
        )
        for kind, color in [("raw", "#2581b8"), ("ema", "#d45a18")]:
            sequence = sorted(
                [r for r in d if r["phase"] == "continuation" and r["weights"] == kind],
                key=lambda r: int(r["cumulative_step"]),
            )
            ax.plot(
                [int(r["cumulative_step"]) / 1000 for r in sequence],
                [float(r[metric]) for r in sequence],
                color=color,
                marker=".",
                ms=3,
                lw=1.3,
                label="Continued " + kind.upper(),
            )
        if arm == "regression":
            unused = next(
                r for r in d if r["phase"] == "initial" and r["step"] == "20000"
            )
            ax.plot(
                20,
                float(unused[metric]),
                marker="x",
                color="#777777",
                ls="none",
                ms=5,
                label="Initial 20k (not resumed)",
            )
            ax.axhline(
                float(initial[0]["persistence_rmse_c"]),
                color="#999999",
                ls=":",
                lw=1,
                label="Persistence",
            )
            ax.set_yscale("log")
            lower, upper = ax.get_ylim()
            ax.set_yticks(np.geomspace(float(lower), float(upper), 4))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: f"{x:.3g}"))
        ax.axvline(
            offset / 1000,
            color="#555555",
            lw=1,
            ls="--",
            label="Continuation / LR restart",
        )
        ax.set_title(f"{float(d[0]['depth_m']):g} m", loc="left", fontweight="bold")
        ax.set_xlim(0, 28)
        ax.set_xticks([0, 10, 20, 28])
        ax.grid(alpha=0.2)
        if depth % 4 == 0:
            ax.set_ylabel(
                "RMSE (°C, log scale)" if metric == "rmse_c" else "Bin NLL (nats)"
            )
        if depth >= 15:
            ax.set_xlabel("Cumulative updates (thousands)")
        assert all(np.isfinite(float(r[metric])) for r in d)
    axes.flat[19].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[19].legend(handles, labels, loc="upper left", frameon=False, fontsize=8)
    axes.flat[19].text(
        0,
        0.35,
        "First validation: update 2,000\nNo update-0/1 measurement\n512 cases/depth · 8 fixed dates\nBatch: 32 initial → 256 continued\nLower is better; y limits vary",
        va="top",
        fontsize=8,
        transform=axes.flat[19].transAxes,
    )
    fig.suptitle(title + "\n33×33 crop · +5-day temperature · 19 depths", fontsize=15)
    fig.savefig(root / filename, dpi=110)
    plt.close(fig)
print("Rendered two full-history figures from", len(rows), "per-depth records")
