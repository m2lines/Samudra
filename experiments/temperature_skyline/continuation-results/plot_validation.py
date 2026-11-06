# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export archived per-depth validations and plot raw/EMA continuation curves."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--artifacts", type=Path, help="Optional archived arm directories to export"
)
args = parser.parse_args()
root = Path(__file__).parent
csv_path = root / "validation-per-depth.csv"
if args.artifacts:
    records = []
    aggregate = list(csv.DictReader((root / "validation-curve.csv").open()))
    for arm in ["regression", "absolute"]:
        initial = json.loads((root / f"initial-{arm}-validation.json").read_text())
        snapshots = [("initial", "raw", initial)]
        for path in sorted((args.artifacts / arm).glob("validation-*.json")):
            if "comparison" in path.name:
                continue
            snapshots.append(
                (
                    "continuation",
                    "ema" if "-ema-" in path.name else "raw",
                    json.loads(path.read_text()),
                )
            )
        assert len(snapshots) == 39, (arm, len(snapshots))
        for source, weights, snapshot in snapshots:
            assert snapshot["origins"] == initial["origins"]
            assert len(snapshot["per_depth"]) == 19
            match = [
                r
                for r in aggregate
                if r["arm"] == arm
                and r["source"] == source
                and r["weights"] == weights
                and int(r["step"]) == snapshot["step"]
            ]
            assert len(match) == 1
            assert np.isclose(
                float(match[0]["nmse"]),
                np.mean([r["nmse"] for r in snapshot["per_depth"]]),
                rtol=1e-10,
            )
            if arm == "absolute":
                assert np.isclose(
                    float(match[0]["nll"]),
                    np.mean([r["bin_nll"] for r in snapshot["per_depth"]]),
                    rtol=1e-10,
                )
            for row in snapshot["per_depth"]:
                assert row["count"] == 512
                records.append(
                    dict(
                        arm=arm,
                        source=source,
                        weights=weights,
                        step=snapshot["step"],
                        depth=row["depth"],
                        depth_m=row["depth_m"],
                        count=row["count"],
                        rmse_c=row["rmse_c"],
                        nmse=row["nmse"],
                        bin_nll=row["bin_nll"],
                        persistence_rmse_c=row["persistence_rmse_c"],
                    )
                )
    with csv_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

with csv_path.open() as stream:
    rows = list(csv.DictReader(stream))
plt.rcParams.update(
    {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
for arm, metric, title, filename in [
    (
        "regression",
        "rmse_c",
        "Regression validation RMSE",
        "validation-regression-depths.png",
    ),
    (
        "absolute",
        "rmse_c",
        "Classifier argmax validation RMSE",
        "validation-classifier-rmse-depths.png",
    ),
    (
        "absolute",
        "bin_nll",
        "Classifier validation bin NLL",
        "validation-classifier-nll-depths.png",
    ),
]:
    fig, axes = plt.subplots(5, 4, figsize=(13, 12), layout="constrained")
    for depth, ax in enumerate(axes.flat[:19]):
        selected = [r for r in rows if r["arm"] == arm and int(r["depth"]) == depth]
        initial = next(r for r in selected if r["source"] == "initial")
        for kind, color in [("raw", "#2581b8"), ("ema", "#d45a18")]:
            sequence = sorted(
                [
                    r
                    for r in selected
                    if r["source"] == "continuation" and r["weights"] == kind
                ],
                key=lambda r: int(r["step"]),
            )
            ax.plot(
                [int(r["step"]) / 1000 for r in sequence],
                [float(r[metric]) for r in sequence],
                color=color,
                lw=1.3,
                marker=".",
                ms=3,
                label=kind.upper(),
            )
        ax.axhline(
            float(initial[metric]),
            color="#555555",
            ls="--",
            lw=1,
            label="Initial raw checkpoint",
        )
        if metric == "rmse_c":
            ax.axhline(
                float(initial["persistence_rmse_c"]),
                color="#999999",
                ls=":",
                lw=1,
                label="Persistence",
            )
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.3g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: f"{x:.3g}"))
        ax.set_title(f"{float(initial['depth_m']):g} m", loc="left", fontweight="bold")
        ax.set_xlim(0, 8)
        ax.set_xticks([0, 2, 4, 6, 8])
        ax.grid(alpha=0.2)
        if depth % 4 == 0:
            ax.set_ylabel(
                "RMSE (°C, log scale)" if metric == "rmse_c" else "Bin NLL (nats)"
            )
        if depth >= 15:
            ax.set_xlabel("Continuation updates (thousands)")
    axes.flat[19].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[19].legend(handles, labels, loc="upper left", frameon=False)
    axes.flat[19].text(
        0,
        0.48,
        "512 cases/depth · 8 dates\n2013–2014 · 19 checkpoints\nSame raw/EMA cases\nLower is better\nDepth-specific y limits\nNo smoothing",
        va="top",
        transform=axes.flat[19].transAxes,
    )
    fig.suptitle(title + "\n33×33 crop · +5-day temperature · 19 depths", fontsize=16)
    fig.savefig(root / filename, dpi=110)
    plt.close(fig)
print(
    f"Plotted {len(rows)} per-depth validation rows; raw/EMA aggregates verified against validation-curve.csv on export."
)
