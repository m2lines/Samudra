# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Render cached matched Samudra / small all-fields classifier comparisons."""

import csv
import gzip
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "small-regression-25k/report"
REGIONS = [
    ("north-pacific", "North Pacific"),
    ("gulf-stream", "Gulf Stream"),
    ("agulhas", "Agulhas"),
]
DEPTHS = [(0, 2.5), (7, 250), (13, 1850)]
MODELS = [
    ("gold", "OM4", "black"),
    ("new-all-fields", "Small all-fields classifier", "#d55e00"),
    ("samudra", "Samudra", "#0072b2"),
    ("persistence", "Persistence", "#999900"),
]


def main():
    metrics = list(csv.DictReader((SOURCE / "map-metrics.csv").open()))
    selected = [
        r
        for r in metrics
        if r["field"] == "thetao" and r["model"] in {m[0] for m in MODELS}
    ]
    with (HERE / "map-metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    with np.load(SOURCE / "source-fields.npz") as archive:
        for region, label in REGIONS:
            fig = plt.figure(figsize=(16, 10.6), dpi=100)
            for row, (z, depth) in enumerate(DEPTHS):
                prefix = f"{region}-z{z:02d}/thetao/"
                gold = archive[prefix + "gold"]
                wet = archive[prefix + "wet"]
                area = archive[prefix + "area"]
                classifier = archive[prefix + "new-all-fields"]
                samudra = archive[prefix + "samudra"]
                fields = [gold, classifier, samudra]
                low = min(float(v[wet].min()) for v in fields)
                high = max(float(v[wet].max()) for v in fields)
                errors = [classifier - gold, samudra - gold]
                limit = max(float(abs(v[wet]).max()) for v in errors)
                for col, values in enumerate(fields + errors):
                    ax = fig.add_axes(
                        (
                            (45 + col * 300) / 1600,
                            (710 - row * 330) / 1060,
                            256 / 1600,
                            256 / 1060,
                        )
                    )
                    im = ax.imshow(
                        np.ma.masked_where(~wet, values),
                        origin="lower",
                        interpolation="nearest",
                        resample=False,
                        cmap="viridis" if col < 3 else "RdBu_r",
                        vmin=low if col < 3 else -limit,
                        vmax=high if col < 3 else limit,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if row == 0:
                        ax.set_title(
                            [
                                "OM4",
                                "Small all-fields classifier",
                                "Samudra",
                                "Classifier − OM4",
                                "Samudra − OM4",
                            ][col],
                            fontsize=11,
                        )
                    if col == 0:
                        ax.set_ylabel(f"{depth:g} m", fontsize=12)
                    if col in (1, 3):
                        ax.add_patch(
                            Rectangle(
                                (23, 23),
                                17,
                                17,
                                fill=False,
                                ls="--",
                                lw=0.8,
                                color="black",
                            )
                        )
                    if col >= 3:
                        score = np.sqrt(np.average(values[wet] ** 2, weights=area[wet]))
                        ax.set_xlabel(f"Area RMSE {score:.5f} °C", fontsize=10)
                    if col in (2, 4):
                        cax = fig.add_axes(
                            (
                                (45 + col * 300 + 264) / 1600,
                                (710 - row * 330) / 1060,
                                9 / 1600,
                                256 / 1060,
                            )
                        )
                        fig.colorbar(im, cax=cax).ax.tick_params(labelsize=8)
            fig.suptitle(
                f"{label} temperature · +5 days · target 2016-07-21\n64×64 native patches · 4×4 pixels per cell · dashed box = classifier 17×17 context · all scales in °C",
                y=0.985,
                fontsize=14,
            )
            fig.savefig(HERE / f"{region}-maps.png", dpi=100)
            plt.close(fig)
    with gzip.open(SOURCE / "temperature-spectra.csv.gz", "rt") as handle:
        spectra = [
            r for r in csv.DictReader(handle) if r["model"] in {m[0] for m in MODELS}
        ]
    with gzip.open(HERE / "spectra.csv.gz", "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spectra[0]))
        writer.writeheader()
        writer.writerows(spectra)
    for kind in ["field", "error"]:
        fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True)
        for row, (region, label) in enumerate(REGIONS):
            for col, (_, depth) in enumerate(DEPTHS):
                ax = axes[row, col]
                for key, name, color in MODELS:
                    data = [
                        r
                        for r in spectra
                        if r["region"] == region
                        and float(r["depth_m"]) == depth
                        and r["kind"] == kind
                        and r["model"] == key
                    ]
                    if not data:
                        continue
                    ax.loglog(
                        [float(r["cycles_per_cell"]) for r in data],
                        [float(r["power_C2"]) for r in data],
                        label=name,
                        color=color,
                        ls=":" if key == "persistence" else "-",
                        lw=1.7,
                    )
                ax.set_title(f"{label} · {depth:g} m")
                ax.axvspan(0.25, 0.5, alpha=0.1, color="gray")
                ax.grid(alpha=0.2)
                if row == 2:
                    ax.set_xlabel("Cycles / native grid cell")
                if col == 0:
                    ax.set_ylabel("Power per shell (°C²)")
        fig.suptitle(
            f"Temperature {kind} spectra · target 2016-07-21\nShaded wavelengths 2–4 cells · matched wet tiles · plane removed + periodic Hann",
            fontsize=14,
            y=0.99,
        )
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=4,
            frameon=False,
        )
        fig.subplots_adjust(
            top=0.86, bottom=0.07, hspace=0.28, wspace=0.28, left=0.08, right=0.98
        )
        fig.savefig(HERE / f"{kind}-spectra.png", dpi=105)
        plt.close(fig)
    with gzip.open(SOURCE / "validation-channels.csv.gz", "rt") as handle:
        curves = [
            r
            for r in csv.DictReader(handle)
            if r["run"] == "all-fields-both-25k" and r["field"] == "thetao"
        ]
    assert curves
    with gzip.open(HERE / "validation-temperature.csv.gz", "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    field_rows = list(csv.DictReader((SOURCE / "validation-fields.csv").open()))
    for weights, style in [("ema", "-"), ("raw", "--")]:
        data = [
            r
            for r in field_rows
            if r["run"] == "all-fields-both-25k"
            and r["field"] == "thetao"
            and r["weights"] == weights
            and int(r["step"]) > 0
        ]
        data.sort(key=lambda r: int(r["step"]))
        for ax, column in zip(axes, ["normalized_rmse", "wet_cell_nll"], strict=True):
            ax.plot(
                [int(r["step"]) for r in data],
                [float(r[column]) for r in data],
                style,
                label=weights.upper(),
            )
    axes[0].axhline(
        float(data[0]["persistence_normalized_rmse"]),
        color="#999900",
        ls=":",
        label="Persistence",
    )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Normalized temperature RMSE")
    axes[1].set_ylabel("Wet-cell hard-label NLL")
    fig.suptitle(
        "Small all-fields classifier validation · four dates, 64 date-cells per channel\nSamudra validation history on these samples is unavailable"
    )
    fig.savefig(HERE / "validation-curves.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(4, 5, figsize=(16, 11), layout="constrained")
    for depth, ax in enumerate(axes.flat):
        data = [
            r
            for r in curves
            if int(r["depth"]) == depth and r["weights"] == "ema" and int(r["step"]) > 0
        ]
        data.sort(key=lambda r: int(r["step"]))
        if not data:
            ax.axis("off")
            continue
        ax.loglog(
            [int(r["step"]) for r in data],
            [float(r["normalized_rmse"]) for r in data],
            color="#d55e00",
        )
        ax.axhline(
            float(data[0]["persistence_normalized_rmse"]), color="#999900", ls=":"
        )
        ax.set_title(f"{float(data[0]['depth_m']):g} m")
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Classifier EMA normalized temperature RMSE by depth · dotted = persistence\nLog axes · updates 1–25,000 · step zero retained in CSV"
    )
    fig.savefig(HERE / "validation-depths.png", dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    main()
