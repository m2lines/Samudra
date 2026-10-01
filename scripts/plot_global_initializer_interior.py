#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Learned initialization: T/S profiles, context agreement, anomalies and sections."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from global_physical_heat_content import cell_areas  # type: ignore[import-not-found]
from plot_observation_missingness import panel_plot  # type: ignore[import-not-found]

LABELS = {
    "obs00050": "Early: 50 obs",
    "obs00500": "Developing: 500 obs",
    "obs02000": "Middle: 2,000 obs",
    "obs08000": "Final: 8,000 obs",
    "scratch08000": "Obs-only: 8,000 obs",
    "scratch16000": "Obs-only: 16,000 obs",
}
COLORS = {
    "obs00050": "#c94438",
    "obs00500": "#9c65c5",
    "obs02000": "#35a38f",
    "obs08000": "#2167ad",
    "scratch08000": "#c87526",
    "scratch16000": "#7b4caf",
}
REGIONS = [
    ("Global", (0, 360), (-90, 90)),
    ("North Atlantic", (280, 360), (0, 65)),
    ("North Pacific", (120, 260), (0, 65)),
    ("Southern Ocean", (0, 360), (-60, -30)),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reduce_depth(values, support, area):
    weights = np.where(support, area, 0)
    if not (weights.sum((-2, -1)) > 0).all():
        raise ValueError("Empty depth/region support")
    if not np.isfinite(values[support]).all():
        raise ValueError("Nonfinite initialized state on reference support")
    return (np.where(support, values, 0) * weights).sum((-2, -1)) / weights.sum(
        (-2, -1)
    )


def depth_axis(ax):
    ax.set_yscale("symlog", linthresh=150)
    ax.set_ylim(2000, 0)
    ax.set_yticks([0, 100, 500, 1000, 1850], ["0", "100", "500", "1000", "1850"])
    ax.grid(alpha=0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--scratch-arrays", type=Path)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoints-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    roots = [args.arrays] + ([args.scratch_arrays] if args.scratch_arrays else [])
    sources = {}
    grids = [dict(np.load(root / "grid.npz")) for root in roots]
    grid = grids[0]
    for other in grids[1:]:
        for key in grid:
            np.testing.assert_array_equal(grid[key], other[key])
    lat, lon, depths = grid["lat"], grid["lon"], grid["depths"][:14]
    area = cell_areas(lat, lon)
    wet = grid["mask"][:, :14]
    climate = np.load(args.arrays / "december-climatology.npz")["ts"]
    for root in roots:
        receipt = json.loads((root / "COMPLETE.json").read_text())
        for name, info in receipt["files"].items():
            assert sha(root / name) == info["sha256"], name
            sources[str(root / name)] = info["sha256"]
        if root != args.arrays:
            np.testing.assert_array_equal(
                np.load(root / "december-climatology.npz")["ts"], climate
            )
    stage_roots = {
        stage: root
        for root in roots
        for stage in json.loads((root / "COMPLETE.json").read_text())["stages"]
    }
    stages = [k for k in LABELS if k in stage_roots]
    if args.endpoints_only:
        stages = [
            key for key in stages if key == "obs08000" or key.startswith("scratch")
        ]
    result: dict = {
        "sources": sources,
        "origins": {},
        "definition": "Last initialized pre-January five-day state compared descriptively with preceding December IAP monthly analysis. No instantaneous truth claim, forecast rollout, or surface infilling score. T at 2.5 m is copied and excluded from profile agreement summaries; all salinity and deeper temperature are learned. All curves share finite IAP/climatology/model wet support at each depth. December climatology uses 20 training-only December analyses, 1993-2012. No inputs are replaced with climatology.",
    }
    for origin, month in [
        ("2015-01-01", "2014-12"),
        ("2018-01-01", "2017-12"),
        ("2021-01-01", "2020-12"),
    ]:
        reference_path = args.references / (month + ".npz")
        reference_data = dict(np.load(reference_path))
        np.testing.assert_array_equal(reference_data["depth"], depths)
        reference = reference_data["values"]
        result["sources"][str(reference_path)] = sha(reference_path)
        prediction = {
            stage: np.load(stage_roots[stage] / (stage + "-" + origin + ".npz"))["ts"][
                :, :14
            ]
            for stage in stages
        }
        support = wet & np.isfinite(reference) & np.isfinite(climate)
        for values in prediction.values():
            if not np.isfinite(values[support]).all():
                raise ValueError("Nonfinite prediction on common reference support")
        record: dict = {
            "monthly_context": month,
            "profiles": {},
            "context_agreement": {},
            "maps": {},
        }
        for name, xb, yb in REGIONS:
            regional = (
                (lon[None, :] >= xb[0])
                & (lon[None, :] < xb[1])
                & (lat[:, None] >= yb[0])
                & (lat[:, None] <= yb[1])
            )
            common = support & regional
            methods = {
                **prediction,
                "IAP monthly context": reference,
                "Training December climatology": climate,
            }
            profiles = {
                method: reduce_depth(values, common, area).tolist()
                for method, values in methods.items()
            }
            record["profiles"][name] = {
                "bounds": {"longitude": xb, "latitude": yb},
                "means": profiles,
                "cells_by_depth": common.sum((-2, -1)).tolist(),
            }
        for method, values in {
            **prediction,
            "Training December climatology": climate,
        }.items():
            rmse = np.sqrt(reduce_depth((values - reference) ** 2, support, area))
            anomaly_rms = np.sqrt(reduce_depth((values - climate) ** 2, support, area))
            record["context_agreement"][method] = {
                "temperature_rmse_by_depth_C": rmse[0].tolist(),
                "salinity_rmse_by_depth_psu": rmse[1].tolist(),
                "mean_temperature_rmse_10_1850_C": float(rmse[0, 1:].mean()),
                "mean_salinity_rmse_2_5_1850_psu": float(rmse[1].mean()),
                "anomaly_rms_by_depth": anomaly_rms.tolist(),
            }
        record["reference_anomaly_rms_by_depth"] = np.sqrt(
            reduce_depth((reference - climate) ** 2, support, area)
        ).tolist()
        fig, axes = plt.subplots(2, 4, figsize=(15, 8), layout="constrained")
        for column, (region, xb, yb) in enumerate(REGIONS):
            for variable in [0, 1]:
                ax = axes[variable, column]
                start = 1 if variable == 0 else 0
                for method, profile in record["profiles"][region]["means"].items():
                    ax.plot(
                        np.array(profile)[variable, start:],
                        depths[start:],
                        label=LABELS.get(method, method),
                        color=COLORS.get(
                            method,
                            "#222222" if method == "IAP monthly context" else "#e29624",
                        ),
                        ls="--"
                        if method == "Training December climatology"
                        or method.startswith("scratch")
                        else "-",
                    )
                depth_axis(ax)
                ax.set_title(
                    region + f" ({xb[0]}–{xb[1]}°E, {yb[0]}–{yb[1]}°N)", fontsize=9
                )
                ax.set_xlabel(
                    "Temperature (°C); copied 2.5 m omitted"
                    if variable == 0
                    else "Practical salinity (psu)"
                )
                if column == 0:
                    ax.set_ylabel("Depth (m)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
        fig.suptitle(
            f"Learned initialization before {origin}; {month} IAP monthly context"
        )
        fig.savefig(args.output / (origin + "-initializer-profiles.png"), dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
        for variable, metric in [
            (0, "temperature_rmse_by_depth_C"),
            (1, "salinity_rmse_by_depth_psu"),
        ]:
            start = 1 if variable == 0 else 0
            for method, statistics in record["context_agreement"].items():
                axes[variable].plot(
                    np.array(statistics[metric])[start:],
                    depths[start:],
                    label=LABELS.get(method, method),
                    color=COLORS.get(method, "#e29624"),
                    ls="--"
                    if method == "Training December climatology"
                    or method.startswith("scratch")
                    else "-",
                )
            depth_axis(axes[variable])
            axes[variable].set_xlabel(
                "Temperature context RMSE (°C), 10–1850 m"
                if variable == 0
                else "Salinity context RMSE (psu), 2.5–1850 m"
            )
        axes[0].set_ylabel("Depth (m)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
        fig.suptitle(
            f"Initialization / {month} monthly-context agreement; not instantaneous skill"
        )
        fig.savefig(args.output / (origin + "-initializer-context-errors.png"), dpi=140)
        plt.close(fig)
        for variable, depth, field_name, limits, anomaly_limits in [
            (0, 9, "temperature550", (-2, 25), (-3, 3)),
            (1, 0, "salinity0", (28, 39), (-1.5, 1.5)),
        ]:
            local_grid = {"lat": lat, "lon": lon, "mask": wet[variable, depth][None]}
            methods = {
                **prediction,
                "IAP monthly context": reference,
                "Training December climatology": climate,
            }
            for phase in ["absolute", "anomaly"]:
                panels = []
                for method, values in methods.items():
                    if phase == "anomaly" and method == "Training December climatology":
                        continue
                    values = values[variable, depth]
                    if phase == "anomaly":
                        values = values - climate[variable, depth]
                    panels.append(
                        (LABELS.get(method, method), values[None], local_grid)
                    )
                unit = "°C" if variable == 0 else "psu"
                field = (
                    0,
                    field_name,
                    unit,
                    limits if phase == "absolute" else anomaly_limits,
                    "turbo" if phase == "absolute" else "RdBu_r",
                )
                filename = origin + "-initializer-" + field_name + "-" + phase + ".png"
                record["maps"][filename] = panel_plot(
                    panels,
                    f"Learned {field_name}, {origin}, {phase}; gray = model mask, white = missing reference/baseline",
                    field,
                    args.output / filename,
                    distinguish_missing=True,
                )
        section_stages = [
            s
            for s in [
                "obs00050",
                "obs02000",
                "obs08000",
                "scratch08000",
                "scratch16000",
            ]
            if s in prediction
        ]
        methods = {stage: prediction[stage] for stage in section_stages}
        methods["IAP monthly context"] = reference
        fig, axes = plt.subplots(
            2,
            len(methods),
            figsize=(3.5 * len(methods), 7),
            layout="constrained",
            squeeze=False,
        )
        for column, (method, values) in enumerate(methods.items()):
            anomaly = values - climate
            weight = np.where(support, area, 0)
            section = np.divide(
                (np.where(support, anomaly, 0) * weight).sum(-1),
                weight.sum(-1),
                out=np.full(anomaly.shape[:-1], np.nan),
                where=weight.sum(-1) > 0,
            )
            for variable in [0, 1]:
                ax = axes[variable, column]
                start = 1 if variable == 0 else 0
                limit = 1.5 if variable == 0 else 0.5
                im = ax.pcolormesh(
                    lat,
                    depths[start:],
                    section[variable, start:],
                    vmin=-limit,
                    vmax=limit,
                    cmap="RdBu_r",
                    shading="auto",
                )
                depth_axis(ax)
                ax.set_xlabel("Latitude (°N)")
                ax.set_title(LABELS.get(method, method), fontsize=9)
                if column == 0:
                    ax.set_ylabel("T depth (m)" if variable == 0 else "S depth (m)")
                if column == len(methods) - 1:
                    fig.colorbar(
                        im,
                        ax=list(axes[variable]),
                        label="Temperature anomaly (°C)"
                        if variable == 0
                        else "Salinity anomaly (psu)",
                        shrink=0.8,
                    )
        fig.suptitle(
            f"Initialized zonal anomalies from training December climatology; {origin}"
        )
        fig.savefig(
            args.output / (origin + "-initializer-zonal-anomalies.png"), dpi=140
        )
        plt.close(fig)
        result["origins"][origin] = record
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print("INITIALIZER_INTERIOR_RENDER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
