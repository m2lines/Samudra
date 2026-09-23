#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plot recorded validation or held-out metrics without running a model."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARM_LABELS = (
    ("primary", "Full fine-tuning"),
    ("adapter-only", "Adapter only"),
    ("scratch", "Observation-only scratch"),
)


def candidates_for_split(snapshot, split):
    arms = snapshot["arms"]
    reference = arms["fitting"]["selection-reference"]
    if reference is None:
        raise ValueError("No qualified validation reference has been recorded")
    if split == "validation":
        control = reference["control"]
        candidates = {"Seasonal climatology": control}
        baseline = arms["fitting"]["baseline-validation"]
        if baseline:
            candidates["Pretrained / zero adapter"] = baseline
        for arm, label in ARM_LABELS:
            best = arms[arm]["best"]
            if best:
                status = "selected" if arms[arm]["TRAIN_COMPLETE"] else "best so far"
                candidates[f"{label} ({status}; {best['phase']}:{best['step']})"] = (
                    best["metrics"]
                )
        return reference, control, candidates
    if split != "test":
        raise ValueError("Unknown split")
    completed = []
    for arm, label in ARM_LABELS:
        directory = (
            "adapter-evaluation" if arm == "adapter-only" else arm + "-evaluation"
        )
        evaluation = snapshot["evaluations"].get(directory, {})
        if not evaluation.get("COMPLETE"):
            continue
        best = arms[arm]["best"]
        selected = evaluation["selected"]
        if (
            not arms[arm]["TRAIN_COMPLETE"]
            or selected["sha256"] != best["checkpoint_sha256"]
            or selected["split"] != "test"
            or evaluation["COMPLETE"]["origins"] != 96
        ):
            raise ValueError(
                "Held-out result does not match a completed selected checkpoint"
            )
        completed.append((label, evaluation))
    if not completed:
        raise ValueError("No complete held-out evaluation; do not relabel validation")
    controls = completed[0][1]
    control = controls["seasonal-climatology"]
    candidates = {
        "Seasonal climatology": control,
        "Pretrained / zero adapter": controls["source-with-zero-forcing"],
        "Source-state persistence": controls["source-inferred-persistence"],
        "Source-state anomaly persistence": controls["inferred-anomaly-persistence"],
    }
    for label, evaluation in completed:
        candidates[label + " (selected)"] = evaluation["selected"]["metrics"]
    expected = [
        f"{year}-{month:02d}" for year in range(2015, 2023) for month in range(1, 13)
    ]
    for label, result in candidates.items():
        if result["origins"] != expected:
            raise ValueError(f"Incomplete held-out cohort: {label}")
        for key in reference["spectral_keys"]:
            curve, target = result["spectra"][key], control["spectra"][key]
            for field in ("k_rad_km", "reference_power"):
                if not np.allclose(curve[field], target[field], rtol=1e-8, atol=0):
                    raise ValueError(
                        f"Different held-out spectral reference: {label}/{key}"
                    )
    return reference, control, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--grid", type=Path, help="Verified pilot grid for depth-resolved T/S plots"
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    reference, control, candidates = candidates_for_split(snapshot, args.split)
    stem = "validation" if args.split == "validation" else "heldout"
    caption = "Validation" if args.split == "validation" else "Held-out 2015–2022"
    args.output.mkdir(parents=True, exist_ok=True)
    names = reference["protocol"]["integrated"]
    labels = ["SST", "Geostrophic velocity", "EKE", "OHC 0–700 m", "OHC 700–2000 m"]
    colors = dict(
        zip(candidates, plt.get_cmap("tab10")(np.arange(len(candidates))), strict=False)
    )
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    width = 0.8 / len(candidates)
    for index, (label, result) in enumerate(candidates.items()):
        ratios = [result["metrics"][k] / control["metrics"][k] for k in names]
        positions = np.arange(len(names)) + (index - (len(candidates) - 1) / 2) * width
        ax.bar(positions, ratios, width, label=label, color=colors[label])
    ax.axhline(1, color="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(names)), labels)
    ax.set_ylabel(f"Error / {caption.lower()} climatology error (lower is better)")
    ax.set_title(
        f"{caption} integrated components"
        + (" — not held-out performance" if args.split == "validation" else "")
    )
    ax.legend(fontsize=8)
    fig.savefig(args.output / f"{stem}-components.png", dpi=180)
    fig.savefig(args.output / f"{stem}-components.pdf")
    plt.close(fig)

    leads = reference["protocol"]["leads_days"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
    for ax, metric, title, unit in zip(
        axes,
        ("sst_rmse", "velocity_rmse", "eke_rmse"),
        ("SST", "Geostrophic velocity", "EKE"),
        ("°C", "m s⁻¹", "m² s⁻²"),
        strict=True,
    ):
        for label, result in candidates.items():
            ax.plot(
                leads,
                [result["metrics"][f"{metric}/day{lead}"] for lead in leads],
                "o-",
                label=label,
                color=colors[label],
            )
        ax.set_title(title)
        ax.set_xlabel("Forecast bin end (day)")
        ax.set_ylabel(f"RMSE ({unit})")
        ax.set_xticks(leads)
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=8)
    fig.suptitle(f"{caption} surface errors · matched five-day means")
    fig.savefig(args.output / f"{stem}-leads.png", dpi=180)
    fig.savefig(args.output / f"{stem}-leads.pdf")
    plt.close(fig)

    fields = ["sst", "adt", "eke"]
    regions = reference["protocol"]["regions"]
    fig, axes = plt.subplots(3, len(regions), figsize=(12, 10), layout="constrained")
    for row, field in enumerate(fields):
        for column, region in enumerate(regions):
            ax = axes[row, column]
            key = f"{field}/{region}/day30"
            if key not in reference["spectral_keys"]:
                ax.text(0.5, 0.5, "Not qualified", ha="center", transform=ax.transAxes)
                continue
            curve = control["spectra"][key]
            wavelength = 2 * np.pi / np.asarray(curve["k_rad_km"])
            ax.loglog(
                wavelength, curve["reference_power"], "k--o", label="Observations"
            )
            for label, result in candidates.items():
                curve = result["spectra"][key]
                ax.loglog(
                    2 * np.pi / np.asarray(curve["k_rad_km"]),
                    curve["prediction_power"],
                    "o-",
                    color=colors[label],
                    label=label,
                )
            ax.set_title(f"{field.upper()} · {region}")
            ax.set_xlabel("Wavelength (km)")
            units = {"sst": "°C² m", "adt": "m³", "eke": "m⁵ s⁻⁴"}
            ax.set_ylabel(f"Radial spectrum ({units[field]})")
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=8)
    fig.suptitle(f"{caption} day-30 spatial spectra · broad scales only")
    fig.savefig(args.output / f"{stem}-spectra.png", dpi=180)
    fig.savefig(args.output / f"{stem}-spectra.pdf")
    plt.close(fig)
    grid_sha256 = None
    if args.grid:
        grid_sha256 = hashlib.sha256(args.grid.read_bytes()).hexdigest()
        expected = snapshot["arms"]["fitting"]["manifest"]["grid_sha256"]
        if grid_sha256 != expected:
            raise ValueError("Depth coordinate grid differs from the frozen pilot grid")
        with np.load(args.grid) as grid:
            depths = grid["depth"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
        for row, (variable, title, unit) in enumerate(
            (
                ("thetao", "Temperature", "°C"),
                ("so", "Practical salinity", "dimensionless"),
            )
        ):
            for column, statistic in enumerate(("rmse", "bias")):
                ax = axes[row, column]
                for label, result in candidates.items():
                    ts = result["thermohaline"]
                    indices = [
                        i
                        for i, name in enumerate(ts["channels"])
                        if name.startswith(variable + "_")
                    ]
                    z = [
                        depths[int(ts["channels"][i].rsplit("_", 1)[1])]
                        for i in indices
                    ]
                    ax.plot(
                        [ts[statistic][i] for i in indices],
                        z,
                        "o-",
                        label=label,
                        color=colors[label],
                        markersize=3,
                    )
                if statistic == "bias":
                    ax.axvline(0, color="black", linewidth=0.6)
                ax.set_yscale("log")
                ax.invert_yaxis()
                ax.set_ylabel("Depth (m, logarithmic)")
                ax.set_xlabel(f"{statistic.upper()} ({unit})")
                ax.set_title(title)
                ax.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=8)
        fig.suptitle(f"{caption} monthly forecast T/S against IAP")
        fig.savefig(args.output / f"{stem}-thermohaline.png", dpi=180)
        fig.savefig(args.output / f"{stem}-thermohaline.pdf")
        plt.close(fig)
    provenance = {
        "snapshot": str(args.snapshot),
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "snapshot_time": snapshot["snapshot_finished_utc"],
        "scope": f"Recorded {caption.lower()} metrics; no model execution",
        "split": args.split,
        "grid_sha256": grid_sha256,
        "bar_normalization": f"{caption} seasonal climatology errors; descriptive plot, not a new selection score",
        "spectral_ordinate": "Existing kernel k times azimuthally averaged 2D power; spacing in metres and k in cycles/metre. Only the abscissa is converted to wavelength in km.",
        "candidates": list(candidates),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
