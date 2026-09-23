#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plot recorded validation components and spectra without running a model."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    arms = snapshot["arms"]
    reference = arms["fitting"]["selection-reference"]
    if reference is None:
        raise ValueError("No qualified validation reference has been recorded")
    control = reference["control"]
    candidates = {"Seasonal climatology": control}
    baseline = arms["fitting"]["baseline-validation"]
    if baseline:
        candidates["Pretrained / zero adapter"] = baseline
    for arm, label in (
        ("primary", "Full fine-tuning"),
        ("adapter-only", "Adapter only"),
        ("scratch", "Observation-only scratch"),
    ):
        best = arms[arm]["best"]
        if best:
            status = "selected" if arms[arm]["TRAIN_COMPLETE"] else "best so far"
            candidates[f"{label} ({status}; {best['phase']}:{best['step']})"] = best[
                "metrics"
            ]
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
    ax.set_ylabel("Error / fixed validation climatology error (lower is better)")
    ax.set_title("Validation integrated components — not held-out performance")
    ax.legend(fontsize=8)
    fig.savefig(args.output / "validation-components.png", dpi=180)
    fig.savefig(args.output / "validation-components.pdf")
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
    fig.suptitle("Validation day-30 spatial spectra · broad scales only")
    fig.savefig(args.output / "validation-spectra.png", dpi=180)
    fig.savefig(args.output / "validation-spectra.pdf")
    plt.close(fig)
    provenance = {
        "snapshot": str(args.snapshot),
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "snapshot_time": snapshot["snapshot_finished_utc"],
        "scope": "Recorded validation metrics; no model execution or held-out claim",
        "spectral_ordinate": "Existing kernel k times azimuthally averaged 2D power; spacing in metres and k in cycles/metre. Only the abscissa is converted to wavelength in km.",
        "candidates": list(candidates),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
