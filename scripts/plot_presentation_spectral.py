#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot the original 27-term spectral score for the 17 presentation models."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from samudra.metrics.spectra import log10_rmse_between_curves
from scripts.export_presentation_annual_maps import GROUPS
from scripts.summarize_extent_results import controls_match


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot(
    repo,
    output,
    *,
    names=None,
    audits=None,
    checkpoints=None,
    title_prefix="",
    figsize=(10, 8),
):
    artifacts = repo / "docs/experiments/observation-d/artifacts"
    if names is None:
        names = list(dict.fromkeys(n for group in GROUPS.values() for n in group))
    audits = audits or [
        artifacts / "extent-review-2026-10-05/summary/summary-provenance.json",
        artifacts / "early-fine-2026-10-07/final/summary/summary-provenance.json",
    ]
    sources = {}
    arms: dict[str, Any] = {}
    selection = None
    for audit_path in audits:
        audit = json.loads(audit_path.read_text())
        for filename, checksum in audit["sources"].items():
            if filename in sources:
                continue
            path = repo / filename
            if digest(path) != checksum:
                raise ValueError("Source bundle hash differs: " + filename)
            sources[filename] = checksum
            bundle = json.loads(path.read_text())
            if selection is None:
                selection = bundle["selection_reference"]
            elif bundle["selection_reference"] != selection:
                raise ValueError("Validation metric definitions differ")
            for name, record in bundle["arms"].items():
                if name not in names:
                    continue
                if name in arms and arms[name] != record:
                    raise ValueError("Conflicting source records for " + name)
                arms[name] = record
    if set(arms) != set(names) or selection is None:
        raise ValueError("Incomplete presentation cohort")
    keys = selection["spectral_keys"]
    expected = {
        f"{v}/{r}/day{d}"
        for v in ("sst", "adt", "eke")
        for r in ("North Pacific", "Gulf Stream", "Agulhas")
        for d in (5, 15, 30)
    }
    if set(keys) != expected:
        raise ValueError("Expected the original 27 spectral terms")
    if checkpoints is None:
        annual_folder = artifacts / "presentation-annual-2026-10-08/final"
        annual_bytes = gzip.decompress((annual_folder / "results.json.gz").read_bytes())
        annual_audit = json.loads(
            (annual_folder / "COLLECTION_COMPLETE.json").read_text()
        )
        if (
            hashlib.sha256(annual_bytes).hexdigest()
            != annual_audit["files"]["results.json"]
        ):
            raise ValueError("Presentation checkpoint provenance changed")
        annual = json.loads(annual_bytes)
        checkpoints = {
            n: annual["models"][n]["model"]["checkpoint_sha256"] for n in names
        }
    control = arms[names[0]]["files"]["seasonal-climatology.json"]["data"]
    cohort = control["origins"]
    if len(cohort) != 96:
        raise ValueError("Expected 96 held-out monthly origins")
    rows = []

    def score(name, forecast, data):
        if data["origins"] != cohort or set(data["spectra"]) != expected:
            raise ValueError("Forecast cohort or spectral terms differ")
        for key in keys:
            curve = data["spectra"][key]
            for field in ("k_rad_km", "reference_power"):
                np.testing.assert_allclose(
                    curve[field], control["spectra"][key][field], rtol=1e-10, atol=0
                )
            value = curve["error_dex"]
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid spectral error")
            np.testing.assert_allclose(
                value,
                log10_rmse_between_curves(
                    curve["k_rad_km"],
                    curve["reference_power"],
                    curve["k_rad_km"],
                    curve["prediction_power"],
                ),
                rtol=1e-10,
                atol=1e-12,
            )
        components = {
            v + "_spectral_dex": float(
                np.mean(
                    [
                        data["spectra"][k]["error_dex"]
                        for k in keys
                        if k.startswith(v + "/")
                    ]
                )
            )
            for v in ("sst", "adt", "eke")
        }
        rows.append(
            dict(
                model=name,
                forecast=forecast,
                spectral_dex=float(np.mean(list(components.values()))),
                **components,
            )
        )

    for name in names:
        arm = arms[name]
        checkpoint = checkpoints[name]
        if arm["best"]["checkpoint_sha256"] != checkpoint:
            raise ValueError("Different selected checkpoint: " + name)
        if arm["evaluation_complete"]["selected_sha256"] != checkpoint:
            raise ValueError("Evaluation checkpoint mismatch")
        own_control = arm["files"]["seasonal-climatology.json"]["data"]
        if own_control["origins"] != cohort or not controls_match(
            own_control["spectra"], control["spectra"], 1e-10
        ):
            raise ValueError("Different climatology control")
        score(name, "evolved", arm["files"]["selected.json"]["data"]["metrics"])
        score(
            name,
            "initialized persistence",
            arm["files"]["selected-inferred-persistence.json"]["data"],
        )
    score("Training seasonal climatology", "climatology", control)
    frame = pd.DataFrame(rows)
    # Independently check agreement with both already published score tables.
    for audit_path in audits:
        old = pd.read_csv(audit_path.parent / "scores.csv")
        for record in frame[frame.forecast != "climatology"].to_dict("records"):
            matched = old[
                (old.arm == record["model"]) & (old.forecast == record["forecast"])
            ]
            if len(matched):
                np.testing.assert_allclose(
                    float(record["spectral_dex"]),
                    float(matched.iloc[0].spectral_dex),
                    rtol=1e-12,
                )
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "spectral-scores.csv", index=False)
    values = frame.set_index(["model", "forecast"]).spectral_dex
    prediction = [values.loc[n, "evolved"] for n in names]
    persistence = [values.loc[n, "initialized persistence"] for n in names]
    climatology = values.loc["Training seasonal climatology", "climatology"]
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(names))
    ax.hlines(y, prediction, persistence, color="0.7", lw=1)
    ax.scatter(prediction, y, color="C0", label="Evolved forecast", zorder=3)
    for row, value in enumerate(prediction):
        ax.annotate(
            f"{value:.3f}",
            (value, row),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="C0",
        )
    ax.scatter(persistence, y, marker="|", s=65, color="0.4", label="Own persistence")
    ax.axvline(
        climatology,
        color="black",
        ls=":",
        label=f"Training climatology ({climatology:.3f})",
    )
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Mean spectral error (dex; lower is better)")
    ax.set_title(
        title_prefix
        + "Original 5 / 15 / 30-day spectral score\nSST + ADT + EKE · 3 regions · 96 held-out monthly origins",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"spectral-comparison.{extension}", dpi=160)
    plt.close(fig)
    files = (
        "spectral-scores.csv",
        "spectral-comparison.png",
        "spectral-comparison.pdf",
    )
    provenance = dict(
        script_sha256=digest(Path(__file__)),
        sources=sources,
        selection_reference_sha256=json.loads(audits[0].read_text())[
            "selection_reference_sha256"
        ],
        selected_checkpoint_sha256={
            n: arms[n]["best"]["checkpoint_sha256"] for n in names
        },
        cohort=cohort,
        spectral_keys=keys,
        definition="Same spectral component used for validation, evaluated on the original 96 held-out monthly origins; no RMSE blend",
        aggregation="Average spatial power across origins at each fixed lead, then RMS log10-power mismatch per curve; arithmetic mean of 27 curves (3 quantities x 3 regions x 3 leads)",
        eke="SSH-derived geostrophic velocity anomalies across origins at each fixed lead; original monthly definition, not within-year EKE",
        control_relative_tolerance=1e-10,
        checks=f"Source bundle hashes; {len(names)} matching selected checkpoints; identical cohorts; common controls and reference spectra within float roundoff; all {(2 * len(names) + 1) * len(keys)} constituent errors reproduced from saved power curves; agreement with original published score tables",
        output_sha256={f: digest(output / f) for f in files},
    )
    (output / "spectral-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.repo, args.output)
