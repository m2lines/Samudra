#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Add held-out anomaly and paired-year diagnostics from completed pilot exports.

This reads predictions only and never selects or modifies a checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from samudra.experiments.observation_metrics import PROTOCOL, score, selection_score


def anomaly_statistics(prediction, reference, climatology, area):
    """Pooled space/origin correlation and RMS amplitude about training climatology."""
    valid = np.isfinite(reference) & (np.broadcast_to(area, reference.shape) > 0)
    if not valid.any():
        raise ValueError("No observed anomaly support")
    if (
        not np.isfinite(prediction[valid]).all()
        or not np.isfinite(climatology[valid]).all()
    ):
        raise ValueError("Missing candidate or climatology on fixed reference support")
    weights = np.broadcast_to(area, reference.shape)[valid].astype(np.float64)
    weights /= weights.sum()
    p = prediction[valid].astype(np.float64) - climatology[valid]
    r = reference[valid].astype(np.float64) - climatology[valid]
    pm, rm = np.sum(weights * p), np.sum(weights * r)
    pv, rv = np.sum(weights * (p - pm) ** 2), np.sum(weights * (r - rm) ** 2)
    reference_power = np.sum(weights * r**2)
    return {
        "correlation": float(np.sum(weights * (p - pm) * (r - rm)) / np.sqrt(pv * rv))
        if pv > 0 and rv > 0
        else None,
        "rms_amplitude_ratio": float(np.sqrt(np.sum(weights * p**2) / reference_power))
        if reference_power > 0
        else None,
        "anomaly_bias": float(pm - rm),
        "pointwise_rmse": float(np.sqrt(np.sum(weights * (p - r) ** 2))),
        "accepted_values": int(valid.sum()),
    }


def load(path):
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--selection-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.evaluation / "COMPLETE.json").exists():
        raise ValueError("Wait for complete held-out evaluation before reporting")
    signature = json.loads((args.evaluation / "evaluation-input.json").read_text())
    if signature["split"] != "test":
        raise ValueError("This report requires the held-out test cohort")
    frozen = json.loads(args.selection_reference.read_text())
    grid = load(args.grid)
    climatology = load(args.evaluation / "seasonal-climatology.npz")
    expected = [
        f"{year}-{month:02d}" for year in range(2015, 2023) for month in range(1, 13)
    ]
    if climatology["origins"].tolist() != expected:
        raise ValueError("Incomplete or reordered held-out origins")
    years = np.array([int(origin[:4]) for origin in expected])
    domain = grid["mask"][0] & (np.abs(grid["lat"][:, None]) <= 60)
    area = np.cos(np.deg2rad(grid["lat"]))[:, None] * domain
    output = {
        "scope": "Post-selection held-out diagnostics; yearly variation is not a confidence interval",
        "anomalies": "Subtract training seasonal climatology at each origin, lead and cell; pooled area-weighted space/origin correlation and RMS amplitude",
        "evaluation_input": signature,
        "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "grid_sha256": hashlib.sha256(args.grid.read_bytes()).hexdigest(),
        "selection_reference_sha256": hashlib.sha256(
            args.selection_reference.read_bytes()
        ).hexdigest(),
        "protocol": PROTOCOL,
        "methods": {},
        "files_sha256": {},
    }
    files = {
        "selected": "selected-predictions.npz",
        "source": "source-with-zero-forcing.npz",
        "inferred-persistence": "source-inferred-persistence.npz",
        "seasonal-climatology": "seasonal-climatology.npz",
        "inferred-anomaly-persistence": "inferred-anomaly-persistence.npz",
    }
    for method, filename in files.items():
        path = args.evaluation / filename
        with path.open("rb") as stream:
            output["files_sha256"][filename] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
        arrays = load(path)
        for key in ("origins", "reference", "reference_ohc"):
            if not np.array_equal(
                arrays[key], climatology[key], equal_nan=key != "origins"
            ):
                raise ValueError(f"Different reporting cohort/support: {method}/{key}")
        diagnostics = {}
        for lead in (0, 2, 5):
            for channel, field in enumerate(("sst", "adt")):
                diagnostics[f"{field}/day{5 * (lead + 1)}"] = anomaly_statistics(
                    arrays["prediction"][:, lead, channel],
                    arrays["reference"][:, lead, channel],
                    climatology["prediction"][:, lead, channel],
                    area,
                )
        for channel, field in enumerate(("ohc_0_700", "ohc_700_2000")):
            diagnostics[field] = anomaly_statistics(
                arrays["predicted_ohc"][:, channel],
                arrays["reference_ohc"][:, channel],
                climatology["predicted_ohc"][:, channel],
                area,
            )
        annual = {}
        for year in np.unique(years):
            subset = {
                key: arrays[key][years == year]
                for key in ("prediction", "reference", "predicted_ohc", "reference_ohc")
            }
            result = score(
                **subset, lat=grid["lat"], lon=grid["lon"], mask=grid["mask"][0]
            )
            annual[str(year)] = {
                "metrics": result["metrics"],
                "spectral_errors_dex": {
                    k: v["error_dex"] for k, v in result["spectra"].items()
                },
                "composite_fixed_validation_scale": selection_score(
                    result, frozen["control"], frozen["spectral_keys"]
                ),
            }
        output["methods"][method] = {
            "anomaly_diagnostics": diagnostics,
            "annual": annual,
        }
    paired = {}
    selected = output["methods"]["selected"]["annual"]
    for method, result in output["methods"].items():
        if method == "selected":
            continue
        deltas = {
            year: selected[year]["composite_fixed_validation_scale"]
            - result["annual"][year]["composite_fixed_validation_scale"]
            for year in selected
        }
        values = np.array(list(deltas.values()))
        paired[method] = {
            "selected_minus_control_by_year": deltas,
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "years_selected_better": int((values < 0).sum()),
            "note": "Eight paired calendar-year summaries; dependent years, not eight independent trials",
        }
    output["paired_year_composite_differences"] = paired
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
