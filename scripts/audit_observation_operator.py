#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic only: score contemporaneous observed ADT through the model operator."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from samudra.experiments.observation_metrics import PROTOCOL, score


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climatology-export", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(args.climatology_export) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(args.grid) as archive:
        grid = {key: archive[key] for key in archive.files}
    expected = [f"{y}-{m:02d}" for y in range(2015, 2023) for m in range(1, 13)]
    if arrays["origins"].tolist() != expected:
        raise ValueError("Expected the complete held-out cohort")
    surface = arrays["reference"][:, :, :2]
    # Actual target-time observations, not persisted initial observations. Fill
    # only unavailable targets to keep the operator's surrounding model domain.
    prediction = np.where(np.isfinite(surface), surface, arrays["prediction"])
    result = score(
        prediction=prediction,
        reference=arrays["reference"],
        predicted_ohc=arrays["reference_ohc"],
        reference_ohc=arrays["reference_ohc"],
        lat=grid["lat"],
        lon=grid["lon"],
        mask=grid["mask"][0],
    )
    if len(result["spectra"]) != 27:
        raise ValueError("Missing spectral components")
    for name in ("sst_rmse", "ohc_0_700_rmse", "ohc_700_2000_rmse"):
        if result["metrics"][name] != 0:
            raise ValueError("Observation self-comparison must have zero direct error")
    for key, curve in result["spectra"].items():
        if not key.startswith("eke/") and abs(curve["error_dex"]) > 1e-12:
            raise ValueError("Direct observed SST/ADT spectra must agree")
    output = {
        "completed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "scope": "Data-only operator diagnostic, not a forecast, candidate or error floor",
        "method": "Contemporaneous target SST/ADT on the common coarse grid, training climatology only at missing cells, then unchanged geostrophic and EKE scoring operators against directly coarsened DUACS velocity",
        "origins": expected,
        "climatology_export_sha256": digest(args.climatology_export),
        "grid_sha256": digest(args.grid),
        "script_sha256": digest(Path(__file__)),
        "protocol": PROTOCOL,
        "result": result,
    }
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
