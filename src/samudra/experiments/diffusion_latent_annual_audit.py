# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the original latent annual export to a retrospective source-manifest audit."""

import argparse
import json
from pathlib import Path

import numpy as np

from samudra.experiments.observation_pilot import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-checksums", type=Path, required=True)
    args = parser.parse_args()
    observed = {}
    for line in args.source_checksums.read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        if name in observed:
            raise ValueError("Duplicate source checksum")
        observed[name] = checksum
    protocol = json.loads((args.report / "protocol.json").read_text())
    complete = args.report / "annual/COMPLETE.json"
    completed = json.loads(complete.read_text())["inputs"]
    reference = json.loads((args.baseline / "annual/COMPLETE.json").read_text())[
        "inputs"
    ]
    if completed["checkpoint_sha256"] != protocol["checkpoint_sha256"]:
        raise ValueError("Annual checkpoint differs from monthly checkpoint")
    expected = {"DATA_READY.json": protocol["training_protocol"]["data_sha256"]}
    expected.update(
        {
            "data/annual_observations/" + k: v
            for k, v in reference["annual_manifests"].items()
        }
    )
    if (
        observed != expected
        or completed["annual_origins"] != reference["annual_origins"]
    ):
        raise ValueError("Source manifests differ from training audit or baseline")
    for origin in completed["annual_origins"]:
        with (
            np.load(args.report / "annual" / f"{origin}.npz") as current,
            np.load(args.baseline / "annual" / f"{origin}.npz") as baseline,
        ):
            for key in ("reference", "reference_ohc", "months", "leads_days"):
                np.testing.assert_array_equal(current[key], baseline[key])
    atomic_json(
        dict(
            annual_completion_sha256=digest(complete),
            checkpoint_sha256=protocol["checkpoint_sha256"],
            data_audit_sha256=observed["DATA_READY.json"],
            annual_manifests=reference["annual_manifests"],
            source_checksum_record_sha256=digest(args.source_checksums),
            evidence="Post-run source manifest hashes match baseline and the training-bound DATA_READY audit; all exported annual surface/OHC references, months and leads exactly equal baseline, including NaNs. Original evaluator verified each input payload on read.",
            limitation="Retrospective provenance supplement; original completion record is preserved unchanged.",
        ),
        args.report / "annual/source-manifests.json",
    )


if __name__ == "__main__":
    main()
