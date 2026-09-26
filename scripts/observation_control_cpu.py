#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Preview the seasonal observational control on CPU while GPUs are queued."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.observation_metrics import PROTOCOL, selection_score
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import Pilot, digest
from samudra.experiments.observation_training import Samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--selection-reference")
    args = parser.parse_args()
    evaluator = Pilot.__new__(Pilot)
    evaluator.data = Samples(args.data, "cpu")
    # The climatology branch only calls eval(); no network weights or forward pass.
    with torch.device("meta"):
        evaluator.model = ObservationTransfer(evaluator.data.grid["names"].tolist())
    paths = evaluator.data.paths(args.split)
    if len(paths) != (96 if args.split == "test" else 9):
        raise ValueError("Incomplete requested cohort")
    result = evaluator.evaluate(paths, climatology=True)
    keys = sorted(result["spectra"])
    if {key.split("/")[0] for key in keys} != {"sst", "adt", "eke"}:
        raise ValueError("Missing required spectral field")
    for name in PROTOCOL["integrated"]:
        value = result["metrics"][name]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid control denominator: {name}")
    record = {
        "completed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "scope": "CPU control diagnostic; no model checkpoint is trained or selected",
        "split": args.split,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "training-only seasonal climatology",
        "code_commit": os.environ.get("SAMUDRA_CODE_COMMIT"),
        "data_manifest_sha256": digest(Path(args.data) / "SHA256SUMS"),
        "protocol": PROTOCOL,
        "spectral_keys": keys,
        "control_self_score": selection_score(result, result, keys),
        "result": result,
    }
    if args.selection_reference:
        reference = json.loads(Path(args.selection_reference).read_text())
        if reference["data_manifest_sha256"] != record["data_manifest_sha256"]:
            raise ValueError(
                "Control and frozen selection reference use different data"
            )
        record["selection_reference_sha256"] = digest(args.selection_reference)
        record["composite_fixed_validation_scale"] = selection_score(
            result, reference["control"], reference["spectral_keys"]
        )
    output = Path(args.output)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "spectral_components": len(keys),
                "metrics": result["metrics"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
