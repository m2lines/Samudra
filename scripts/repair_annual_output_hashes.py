#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Repair the known empty-output-hash audit without changing saved forecasts."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from samudra.experiments.annual_artifacts import annual_output_hashes
from samudra.experiments.observation_pilot import atomic_json, digest


def repair(root):
    config = json.loads((root / "paths.json").read_text())
    legacy = "5f61785a0aea8a55e7c4313518efe3d9d77e2a7f"  # pragma: allowlist secret
    if config["producer"] != legacy:
        raise ValueError("This repair applies only to the known original verifier")
    audit = json.loads((root / "DATA_AUDITED.json").read_text())
    if audit["config_sha256"] != digest(root / "paths.json"):
        raise ValueError("Original data audit differs")
    for model in config["models"]:
        folder = root / model["name"]
        verified = json.loads((folder / "VERIFIED.json").read_text())
        if (
            verified["outputs"]
            or verified["model"] != model
            or verified["config_sha256"] != digest(root / "paths.json")
        ):
            raise ValueError("Not the known empty-hash audit")
        complete = json.loads((folder / "COMPLETE.json").read_text())
        if sorted(complete["origins"]) != config["origins"]:
            raise ValueError("Annual cohort differs")
        if (
            complete["inputs"]["checkpoint_sha256"] != model["checkpoint_sha256"]
            or digest(Path(model["run"]) / "best.pt") != model["checkpoint_sha256"]
        ):
            raise ValueError("Selected checkpoint differs")
        if complete["inputs"] != json.loads((folder / "input.json").read_text()):
            raise ValueError("Evaluation input signature differs")
        hashes = annual_output_hashes(folder, complete)
        for origin, item in complete["origins"].items():
            for kind in ("metrics", "persistence"):
                metrics = json.loads((folder / item[kind]).read_text())
                for day in ("5", "15", "30", "90", "180", "365"):
                    if any(not np.isfinite(v) for v in metrics["leads"][day].values()):
                        raise ValueError("Nonfinite lead metric")
            with np.load(folder / item["arrays"]) as arrays:
                if arrays["surface"].shape[0:2] != (73, 2) or arrays["reference"].shape[
                    0:2
                ] != (73, 4):
                    raise ValueError("Incomplete saved forecast")
                np.testing.assert_array_equal(
                    arrays["months"], [f"{origin[:4]}-{m:02d}" for m in range(1, 13)]
                )
                for key in (
                    "surface",
                    "initial",
                    "state_at_leads",
                    "persistence_surface",
                ):
                    if not np.isfinite(arrays[key]).all():
                        raise ValueError("Nonfinite saved state: " + key)
        record = dict(
            producer=os.environ["SAMUDRA_CODE_COMMIT"],
            original_verified_sha256=digest(folder / "VERIFIED.json"),
            outputs=hashes,
            scope="Post-hoc read-back audit repairing an empty file-hash list; forecasts, weights and original audit remain unchanged",
        )
        destination = folder / "VERIFIED-OUTPUT-HASHES.json"
        if destination.exists():
            if json.loads(destination.read_text()) != record:
                raise ValueError("Conflicting output hash repair")
        else:
            atomic_json(record, destination)
        print(
            json.dumps(
                dict(
                    event="annual_output_hashes_repaired",
                    model=model["name"],
                    files=len(hashes),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    repair(parser.parse_args().root)
