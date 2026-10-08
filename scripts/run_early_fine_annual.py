#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit matched annual data or evaluate one frozen early/fine checkpoint."""

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from samudra.experiments.annual_artifacts import annual_output_hashes
from samudra.experiments.observation_annual import read_origin, verify_sample_root
from samudra.experiments.observation_pilot import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=["audit", "evaluate"], required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    config = json.loads((args.root / "paths.json").read_text())
    if os.environ["SAMUDRA_CODE_COMMIT"] != config["producer"]:
        raise ValueError("Evaluation producer differs")
    runtime = json.loads((args.root / "runtime-contract.json").read_text())
    for name, expected in runtime["modules"].items():
        module = importlib.import_module(name)
        if getattr(module, "__version__", None) != expected["version"]:
            raise ValueError("Runtime version differs: " + name)
        for filename, checksum in expected["binaries"].items():
            if digest(filename) != checksum:
                raise ValueError("Runtime binary differs: " + filename)
    contract = {
        "config_sha256": digest(args.root / "paths.json"),
        "runtime_sha256": digest(args.root / "runtime-contract.json"),
    }
    samples = Path(config["observations"])
    annual = Path(config["annual_data"])
    if args.stage == "audit":
        if digest(config["image"]) != runtime["image_sha256"]:
            raise ValueError("Container checksum differs")
        for row in config["models"]:
            run = Path(row["run"])
            manifest = json.loads((run / "manifest.json").read_text())
            verify_sample_root(samples, manifest)
            best = json.loads((run / "best.json").read_text())
            complete = json.loads((run / "TRAIN_COMPLETE.json").read_text())
            if {
                best["checkpoint_sha256"],
                complete["best_checkpoint_sha256"],
                digest(run / "best.pt"),
            } != {row["checkpoint_sha256"]}:
                raise ValueError("Selected checkpoint lineage differs: " + row["name"])
        if not (annual / "DATA_READY.json").is_file():
            raise ValueError("Missing annual transfer verification")
        origins = sorted((annual / "test").glob("*/COMPLETE.json"))
        if [p.parent.name for p in origins] != config["origins"]:
            raise ValueError("Annual origin cohort differs")
        for path in origins:
            description, raw = read_origin(path.parent)
            if not np.isfinite(raw["atmosphere"]).all():
                raise ValueError("Missing prescribed forcing")
            # Annual and monthly builders must agree before phase resets diverge.
            with np.load(
                samples / "test" / (description["origin"][:7] + ".npz")
            ) as monthly:
                for key in ("surface", "velocity", "atmosphere"):
                    np.testing.assert_array_equal(raw[key][:25], monthly[key][:25])
                np.testing.assert_array_equal(
                    raw["midpoint"][:25], monthly["midpoints"][:25]
                )
            for record in description["monthly_interiors"]:
                file = path.parent / record["file"]
                if file.name != record["file"] or digest(file) != record["sha256"]:
                    raise ValueError("Annual monthly target checksum differs")
                with (
                    np.load(file) as target,
                    np.load(samples / "test" / (file.name[:7] + ".npz")) as monthly,
                ):
                    np.testing.assert_array_equal(target["ohc"], monthly["ohc"])
            print(
                json.dumps(
                    {"event": "annual_data_audited", "origin": description["origin"]}
                ),
                flush=True,
            )
        atomic_json(
            {
                **contract,
                "origins": config["origins"],
                "checkpoints": config["models"],
                "monthly_first_30_days_and_ohc_exact": True,
                "finite_forcing": True,
            },
            args.root / "DATA_AUDITED.json",
        )
        return
    audit = json.loads((args.root / "DATA_AUDITED.json").read_text())
    if any(audit[k] != v for k, v in contract.items()):
        raise ValueError("Annual data audit contract differs")
    if args.index is None:
        parser.error("Evaluation needs a model index")
    row = config["models"][args.index]
    if digest(Path(row["run"]) / "best.pt") != row["checkpoint_sha256"]:
        raise ValueError("Selected model changed after audit")
    output = args.root / row["name"]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "samudra.experiments.observation_annual",
            "--run",
            row["run"],
            "--data",
            str(samples),
            "--annual-data",
            str(annual),
            "--output",
            str(output),
            "--split",
            "test",
            "--global-observations",
            "--include-persistence",
        ],
        check=True,
    )
    complete = json.loads((output / "COMPLETE.json").read_text())
    if (
        sorted(complete["origins"]) != config["origins"]
        or complete["inputs"]["checkpoint_sha256"] != row["checkpoint_sha256"]
    ):
        raise ValueError("Incomplete annual output")
    for item in complete["origins"].values():
        for filename in item.values():
            if not (output / filename).is_file():
                raise ValueError("Annual output missing: " + filename)
    atomic_json(
        {
            **contract,
            "model": row,
            "outputs": annual_output_hashes(output, complete),
        },
        output / "VERIFIED.json",
    )
    print(
        json.dumps({"event": "annual_model_verified", "model": row["name"]}), flush=True
    )


if __name__ == "__main__":
    main()
