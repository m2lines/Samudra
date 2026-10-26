#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export four fixed X-ray cases concurrently, then verify and merge manifests."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from samudra.experiments.observation_pilot import atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    config = json.loads((root / "config.json").read_text())
    cases = config["cases"]
    if len(cases) != 4:
        raise ValueError("Expected four case workers")

    def run(item):
        gpu, row = item
        destination = root / ("worker-" + row["name"])
        with (root / (row["name"] + ".log")).open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("export_monthly_step_anatomy.py")),
                    "--config",
                    str(root / "config.json"),
                    "--case",
                    row["name"],
                    "--output",
                    str(destination),
                ],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        manifest = json.loads((destination / "COMPLETE.json").read_text())
        for name, checksum in manifest["files"].items():
            if digest(destination / name) != checksum:
                raise ValueError("Export checksum mismatch")
        return destination, manifest

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, enumerate(cases)))
    output = root / "output"
    output.mkdir()
    manifest = dict(results[0][1])
    manifest["cases"] = {}
    grid_sha = results[0][1]["files"]["grid.npz"]
    shutil.copy2(results[0][0] / "grid.npz", output / "grid.npz")
    for destination, child in results:
        if (
            child["files"]["grid.npz"] != grid_sha
            or child["shared_era5"] != manifest["shared_era5"]
        ):
            raise ValueError("Cases have differing grid or forcing provenance")
        manifest["cases"].update(child["cases"])
        for case in child["cases"]:
            shutil.move(str(destination / case), output / case)
    manifest["files"] = {
        str(path.relative_to(output)): digest(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    atomic_json(manifest, output / "COMPLETE.json")
    print(
        json.dumps(
            {
                "complete": True,
                "cases": list(manifest["cases"]),
                "files": len(manifest["files"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
