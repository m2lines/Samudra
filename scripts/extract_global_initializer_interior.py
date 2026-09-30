#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only export of saved learned T/S initialization and December climatology."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

DEPTHS = np.array(
    [
        2.5,
        10,
        22.5,
        40,
        65,
        105,
        165,
        250,
        375,
        550,
        775,
        1050,
        1400,
        1850,
        2400,
        3100,
        4000,
        5000,
        6000,
    ]
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-json", type=Path, required=True)
    parser.add_argument("--stages", nargs="+")
    parser.add_argument(
        "--data", type=Path, default=Path("/scratch/jr7309/data/obs-d-pilot")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise ValueError("Preserve exports; choose a new output directory")
    output.mkdir(parents=True)
    specifications = json.loads(args.sources_json.read_text())
    stages = args.stages or list(specifications)
    grid_path = args.data / "grid.npz"
    grid = dict(np.load(grid_path))
    np.savez_compressed(
        output / "grid.npz",
        lat=grid["lat"],
        lon=grid["lon"],
        mask=grid["mask"][38:76].reshape(2, 19, *grid["mask"].shape[-2:]),
        depths=DEPTHS,
    )
    provenance = {
        "source_sha256": digest(__file__),
        "sources": {
            str(args.sources_json): digest(args.sources_json),
            str(grid_path): digest(grid_path),
        },
        "stages": {},
    }
    for stage in stages:
        spec = specifications[stage]
        annual, monthly = Path(spec["annual"]), Path(spec["monthly"])
        complete = json.loads((annual / "COMPLETE.json").read_text())
        fixed = json.loads((monthly / "fixed-budget.json").read_text())
        lineage = fixed["fixed_budget_lineage"]
        assert complete["inputs"]["global_observations"]
        assert (
            complete["inputs"]["checkpoint_sha256"]
            == lineage["checkpoint_sha256"]
            == fixed["sha256"]
        )
        assert lineage["task_counts"] == {
            "om4": spec["om4_updates"],
            "observation": spec["observation_updates"],
        }
        provenance["sources"][str(annual / "COMPLETE.json")] = digest(
            annual / "COMPLETE.json"
        )
        provenance["sources"][str(monthly / "fixed-budget.json")] = digest(
            monthly / "fixed-budget.json"
        )
        provenance["stages"][stage] = lineage
        for origin in ["2015-01-01", "2018-01-01", "2021-01-01"]:
            source = annual / (origin + ".npz")
            initial = np.load(source)["initial"][-1]
            ts = initial[38:76].reshape(2, 19, *initial.shape[-2:])
            np.savez_compressed(output / (stage + "-" + origin + ".npz"), ts=ts)
            provenance["sources"][str(source)] = digest(source)
    total = np.zeros((2, 14, *grid["mask"].shape[-2:]))
    count = np.zeros_like(total)
    training = sorted((args.data / "train").glob("*-12.npz"))
    for path in training:
        assert "1993-05" <= path.stem <= "2013-07"
        values = np.load(path)["interior"]
        assert values.shape == total.shape
        finite = np.isfinite(values)
        total += np.where(finite, values, 0)
        count += finite
        provenance["sources"][str(path)] = digest(path)
    assert len(training) == 20
    climatology = np.divide(
        total, count, out=np.full_like(total, np.nan), where=count > 0
    )
    np.savez_compressed(
        output / "december-climatology.npz", ts=climatology, count=count
    )
    provenance["definition"] = (
        "Last initialized five-day state before each January origin; all 19 model T/S depths preserved. December climatology averages 20 training-only monthly IAP analyses, 1993-2012; no fallback or invented reference. No new model inference."
    )
    provenance["files"] = {
        p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in output.glob("*.npz")
    }
    (output / "COMPLETE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    archive = output.with_suffix(".tar")
    with tarfile.open(archive, "w") as tar:
        tar.add(output, arcname="interior")
    receipt = {"sha256": digest(archive), "bytes": archive.stat().st_size}
    (output.parent / (output.name + "-receipt.json")).write_text(
        json.dumps(receipt) + "\n"
    )
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
