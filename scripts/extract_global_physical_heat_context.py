#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Preserve initial column heat content and training-only OHC climatology."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_ohc(temp, mask):
    edges = np.array(
        [
            0,
            5,
            15,
            30,
            50,
            80,
            130,
            200,
            300,
            450,
            650,
            900,
            1200,
            1600,
            2100,
            2700,
            3500,
            4500,
            5500,
            6750,
        ]
    )
    result = []
    for low, high in [(0, 700), (700, 2000)]:
        dz = np.maximum(0, np.minimum(edges[1:], high) - np.maximum(edges[:-1], low))
        active = dz > 0
        valid = mask[active].all(0)
        result.append(
            np.where(
                valid,
                np.sum(temp[active] * dz[active, None, None], 0) * 1035 * 3850,
                np.nan,
            )
        )
    return np.array(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    metadata = args.compact / "COMPLETE.json"
    meta = json.loads(metadata.read_text())
    grid_path = args.data / "grid.npz"
    assert digest(grid_path) == meta["sources"][str(grid_path)]
    grid = dict(np.load(grid_path))
    values = {}
    sources = {str(metadata): digest(metadata), str(grid_path): digest(grid_path)}
    for label, checkpoint in meta["checkpoints"].items():
        for origin in checkpoint["annual"]:
            candidates = [
                Path(path)
                for path in meta["sources"]
                if Path(path).name == origin + ".npz"
                and (
                    Path(path).parent.name == label + "-annual-retry1"
                    or (
                        label == "obs08000"
                        and Path(path).parent.name == "global-endpoint-annual"
                    )
                )
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Ambiguous annual source for {label}/{origin}: {candidates}"
                )
            source = candidates[0]
            actual_hash = digest(source)
            if actual_hash != meta["sources"][str(source)]:
                raise ValueError(f"Annual source changed: {source}")
            initial = np.load(source)["initial"][-1]
            values[label + "-" + origin] = initial_ohc(
                initial[38:57], grid["mask"][38:57]
            )
            sources[str(source)] = actual_hash
    total = np.zeros((12, 2, *grid["mask"].shape[-2:]))
    count = np.zeros_like(total)
    training = sorted((args.data / "train").glob("*.npz"))
    for path in training:
        ohc = np.load(path)["ohc"]
        month = int(path.stem[5:7]) - 1
        finite = np.isfinite(ohc)
        total[month] += np.where(finite, ohc, 0)
        count[month] += finite
        sources[str(path)] = digest(path)
    values["climatology"] = np.divide(
        total, count, out=np.full_like(total, np.nan), where=count > 0
    )
    output = args.output / "heat-context.npz"
    np.savez_compressed(output, **values)
    receipt = {
        "source_sha256": digest(__file__),
        "sources": sources,
        "output_sha256": digest(output),
        "bytes": output.stat().st_size,
        "training_months": len(training),
        "definition": "Native-layer training-only IAP monthly OHC climatology, and initialized states integrated over OM4 layers; J/m2. No new model inference.",
    }
    (args.output / "HEAT_CONTEXT_COMPLETE.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()
