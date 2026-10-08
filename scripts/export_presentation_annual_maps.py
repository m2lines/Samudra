#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export matched map bundles from the hash-verified presentation comparison."""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from samudra.experiments.annual_artifacts import annual_output_hashes
from samudra.experiments.observation_pilot import atomic_json, digest

GROUPS = {
    "patch": ["U-global", "U-multitask", "U-omit-patch", "U-patch-loss01"],
    "accessory": [
        "U-global",
        "U-aux01",
        "U-aux01-static",
        "U-aux01-seasonal",
        "U-aux01-shuffled",
        "U-aux01-anomaly",
    ],
    "capacity": ["U-global", "W-global", "W-multitask", "A-global", "A-multitask"],
    "early-fine": [
        "U-global",
        "U-multitask-early",
        "U-multitask-early-latent",
        "U-multitask-early-fine",
        "U-multitask-early-fine-latent",
    ],
}


def export(root, results_path, output):
    results = json.loads(results_path.read_text())
    config = results["configuration"]
    origins = config["origins"]
    if set(results["models"]) != {n for names in GROUPS.values() for n in names}:
        raise ValueError("Map models differ from the collected comparison")
    samples = Path(config["observations"])
    with np.load(samples / "grid.npz") as grid:
        base = {k: grid[k] for k in ("lat", "lon")}
        base["mask"] = grid["mask"][0]
    with np.load(samples / "statistics.npz") as stats:
        seasonal = stats["surface_climatology"]
    fields: dict[tuple[str, int], list[np.ndarray]] = {}
    references: dict[tuple[str, int], np.ndarray] = {}
    sources = {}
    for name, metadata in results["models"].items():
        evaluation_root = (
            Path(config["reuse_annual_root"])
            if name in config["reuse_models"]
            else root
        )
        run = evaluation_root / name
        complete = json.loads((run / "COMPLETE.json").read_text())
        verified = metadata["verified"]
        hashes = verified["outputs"] or metadata["repaired_output_audit"]["outputs"]
        if annual_output_hashes(run, complete) != hashes:
            raise ValueError("Forecast changed after collection: " + name)
        sources[name] = dict(
            method=name,
            path=str(run),
            selected_checkpoint_sha256=metadata["model"]["checkpoint_sha256"],
            source_checkpoint_sha256=None,
            annual_output_hashes=hashes,
        )
        for day in (30, 365):
            fields[name, day] = []
            fields[name + " / initialized persistence", day] = []
        for origin in origins:
            item = complete["origins"][origin]
            with np.load(run / item["arrays"]) as arrays:
                for day in (30, 365):
                    lead = day // 5 - 1
                    truth = arrays["reference"][lead, :2]
                    if (origin, day) in references:
                        np.testing.assert_array_equal(references[origin, day], truth)
                    else:
                        references[origin, day] = truth
                    fields[name, day].append(arrays["surface"][lead])
                    fields[name + " / initialized persistence", day].append(
                        arrays["persistence_surface"][lead]
                    )
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for group, names in GROUPS.items():
        for day in (30, 365):
            labels = names + ["U-global / initialized persistence"]
            map_sources = [sources[n] for n in names] + [
                {**sources["U-global"], "method": labels[-1]}
            ]
            month = 0 if day == 30 else 11
            provenance = dict(
                cases=origins,
                lead_bin_end_days=day,
                scope="Same selected checkpoints, same annual cases, one initialization; all models in the group shown",
                grid_sha256=digest(samples / "grid.npz"),
                statistics_sha256=digest(samples / "statistics.npz"),
                results_sha256=digest(results_path),
                exporter_sha256=digest(__file__),
                sources=map_sources,
                case_times={
                    o: {
                        f"day{day}_bin_midpoint": (
                            datetime.datetime.fromisoformat(o)
                            + datetime.timedelta(days=day - 2.5)
                        ).isoformat()
                    }
                    for o in origins
                },
            )
            path = output / f"{group}-day{day}.npz"
            np.savez_compressed(
                path,
                **base,
                climatology=np.stack([seasonal[month]] * len(origins)),
                reference=np.stack([references[o, day] for o in origins]),
                prediction=np.stack([fields[n, day] for n in labels]),
                provenance=np.array(json.dumps(provenance)),
            )
            artifacts[path.name] = digest(path)
    atomic_json(artifacts, output / "MAP_BUNDLES_COMPLETE.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.results, args.output)
