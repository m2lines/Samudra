#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect verified annual statistics and compact, matched day-365 map bundles."""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from samudra.experiments.observation_pilot import digest


def collect(root, output):
    config = json.loads((root / "paths.json").read_text())
    result = {
        "collected_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "configuration": config,
        "collector_sha256": digest(__file__),
        "models": {},
    }
    samples = Path(config["observations"])
    with np.load(samples / "grid.npz") as grid:
        base = {k: grid[k] for k in ["lat", "lon"]}
        base["mask"] = grid["mask"][0]
    with np.load(samples / "statistics.npz") as stats:
        # All established annual cases start January 1 and end in December.
        base["climatology"] = np.stack(
            [stats["surface_climatology"][11]] * len(config["origins"])
        )
    fields, references = {}, {}
    for row in config["models"]:
        run = root / row["name"]
        verified = json.loads((run / "VERIFIED.json").read_text())
        if (
            verified["config_sha256"] != digest(root / "paths.json")
            or verified["model"] != row
        ):
            raise ValueError("Annual result audit differs")
        for name, checksum in verified["outputs"].items():
            if digest(run / name) != checksum:
                raise ValueError("Annual output changed: " + str(run / name))
        complete = json.loads((run / "COMPLETE.json").read_text())
        if (
            sorted(complete["origins"]) != config["origins"]
            or complete["inputs"]["checkpoint_sha256"] != row["checkpoint_sha256"]
        ):
            raise ValueError("Annual cohort or selected checkpoint differs")
        records, evolved, persisted, truth = {}, [], [], []
        for origin in config["origins"]:
            item = complete["origins"][origin]
            records[origin] = {
                kind: json.loads((run / item[kind]).read_text())
                for kind in ["metrics", "persistence"]
            }
            with np.load(run / item["arrays"]) as arrays:
                if arrays["surface"].shape[0] != 73:
                    raise ValueError("Incomplete annual forecast")
                evolved.append(arrays["surface"][-1])
                persisted.append(arrays["persistence_surface"][-1])
                truth.append(arrays["reference"][-1, :2])
            records[origin]["arrays_sha256"] = verified["outputs"][item["arrays"]]
        result["models"][row["name"]] = dict(
            inputs=complete["inputs"], selected=row, origins=records
        )
        fields[row["name"]] = np.stack(evolved)
        fields[row["name"] + " / initialized persistence"] = np.stack(persisted)
        references[row["name"]] = np.stack(truth)
    reference = next(iter(references.values()))
    for value in references.values():
        np.testing.assert_array_equal(reference, value)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    rows = {r["name"]: r for r in config["models"]}
    for name in rows:
        if name.endswith("-cooldown"):
            continue
        labels = [
            name,
            name + "-cooldown",
            name + "-cooldown / initialized persistence",
        ]
        sources = []
        for label in labels:
            model = label.split(" / ")[0]
            sources.append(
                dict(
                    method=label,
                    path=str(root / model),
                    selected_checkpoint_sha256=rows[model]["checkpoint_sha256"],
                    source_checkpoint_sha256=None,
                )
            )
        provenance = dict(
            cases=config["origins"],
            lead_bin_end_days=365,
            scope="Continuous 73-step forecasts; last five-day bin; one initialization and no future observation corrections",
            grid_sha256=digest(samples / "grid.npz"),
            statistics_sha256=digest(samples / "statistics.npz"),
            script_sha256=digest(__file__),
            sources=sources,
            case_times={
                origin: {
                    "day365_bin_midpoint": (
                        datetime.datetime.fromisoformat(origin)
                        + datetime.timedelta(days=362.5)
                    ).isoformat()
                }
                for origin in config["origins"]
            },
        )
        np.savez_compressed(
            output / (name + "-maps.npz"),
            **base,
            reference=reference,
            prediction=np.stack([fields[k] for k in labels]),
            provenance=np.array(json.dumps(provenance)),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root, args.output)
