#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Add saved observation-only endpoints to the uncooled presentation RMSE chart."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from samudra.experiments.observation_annual import surface_metrics
from samudra.experiments.observation_metrics import weighted_rmse
from scripts.collect_presentation_annual import KEYS, endpoint, pooled
from scripts.export_presentation_annual_maps import GROUPS
from scripts.plot_presentation_annual import plot_comparison


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_verified(path, checksum):
    if digest(path) != checksum:
        raise ValueError(f"Source hash changed: {path}")
    with np.load(path) as saved:
        return dict(saved)


def metrics(surface, ohc, arrays, data, origins_months):
    result = surface_metrics(surface, arrays["reference"], data)
    area = np.cos(np.deg2rad(data.grid["lat"]))[:, None].astype(np.float32)
    result["monthly_ohc"] = {
        month: {
            layer: weighted_rmse(ohc[i, c], arrays["reference_ohc"][i, c], area)
            for c, layer in enumerate(("0_700", "700_2000"))
        }
        for i, month in enumerate(origins_months)
    }
    # This artifact is RMSE-only; retain the two requested endpoint records.
    return {
        "leads": {day: result["leads"][day] for day in ("30", "365")},
        "monthly_ohc": {
            month: result["monthly_ohc"][month]
            for month in (origins_months[0], origins_months[-1])
        },
    }


def plot(repo, compact, reference, grid, output):
    artifacts = repo / "docs/experiments/observation-d/artifacts"
    original_folder = artifacts / "presentation-annual-2026-10-08/final"
    manifest_path = (
        artifacts / "2026-10-01-global-physical-comparison/compact-manifest.json.gz"
    )
    manifest = json.loads(gzip.decompress(manifest_path.read_bytes()))
    raw = gzip.decompress((original_folder / "results.json.gz").read_bytes())
    receipt = json.loads((original_folder / "COLLECTION_COMPLETE.json").read_text())
    if hashlib.sha256(raw).hexdigest() != receipt["files"]["results.json"]:
        raise ValueError("Original presentation result hash changed")
    original = json.loads(raw)
    origins = original["configuration"]["origins"]
    if origins != ["2015-01-01", "2018-01-01", "2021-01-01"]:
        raise ValueError("Unexpected annual cohort")
    data = SimpleNamespace(
        grid=read_verified(
            grid, manifest["sources"]["/scratch/jr7309/data/obs-d-pilot/grid.npz"]
        ),
        global_observations=True,
    )
    records: dict[str, dict[str, dict]] = {}
    source_hashes, checkpoints = {}, {}
    audit = original["models"]["U-global"]["repaired_output_audit"]["outputs"]
    for origin in origins:
        filename = origin + ".npz"
        baseline = read_verified(reference / filename, audit[filename])
        months = [f"{origin[:4]}-{month:02d}" for month in range(1, 13)]
        np.testing.assert_array_equal(baseline["months"], months)
        calculated = metrics(
            baseline["surface"], baseline["predicted_ohc"], baseline, data, months
        )
        for day in (30, 365):
            if endpoint(calculated, origin, day) != endpoint(
                original["metrics"]["U-global"][origin], origin, day
            ):
                raise ValueError("Recalculation differs from original U-global RMSE")
        source_hashes[str(reference / filename)] = audit[filename]
        for updates in (8000, 16000):
            key = f"scratch{updates:05d}"
            name = f"Obs-only: {updates:,} obs"
            checkpoint = manifest["checkpoints"][key]
            if checkpoint["lineage"]["task_counts"] != {
                "om4": 0,
                "observation": updates,
            }:
                raise ValueError("Unexpected observation-only checkpoint")
            checkpoints[name] = checkpoint["lineage"]
            filename = f"{key}-{origin}.npz"
            checksum = manifest["files"][filename]["sha256"]
            arrays = read_verified(compact / filename, checksum)
            source_hashes[str(compact / filename)] = checksum
            for field in ("reference", "reference_ohc", "months"):
                np.testing.assert_array_equal(arrays[field], baseline[field])
            if arrays["surface"].shape != (73, 2, 180, 360):
                raise ValueError("Incomplete annual surface rollout")
            for persistence in (False, True):
                label = name + (" / initialized persistence" if persistence else "")
                surface = (
                    np.broadcast_to(arrays["initial"][[0, 6]], arrays["surface"].shape)
                    if persistence
                    else arrays["surface"]
                )
                ohc = (
                    np.broadcast_to(
                        arrays["initial_ohc"], arrays["predicted_ohc"].shape
                    )
                    if persistence
                    else arrays["predicted_ohc"]
                )
                records.setdefault(label, {})[origin] = metrics(
                    surface, ohc, arrays, data, months
                )
    rows = []
    for day in (30, 365):
        control = pooled(
            [
                endpoint(
                    original["metrics"]["Training seasonal climatology"][o], o, day
                )
                for o in origins
            ]
        )
        for name, values in records.items():
            for origin in [*origins, "pooled"]:
                selected = origins if origin == "pooled" else [origin]
                values_pooled = pooled([endpoint(values[o], o, day) for o in selected])
                rows.append(
                    dict(
                        model=name,
                        origin=origin,
                        lead_days=day,
                        rmse_score=float(
                            np.mean([values_pooled[k] / control[k] for k in KEYS])
                        ),
                        **values_pooled,
                    )
                )
    original_csv = original_folder / "rmse-scores.csv"
    if digest(original_csv) != receipt["files"]["rmse-scores.csv"]:
        raise ValueError("Original RMSE table changed")
    frame = pd.concat(
        [pd.read_csv(original_csv), pd.DataFrame(rows)], ignore_index=True
    )
    names = list(dict.fromkeys(n for group in GROUPS.values() for n in group)) + list(
        checkpoints
    )
    if len(names) != 19 or any(n.endswith("-cooldown") for n in names):
        raise ValueError(
            "Expected 17 uncooled models and two observation-only endpoints"
        )
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "rmse-scores.csv", index=False)
    (output / "obs-only-metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    plot_comparison(
        frame,
        names,
        output,
        figsize=(12, 9),
        title="Uncooled models + observation-only 8k / 16k comparators\n"
        "Three January starts · same RMSE controls · training budgets differ",
    )
    provenance = dict(
        original_result_sha256=receipt["files"]["results.json"],
        original_scores_sha256=digest(original_csv),
        compact_manifest_sha256=digest(manifest_path),
        grid_sha256=digest(grid),
        sources=source_hashes,
        checkpoints=checkpoints,
        origins=origins,
        verification=[
            "All nine source annual arrays hash-verified",
            "All scratch surface and OHC targets exactly equal U-global targets",
            "Both leads of all three U-global RMSE records reproduced exactly",
            "Original 17-model CSV rows retained without recomputing their scores",
        ],
        score="Equal-origin MSE pooling, square root, then mean of four RMSE ratios to the original pooled training climatology; OHC January/December means",
        scope="Two fixed endpoints of one obs-only trajectory, no cooldown; larger training budgets and a related architecture, not a matched ablation of the 17-model screen. No new inference or checkpoint selection.",
        code={
            path: digest(repo / path)
            for path in (
                "scripts/plot_presentation_obs_comparators.py",
                "scripts/plot_presentation_annual.py",
                "scripts/collect_presentation_annual.py",
                "src/samudra/experiments/observation_annual.py",
                "src/samudra/experiments/observation_metrics.py",
                "src/samudra/metrics/kernels.py",
            )
        },
        files={
            p.name: digest(p)
            for p in sorted(output.iterdir())
            if p.is_file() and p.name != "provenance.json" and p.suffix != ".license"
        },
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        frame[
            (frame.origin == "pooled") & frame.model.str.startswith("Obs-only")
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("repo", "compact", "reference", "grid", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    plot(args.repo, args.compact, args.reference, args.grid, args.output)
