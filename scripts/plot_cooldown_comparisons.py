#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Replot existing cooldown evaluations using the presentation score definitions."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.collect_presentation_annual import KEYS, endpoint, pooled
from scripts.plot_presentation_annual import plot_comparison
from scripts.plot_presentation_spectral import digest
from scripts.plot_presentation_spectral import plot as plot_spectral


def plot(repo, output):
    artifacts = repo / "docs/experiments/observation-d/artifacts"
    early_folder = artifacts / "early-fine-annual-2026-10-08/final"
    original_folder = artifacts / "presentation-annual-2026-10-08/final"
    verification = json.loads((early_folder / "verification.json").read_text())
    if (
        digest(early_folder / "results.json")
        != verification["report_files_sha256"]["results.json"]
    ):
        raise ValueError("Early/fine annual result hash changed")
    annual = json.loads((early_folder / "results.json").read_text())
    original_bytes = gzip.decompress((original_folder / "results.json.gz").read_bytes())
    receipt = json.loads((original_folder / "COLLECTION_COMPLETE.json").read_text())
    if hashlib.sha256(original_bytes).hexdigest() != receipt["files"]["results.json"]:
        raise ValueError("Presentation annual result hash changed")
    original = json.loads(original_bytes)
    for key in ("origins", "image", "observations", "annual_data"):
        if annual["configuration"][key] != original["configuration"][key]:
            raise ValueError("Annual evaluation contract differs: " + key)
    origins = annual["configuration"]["origins"]
    if origins != ["2015-01-01", "2018-01-01", "2021-01-01"]:
        raise ValueError("Unexpected annual cohort")
    names = [name for name in annual["models"] if name.endswith("-cooldown")]
    if len(names) != 5:
        raise ValueError("Expected all five cooldown models")
    comparison_names = [
        n for name in names for n in (name.removesuffix("-cooldown"), name)
    ]
    records = {
        "Training seasonal climatology": original["metrics"][
            "Training seasonal climatology"
        ]
    }
    checkpoints = {}
    for name in names:
        model = annual["models"][name]
        checksum = model["selected"]["checkpoint_sha256"]
        audit = verification["models"][name]
        if (
            checksum != model["inputs"]["checkpoint_sha256"]
            or checksum != audit["COMPLETE.json"]["inputs"]["checkpoint_sha256"]
        ):
            raise ValueError("Annual selected checkpoint mismatch: " + name)
        if (
            model["inputs"]["data_manifests"]
            != annual["models"]["U-global"]["inputs"]["data_manifests"]
        ):
            raise ValueError("Annual target cohort mismatch")
        if sorted(model["origins"]) != origins:
            raise ValueError("Incomplete annual evaluation")
        checkpoints[name] = checksum
        for kind in ("metrics", "persistence"):
            label = name + (
                " / initialized persistence" if kind == "persistence" else ""
            )
            records[label] = {o: model["origins"][o][kind] for o in origins}
        # Connect the two verified collections through the identical parent outputs.
        parent = name.removesuffix("-cooldown")
        if (
            annual["models"][parent]["inputs"]["checkpoint_sha256"]
            != original["models"][parent]["model"]["checkpoint_sha256"]
        ):
            raise ValueError("Shared parent checkpoint differs")
        for label in (parent, parent + " / initialized persistence"):
            records[label] = original["metrics"][label]
        for origin in origins:
            for day in (30, 365):
                a = endpoint(
                    annual["models"][parent]["origins"][origin]["metrics"], origin, day
                )
                b = endpoint(original["metrics"][parent][origin], origin, day)
                if a != b:
                    raise ValueError("Shared parent metric outputs differ")

    rows = []
    for day in (30, 365):
        climatology = records["Training seasonal climatology"]
        control = pooled([endpoint(climatology[o], o, day) for o in origins])
        if any(not np.isfinite(control[k]) or control[k] <= 0 for k in KEYS):
            raise ValueError("Invalid climatology denominator")
        for name, by_origin in records.items():
            for origin in [*origins, "pooled"]:
                selected = origins if origin == "pooled" else [origin]
                values = pooled([endpoint(by_origin[o], o, day) for o in selected])
                rows.append(
                    dict(
                        model=name,
                        origin=origin,
                        lead_days=day,
                        rmse_score=float(
                            np.mean([values[k] / control[k] for k in KEYS])
                        ),
                        **values,
                    )
                )
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "rmse-scores.csv", index=False)
    plot_comparison(
        frame,
        comparison_names,
        output,
        title="Matched constant-rate and cooldown models · selected checkpoints · three January starts",
        figsize=(13, 6.5),
        compare_cooldown=True,
    )
    plot_spectral(
        repo,
        output,
        names=names,
        audits=[
            artifacts
            / "early-fine-cooldown-2026-10-08/final/summary/summary-provenance.json"
        ],
        checkpoints=checkpoints,
        title_prefix="Cooldown models · ",
        figsize=(11.5, 4.6),
    )
    sources = [
        early_folder / "results.json",
        early_folder / "verification.json",
        original_folder / "results.json.gz",
        original_folder / "COLLECTION_COMPLETE.json",
    ]
    provenance = dict(
        script_sha256=digest(Path(__file__)),
        sources={str(p.relative_to(repo)): digest(p) for p in sources},
        selected_checkpoint_sha256=checkpoints,
        rmse_selected_checkpoint_sha256={
            n: annual["models"][n]["inputs"]["checkpoint_sha256"]
            for n in comparison_names
        },
        origins=origins,
        rmse_definition="Same four-component score as presentation: equal-origin MSE pooling then square root, normalized by common pooled training-climatology errors at each lead; SST, geostrophic velocity, January/December OHC in two layers",
        spectral_definition="Original 27-term SST/ADT/EKE score at 5/15/30 days over 96 monthly origins; see spectral-provenance.json",
        selection="Original short-horizon integrated-plus-spectral validation selection; no new training, inference or reselection",
        files={
            p.name: digest(p)
            for p in sorted(output.iterdir())
            if p.is_file() and p.name != "provenance.json" and p.suffix != ".license"
        },
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        frame[frame.origin == "pooled"][["model", "lead_days", "rmse_score"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.repo, args.output)
