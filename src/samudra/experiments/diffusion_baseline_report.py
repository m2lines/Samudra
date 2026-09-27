# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the unchanged selected deterministic baseline with A/B observation operators."""

import argparse
import json
import os
from pathlib import Path

import torch

from samudra.experiments.diffusion_evaluation import (
    DeterministicMembers,
    evaluate_point_metrics,
)
from samudra.experiments.diffusion_report import (
    MAP_ORIGINS,
    member_records,
    verified_annual_origins,
)
from samudra.experiments.observation_annual import evaluate_origins
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    chosen = json.loads((args.root / "checkpoints/selection.json").read_text())
    selected = args.root / "checkpoints" / chosen["selected"]
    manifest = json.loads((selected / "manifest.json").read_text())
    best = json.loads((selected / "best.json").read_text())
    checkpoint = selected / "best.pt"
    if digest(checkpoint) != best["checkpoint_sha256"]:
        raise ValueError("Selected deterministic baseline changed")
    if manifest["arguments"]["normalization"] != "instance":
        raise ValueError("Unqualified baseline normalization")
    data = Samples(args.root / "data/observations", "cuda")
    data.use_observation_normalization()
    for name, key in (
        ("grid.npz", "grid_sha256"),
        ("statistics.npz", "statistics_sha256"),
        ("SHA256SUMS", "data_manifest_sha256"),
    ):
        if digest(data.root / name) != manifest[key]:
            raise ValueError(f"Baseline observation contract changed: {name}")
    model = ObservationTransfer(list(data.grid["names"]), "instance").cuda()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["model"],
        strict=True,
    )
    model.eval().requires_grad_(False)
    members = DeterministicMembers(model).eval()
    paths = data.paths("test")
    if len(paths) != 96 or not set(MAP_ORIGINS).issubset({p.stem for p in paths}):
        raise ValueError("Require frozen monthly cohort")
    signature = dict(staged_data_manifest_sha256=digest(args.root / "DATA_READY.json"))
    origins = verified_annual_origins(args.root, signature)
    protocol = dict(
        evaluator_commit=os.environ["SAMUDRA_CODE_COMMIT"],
        checkpoint_sha256=digest(checkpoint),
        observation_manifest_sha256=manifest["data_manifest_sha256"],
        staged_data_manifest_sha256=signature["staged_data_manifest_sha256"],
        members=1,
        origins=[p.stem for p in paths],
        annual_origins=[p.parent.name for p in origins],
        annual_manifests={
            str(p.relative_to(args.root / "data/annual_observations")): digest(p)
            for p in origins
        },
        scope="Unchanged selected deterministic baseline; no additional decoder, no fitting; point-mass CRPS equals MAE",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    contract = args.output / "protocol.json"
    if contract.exists() and json.loads(contract.read_text()) != protocol:
        raise ValueError("Baseline evaluation contract changed")
    atomic_json(protocol, contract)
    reference = json.loads((selected / "selection-reference.json").read_text())
    metrics, score = evaluate_point_metrics(
        members,
        data,
        paths,
        reference,
        members=1,
        export=args.output / "point-fields.npz",
    )
    atomic_json(
        dict(metrics=metrics, reporting_score=score), args.output / "point-metrics.json"
    )
    member_records(members, data, paths, args.output, members=1)
    atomic_json(
        dict(**protocol, complete=True), args.output / "MONTHLY_REPORT_COMPLETE.json"
    )
    annual = args.output / "annual"
    annual.mkdir(exist_ok=True)
    atomic_json(protocol, annual / "input.json")
    evaluate_origins(model, data, origins, annual, protocol)


if __name__ == "__main__":
    main()
