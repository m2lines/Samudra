# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Held-out A/B monthly reports and optional continuous-year point diagnostics.

Annual ensemble calibration and additional structural diagnostics remain separate.
This command never selects a checkpoint using held-out scores.
"""

import argparse
import json
import os
from pathlib import Path

import torch

from samudra.experiments.diffusion_calibration import observation_ensemble_statistics
from samudra.experiments.diffusion_evaluation import (
    EnsembleMeanForecast,
    evaluate_point_metrics,
)
from samudra.experiments.diffusion_physical import JointPhysicalForecast
from samudra.experiments.observation_annual import evaluate_origins
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


@torch.no_grad()
def calibration_records(model, data, paths, output, *, members=8, seed=4041729):
    """Save additive sums per origin; preserve channel/lead axes for later aggregation."""
    if not model.stochastic or members < 2:
        raise ValueError(
            "Calibration requires a stochastic model and at least two members"
        )
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        sample = data.load(path)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            trajectories, _ = model.forecast(
                sample["surface"],
                sample["atmosphere"],
                sample["contexts"],
                data.mask,
                sample["validity"],
                generator=torch.Generator(device=sample["surface"].device).manual_seed(
                    seed
                ),
                members=members,
            )
        statistics = observation_ensemble_statistics(data, trajectories, sample)
        payload = {
            group: {
                key: value.cpu().tolist() if isinstance(value, torch.Tensor) else value
                for key, value in fields.items()
            }
            for group, fields in statistics.items()
        }
        atomic_json(
            dict(origin=path.stem, statistics=payload), output / (path.stem + ".json")
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4041729)
    parser.add_argument(
        "--annual",
        action="store_true",
        help="Also report the frozen three annual test origins",
    )
    args = parser.parse_args()
    if args.members < 2:
        parser.error("Use at least two members for calibration")
    producer = os.environ["SAMUDRA_CODE_COMMIT"]
    complete = json.loads(
        (args.checkpoint.parent / "OBSERVATION_COMPLETE.json").read_text()
    )
    checkpoint_hash = digest(args.checkpoint)
    if (
        not complete["state"]["complete"]
        or complete["best_checkpoint_sha256"] != checkpoint_hash
    ):
        raise ValueError(
            "Require the selected checkpoint from a completed observation stage"
        )
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    signature = saved["protocol"]["signature"]
    if signature["phase"] != "observation" or signature["arm"] not in ("A", "B"):
        raise ValueError("This evaluator implements completed observation A/B only")
    data = Samples(args.root / "data/observations", "cuda")
    data.use_observation_normalization()
    if digest(data.root / "SHA256SUMS") != signature["observation_manifest_sha256"]:
        raise ValueError("Observation data differ from training")
    chosen = json.loads((args.root / "checkpoints/selection.json").read_text())
    reference_path = (
        args.root / "checkpoints" / chosen["selected"] / "selection-reference.json"
    )
    if digest(reference_path) != signature["selection_reference_sha256"]:
        raise ValueError("Frozen score reference changed")
    reference = json.loads(reference_path.read_text())
    model = JointPhysicalForecast(
        ObservationTransfer(list(data.grid["names"]), "instance"),
        stochastic=signature["arm"] == "B",
        sampling_steps=signature["sampling_steps"],
    ).to("cuda")
    model.load_state_dict(saved["model"], strict=True)
    del saved
    model.eval()
    paths = data.paths("test")
    if len(paths) != 96:
        raise ValueError("Require the complete frozen 96-origin reporting cohort")
    protocol = dict(
        evaluator_commit=producer,
        checkpoint_sha256=checkpoint_hash,
        training_protocol=signature,
        members=args.members if model.stochastic else 1,
        seed=args.seed,
        origins=[path.stem for path in paths],
        scope="Monthly held-out evaluation only; annual and structural reports remain separate",
        calibration_units="Observation-standardized units; raw wet-area additive sums, not pre-averaged scores",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    contract = args.output / "protocol.json"
    if contract.exists() and json.loads(contract.read_text()) != protocol:
        raise ValueError(
            "Existing evaluation belongs to a different checkpoint or protocol"
        )
    atomic_json(protocol, contract)
    metrics, score = evaluate_point_metrics(
        model,
        data,
        paths,
        reference,
        members=args.members,
        seed=args.seed,
        export=args.output / "point-fields.npz",
    )
    atomic_json(
        dict(metrics=metrics, reporting_score=score), args.output / "point-metrics.json"
    )
    if model.stochastic:
        calibration_records(
            model,
            data,
            paths,
            args.output / "calibration",
            members=args.members,
            seed=args.seed,
        )
    atomic_json(
        dict(
            **protocol,
            calibration="complete"
            if model.stochastic
            else "not_applicable_deterministic",
        ),
        args.output / "MONTHLY_REPORT_COMPLETE.json",
    )

    if args.annual:
        annual = args.root / "data/annual_observations"
        if not (annual / "DATA_READY.json").exists():
            raise ValueError("Annual data must pass full transfer verification")
        origins = sorted((annual / "test").glob("*/COMPLETE.json"))
        if [path.parent.name for path in origins] != [
            "2015-01-01",
            "2018-01-01",
            "2021-01-01",
        ]:
            raise ValueError("Require the frozen three-origin annual test cohort")
        annual_protocol = dict(
            **protocol,
            annual_origins=[path.parent.name for path in origins],
            annual_manifests={
                str(path.relative_to(annual)): digest(path) for path in origins
            },
            annual_scope="Single initialization; ensemble mean after independent physical evolution; point metrics only",
            forcing="Prescribed ERA5 through trained adapter; not an operational forcing forecast",
            eke="Anomalies relative to each sequence's own year mean",
        )
        annual_output = args.output / "annual"
        annual_output.mkdir(parents=True, exist_ok=True)
        contract = annual_output / "input.json"
        if contract.exists() and json.loads(contract.read_text()) != annual_protocol:
            raise ValueError("Existing annual evaluation protocol differs")
        atomic_json(annual_protocol, contract)
        evaluate_origins(
            EnsembleMeanForecast(model, members=args.members, seed=args.seed),
            data,
            origins,
            annual_output,
            annual_protocol,
        )


if __name__ == "__main__":
    main()
