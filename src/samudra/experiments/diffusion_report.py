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
from typing import Any

import numpy as np
import torch

from samudra.experiments.diffusion_calibration import observation_ensemble_statistics
from samudra.experiments.diffusion_evaluation import (
    EnsembleMeanForecast,
    evaluate_point_metrics,
)
from samudra.experiments.diffusion_physical import JointPhysicalForecast
from samudra.experiments.diffusion_structure import field_structure
from samudra.experiments.observation_annual import evaluate_origins
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples

MAP_ORIGINS = ("2015-01", "2018-01", "2021-01")


def json_statistics(groups):
    """Keep unsupported structure values null in portable JSON, with zero support."""
    result: dict[str, dict[str, Any]] = {}
    for group, fields in groups.items():
        result[group] = {}
        for key, value in fields.items():
            if isinstance(value, torch.Tensor):
                array = value.cpu().numpy()
                objects = array.astype(object)
                objects[~np.isfinite(array)] = None
                value = objects.tolist()
            result[group][key] = value
    return result


@torch.no_grad()
def member_records(model, data, paths, output, *, members=8, seed=4041729):
    """Save calibration, member/mean structure, and fixed native-grid map examples."""
    if model.stochastic and members < 2:
        raise ValueError("Calibration requires at least two members")
    members = members if model.stochastic else 1
    model.eval()
    for directory in ("calibration", "structure", "members"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for path in paths:
        sample = data.load(path)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            trajectories, initial = model.forecast(
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
        if model.stochastic:
            statistics = observation_ensemble_statistics(data, trajectories, sample)
            atomic_json(
                dict(origin=path.stem, statistics=json_statistics(statistics)),
                output / "calibration" / (path.stem + ".json"),
            )
        physical = trajectories.float() * data.std + data.mean
        monthly = (
            physical * sample["month_weights"][None, None, :, None, None, None]
        ).sum(2)
        observed_mask = data.ts_mask.bool() & torch.isfinite(sample["interior"])
        interior = monthly[:, :, data.ts_indices]
        structure = dict(
            full_state_members=field_structure(physical, data.mask, data.area),
            full_state_ensemble_mean=field_structure(
                physical.mean(0), data.mask, data.area
            ),
            monthly_interior_members=field_structure(
                interior, observed_mask, data.area
            ),
            monthly_interior_ensemble_mean=field_structure(
                interior.mean(0), observed_mask, data.area
            ),
            monthly_interior_observations=field_structure(
                sample["interior"], observed_mask, data.area
            ),
        )
        atomic_json(
            dict(origin=path.stem, statistics=json_statistics(structure)),
            output / "structure" / (path.stem + ".json"),
        )
        if path.stem in MAP_ORIGINS:
            destination = output / "members" / (path.stem + ".npz")
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    initial=(initial.float() * data.std + data.mean).cpu().numpy(),
                    states_at_leads=physical[:, :, [0, 2, 5]].cpu().numpy(),
                    monthly_states=monthly.cpu().numpy(),
                    observed_monthly_interior=sample["interior"].cpu().numpy(),
                    observed_surface=sample["raw_surface"].cpu().numpy(),
                    month_weights=sample["month_weights"].cpu().numpy(),
                    leads_days=[5, 15, 30],
                    forecast_midpoints=sample["raw"]["midpoints"][19:],
                    channel_names=data.grid["names"],
                    interior_channel_indices=data.ts_indices,
                    mask=data.grid["mask"],
                    lat=data.grid["lat"],
                    lon=data.grid["lon"],
                )
            temporary.replace(destination)
        print(
            json.dumps(dict(event="member_diagnostics_origin", origin=path.stem)),
            flush=True,
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
    if len(paths) != 96 or not set(MAP_ORIGINS).issubset({p.stem for p in paths}):
        raise ValueError("Require the complete frozen 96-origin reporting cohort")
    protocol = dict(
        evaluator_commit=producer,
        checkpoint_sha256=checkpoint_hash,
        training_protocol=signature,
        members=args.members if model.stochastic else 1,
        seed=args.seed,
        origins=[path.stem for path in paths],
        scope="Monthly point, calibration and member-field diagnostics; annual point reporting optional; additional controls remain",
        calibration_units="Observation-standardized units; raw wet-area additive sums, not pre-averaged scores",
        map_origins=list(MAP_ORIGINS),
        structure_units="Physical field units and squared native-cell increments, not physical gradients; unsupported values null",
        structure_scope="Individual members and ensemble mean separately; monthly T/S comparisons share observed support; unsupervised velocity is descriptive only",
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
    member_records(
        model, data, paths, args.output, members=args.members, seed=args.seed
    )
    atomic_json(
        dict(
            **protocol,
            member_structure="complete",
            native_grid_member_exports="complete",
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
