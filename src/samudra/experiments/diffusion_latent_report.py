# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Observation point/calibration exports for independently decoded latent forecasts."""

import argparse
import json
import os
from pathlib import Path

import torch

from samudra.experiments.diffusion_evaluation import (
    EnsembleMeanForecast,
    evaluate_point_metrics,
)
from samudra.experiments.diffusion_latent_forecast import LatentOceanForecast
from samudra.experiments.diffusion_report import (
    MAP_ORIGINS,
    member_records,
    verified_annual_origins,
)
from samudra.experiments.observation_annual import evaluate_origins
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def load_latent_checkpoint(path, data):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    signature = saved["protocol"]["signature"]
    if signature["architecture"] != "persistent-latent-diffusion":
        raise ValueError("Not a persistent latent checkpoint")
    core = ObservationTransfer(list(data.grid["names"]), "instance")
    model = LatentOceanForecast(
        core.initializer,
        core.adapter,
        stochastic=True,
        latent_width=signature["latent_width"],
        latent_shape=tuple(signature["latent_shape"]),
        processor_depth=signature["processor_depth"],
        decoder_width=signature["decoder_width"],
        sampling_steps=signature["sampling_steps"],
    ).to(data.device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annual", action="store_true")
    parser.add_argument(
        "--pretraining-maps",
        action="store_true",
        help="Export the three fixed observation-input examples before adaptation",
    )
    parser.add_argument("--members", type=int, default=8)
    args = parser.parse_args()
    if args.members < 2:
        parser.error("Need at least two members")
    if args.pretraining_maps and args.annual:
        parser.error("Pretraining maps do not include annual evaluation")
    marker = (
        "PRETRAIN_COMPLETE.json"
        if args.pretraining_maps
        else "OBSERVATION_COMPLETE.json"
    )
    completed = json.loads((args.checkpoint.parent / marker).read_text())
    if not completed["state"]["complete"] or completed[
        "best_checkpoint_sha256"
    ] != digest(args.checkpoint):
        raise ValueError("Training stage/checkpoint not verified")
    data = Samples(args.root / "data/observations", "cuda")
    data.use_observation_normalization()
    model, signature = load_latent_checkpoint(args.checkpoint, data)
    if digest(data.root / "SHA256SUMS") != signature["observation_manifest_sha256"]:
        raise ValueError("Observation inputs changed")
    if (
        digest(data.root / "statistics.npz")
        != signature["observation_statistics_sha256"]
    ):
        raise ValueError("Observation scales changed")
    paths = data.paths("test")
    if len(paths) != 96 or not set(MAP_ORIGINS).issubset({p.stem for p in paths}):
        raise ValueError("Incomplete frozen held-out cohort")
    selection = json.loads((args.root / "checkpoints/selection.json").read_text())
    ref = args.root / "checkpoints" / selection["selected"] / "selection-reference.json"
    if digest(ref) != signature["selection_reference_sha256"]:
        raise ValueError("Frozen score reference changed")
    if args.pretraining_maps:
        paths = [p for p in paths if p.stem in MAP_ORIGINS]
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        scope="pretraining-maps" if args.pretraining_maps else "observation-report",
        evaluator_commit=os.environ["SAMUDRA_CODE_COMMIT"],
        training_protocol=signature,
        checkpoint_sha256=digest(args.checkpoint),
        selection_reference_sha256=digest(ref),
        members=args.members,
        seed=4041729,
        origins=[p.stem for p in paths],
        uncertainty="Independent conditional readouts on one deterministic latent path. No coherent ensemble-trajectory claim.",
        monthly="Each independently decoded sample sequence is averaged before scoring; temporal noise covariance differs from physical diffusion baseline.",
    )
    manifest = args.output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Report protocol differs")
    atomic_json(protocol, manifest)
    if args.pretraining_maps:
        member_records(
            model, data, paths, args.output, members=args.members, seed=4041729
        )
        atomic_json(protocol, args.output / "PRETRAINING_MAPS_COMPLETE.json")
        return
    metrics, score = evaluate_point_metrics(
        model,
        data,
        paths,
        json.loads(ref.read_text()),
        members=args.members,
        seed=4041729,
        export=args.output / "point-fields.npz",
    )
    atomic_json(
        dict(metrics=metrics, reporting_score=score), args.output / "point-metrics.json"
    )
    member_records(model, data, paths, args.output, members=args.members, seed=4041729)
    atomic_json(protocol, args.output / "MONTHLY_REPORT_COMPLETE.json")
    if args.annual:
        origins = verified_annual_origins(
            args.root, dict(staged_data_manifest_sha256=signature["data_sha256"])
        )
        annual_protocol = dict(
            **protocol,
            annual_origins=[p.parent.name for p in origins],
            annual_manifests={
                str(p.relative_to(args.root / "data/annual_observations")): digest(p)
                for p in origins
            },
            annual_scope="Single latent initialization; conditional diffusion readouts at each lead; ensemble-mean point metrics only",
        )
        output = args.output / "annual"
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(annual_protocol, output / "input.json")
        evaluate_origins(
            EnsembleMeanForecast(model, members=args.members, seed=4041729),
            data,
            origins,
            output,
            annual_protocol,
        )


if __name__ == "__main__":
    main()
