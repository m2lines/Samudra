# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualified A/B OM4 pretraining and observation fine-tuning with native OM4 replay."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from samudra.experiments.diffusion_evaluation import evaluate_point_metrics
from samudra.experiments.diffusion_fit import batch_at, bounded_fit
from samudra.experiments.diffusion_observations import forecast_observation_crps
from samudra.experiments.diffusion_physical import JointPhysicalForecast
from samudra.experiments.initializer_wave import InitializerWave
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--phase", choices=["om4", "observation"], required=True)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--training-members", type=int, default=2)
    parser.add_argument("--validation-members", type=int, default=8)
    parser.add_argument("--replay-every", type=int, default=4)
    parser.add_argument("--replay-weight", type=float, default=0.1)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument(
        "--device-cache",
        action="store_true",
        help="Keep all OM4 frames on GPU; requires sufficient VRAM",
    )
    parser.add_argument("--cache-reserve-gib", type=float, default=64)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    args = parser.parse_args()
    if (
        min(args.hours, args.updates, args.batch_size, args.learning_rate, args.readers)
        <= 0
    ):
        parser.error("Positive budget and training settings required")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("This campaign uses independent single-GPU arms, not DDP")
    if (
        args.training_members < 2
        or args.validation_members < 1
        or args.replay_every < 1
        or args.replay_weight <= 0
    ):
        parser.error(
            "Positive replay settings and at least two training members required"
        )
    if args.phase == "observation" and (
        args.pretrained is None or args.batch_size != 1
    ):
        parser.error("Observation phase requires --pretrained and batch size one")
    if args.phase == "om4" and args.pretrained is not None:
        parser.error("--pretrained is only used for the observation phase")
    producer = os.environ["SAMUDRA_CODE_COMMIT"]
    source = args.root / "checkpoints/om4-source/selected.pt"
    qualification = json.loads(args.qualification.read_text())
    if (
        qualification["arm"] != args.arm
        or qualification["code_commit"] != producer
        or qualification["source_checkpoint_sha256"] != digest(source)
        or qualification["data_manifest_sha256"]
        != digest(args.root / "data/observations/SHA256SUMS")
        or not qualification["dynamics_frozen"]
        or "strict_reload_om4_loss" not in qualification
        or not all(v > 0 for v in qualification["gradient_norms"].values())
    ):
        raise ValueError(
            "Production qualification does not match this arm, producer or data"
        )
    baseline_gate = args.root / "runs/qualification/baseline/QUALIFIED.json"
    if not baseline_gate.is_file() or not (args.root / "DATA_READY.json").is_file():
        raise ValueError("Baseline reproduction and staged-data verification required")
    selection = json.loads((args.root / "checkpoints/selection.json").read_text())
    reference_path = (
        args.root / "checkpoints" / selection["selected"] / "selection-reference.json"
    )
    signature = dict(
        selection_reference_sha256=digest(reference_path),
        arm=args.arm,
        phase=args.phase,
        pretrained_sha256=digest(args.pretrained) if args.pretrained else None,
        training_members=args.training_members,
        validation_members=args.validation_members,
        replay_every=args.replay_every,
        replay_weight=args.replay_weight,
        producer=producer,
        source_sha256=digest(source),
        qualification_sha256=digest(args.qualification),
        staged_data_manifest_sha256=digest(args.root / "DATA_READY.json"),
        seed=args.seed,
        batch_size=args.batch_size,
        device_cache=args.device_cache,
        learning_rate=args.learning_rate,
        sampling_steps=qualification["sampling_steps"],
        observation_manifest_sha256=qualification["data_manifest_sha256"],
        selection=(
            "Frozen upstream observation criterion on validation ensemble means"
            if args.phase == "observation"
            else "Own fixed-noise OM4 validation denoising loss for B; full-interior MSE for A. Not a cross-arm skill score."
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "protocol.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != signature:
        raise ValueError("Existing training protocol differs")
    atomic_json(signature, manifest_path)
    # Reuse native Rust loading and state-scale conversion; GPU caching is optional.
    loader_args = SimpleNamespace(
        arm="D",
        phase="reconstruction",
        seed=args.seed,
        readers=args.readers,
        data_root=str(args.root / "data/om4_onedeg_v3"),
        output=str(args.output / "loader"),
        name=args.name,
        wandb_mode=args.wandb_mode,
        normalization="instance",
        observation_normalization_root=str(args.root / "data/observations"),
        fresh_evolution=True,
        initial_checkpoint=str(source),
        wave1_root="",
        val_origins=12,
        device_cache=args.device_cache,
        device_cache_reserve_gib=args.cache_reserve_gib,
        batch_size=args.batch_size,
    )
    wave = InitializerWave(loader_args)
    try:
        if wave.run:
            wave.run.config.update({"diffusion_protocol": signature})
        core = ObservationTransfer(wave.names, "instance")
        core.initializer = wave.initializer
        core.evolution = wave.model.evolution
        core.adapter.requires_grad_(args.phase == "observation")
        model = JointPhysicalForecast(
            core,
            stochastic=args.arm == "B",
            sampling_steps=qualification["sampling_steps"],
        ).to(wave.device)
        if args.pretrained:
            completed = json.loads(
                (args.pretrained.parent / "PRETRAIN_COMPLETE.json").read_text()
            )
            if not completed["state"]["complete"] or completed[
                "best_checkpoint_sha256"
            ] != digest(args.pretrained):
                raise ValueError("Pretraining is unfinished or selected weights differ")
            saved = torch.load(args.pretrained, map_location="cpu", weights_only=False)
            parent = saved["protocol"]["signature"]
            for key in (
                "arm",
                "producer",
                "source_sha256",
                "observation_manifest_sha256",
                "sampling_steps",
            ):
                if parent[key] != signature[key]:
                    raise ValueError(f"Pretraining contract differs: {key}")
            model.load_state_dict(saved["model"], strict=True)
            del saved
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
        )
        wave.prepare(wave.trainset)
        if args.phase == "om4":
            wave.prepare(wave.valset)

        def loss(dataset, ids, seed):
            surface, past, context, truth = wave.model_initial_sample(dataset, ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.pretraining_loss(
                    surface,
                    past,
                    context,
                    truth,
                    wave.mask,
                    wave.weights,
                    torch.Generator(device=wave.device).manual_seed(seed),
                )

        def objective(step):
            return loss(
                wave.trainset,
                batch_at(step, len(wave.trainset), args.batch_size, args.seed),
                args.seed + step,
            )

        def validate(step):
            return sum(
                float(loss(wave.valset, [index], args.seed + 100000 + index))
                for index in wave.val_ids
            ) / len(wave.val_ids)

        if args.phase == "observation":
            data = Samples(args.root / "data/observations", wave.device)
            data.use_observation_normalization()
            training, validation = data.paths("train"), data.paths("validation")
            if len(training) != 243 or len(validation) != 9:
                raise ValueError("Incomplete fixed observation cohorts")
            reference = json.loads(reference_path.read_text())

            def objective(step):
                index = batch_at(step, len(training), 1, args.seed)[0]
                sample = data.load(training[index])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predictions, _ = model.forecast(
                        sample["surface"],
                        sample["atmosphere"],
                        sample["contexts"],
                        data.mask,
                        sample["validity"],
                        generator=torch.Generator(device=wave.device).manual_seed(
                            args.seed + step
                        ),
                        members=args.training_members if args.arm == "B" else 1,
                    )
                    value = (
                        forecast_observation_crps(data, predictions, sample)
                        if args.arm == "B"
                        else data.forecast_loss(predictions[0], sample)
                    )
                if step % args.replay_every == 0:
                    ids = batch_at(
                        step // args.replay_every, len(wave.trainset), 1, args.seed + 1
                    )
                    value = value + args.replay_weight * loss(
                        wave.trainset, ids, args.seed + 1000000 + step
                    )
                return value

            def validate(step):
                metrics, value = evaluate_point_metrics(
                    model, data, validation, reference, members=args.validation_members
                )
                atomic_json(
                    dict(
                        step=step,
                        score=value,
                        metrics=metrics,
                        selection_reference_sha256=digest(reference_path),
                    ),
                    args.output / f"validation-{step}.json",
                )
                return value

        state = bounded_fit(
            model,
            optimizer,
            objective,
            validate,
            args.output,
            signature,
            max_updates=args.updates,
            max_seconds=args.hours * 3600,
            checkpoint_every=args.checkpoint_every,
            validate_every=args.validate_every,
            emit=wave.emit,
        )
        atomic_json(
            dict(
                state=state,
                phase=args.phase,
                scope="Training stage complete; held-out evaluation and scientific report pending",
                best_checkpoint_sha256=digest(args.output / "best.pt"),
            ),
            args.output
            / (
                "PRETRAIN_COMPLETE.json"
                if args.phase == "om4"
                else "OBSERVATION_COMPLETE.json"
            ),
        )
    finally:
        if wave.run:
            wave.run.finish()


if __name__ == "__main__":
    main()
