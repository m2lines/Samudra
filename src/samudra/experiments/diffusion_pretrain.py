# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualified, resumable A/B OM4 pretraining with the existing native Rust loader."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from samudra.experiments.diffusion_fit import batch_at, bounded_fit
from samudra.experiments.diffusion_physical import JointPhysicalForecast
from samudra.experiments.initializer_wave import InitializerWave
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--readers", type=int, default=4)
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
    signature = dict(
        arm=args.arm,
        producer=producer,
        source_sha256=digest(source),
        qualification_sha256=digest(args.qualification),
        staged_data_manifest_sha256=digest(args.root / "DATA_READY.json"),
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sampling_steps=qualification["sampling_steps"],
        observation_manifest_sha256=qualification["data_manifest_sha256"],
        selection="Own fixed-noise OM4 validation denoising loss for B; full-interior MSE for A. Not a cross-arm skill score.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "protocol.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != signature:
        raise ValueError("Existing pretraining protocol differs")
    atomic_json(signature, manifest_path)
    # Reuse the verified source loading, state-scale conversion and compact GPU cache.
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
        device_cache=True,
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
        core.adapter.requires_grad_(False)
        model = JointPhysicalForecast(
            core,
            stochastic=args.arm == "B",
            sampling_steps=qualification["sampling_steps"],
        ).to(wave.device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
        )
        wave.prepare(wave.trainset)
        wave.prepare(wave.valset)

        def loss(dataset, ids, seed):
            surface, past, context, truth, _, _ = wave.model_sample(dataset, ids)
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

        def validate(step):
            return sum(
                float(loss(wave.valset, [index], args.seed + 100000 + index))
                for index in wave.val_ids
            ) / len(wave.val_ids)

        state = bounded_fit(
            model,
            optimizer,
            lambda step: loss(
                wave.trainset,
                batch_at(step, len(wave.trainset), args.batch_size, args.seed),
                args.seed + step,
            ),
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
                phase="OM4 pretraining only; observation fine-tuning and evaluation pending",
                best_checkpoint_sha256=digest(args.output / "best.pt"),
            ),
            args.output / "PRETRAIN_COMPLETE.json",
        )
    finally:
        if wave.run:
            wave.run.finish()


if __name__ == "__main__":
    main()
