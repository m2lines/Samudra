# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Scratch latent recurrence with direct initial/future OM4 diffusion supervision."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from samudra.experiments.diffusion_evaluation import evaluate_point_metrics
from samudra.experiments.diffusion_fit import batch_at, bounded_fit
from samudra.experiments.diffusion_latent_forecast import LatentOceanForecast
from samudra.experiments.diffusion_observations import forecast_observation_crps
from samudra.experiments.initializer_wave import InitializerWave
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_state import advance_season


def native_loss(model, wave, dataset, ids, seed):
    surface, past, context, truth, forcing, labels = wave.model_sample(dataset, ids)
    # Context describes the target midpoint, as in the observational forecast.
    contexts = torch.stack(
        [advance_season(context, (step + 1) * 5) for step in range(forcing.shape[1])], 1
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model.pretraining_loss(
            surface,
            past,
            context,
            truth,
            forcing,
            contexts,
            labels,
            wave.mask,
            wave.weights,
            torch.Generator(device=wave.device).manual_seed(seed),
        )


def gradient_norm(module):
    return math.sqrt(
        sum(
            float(p.grad.float().square().sum())
            for p in module.parameters()
            if p.grad is not None
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["qualify", "om4", "observation"], required=True
    )
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--updates", type=int, default=24000)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--invocation-hours", type=float)
    parser.add_argument("--decoder-width", type=int, default=192)
    parser.add_argument("--latent-width", type=int, default=128)
    parser.add_argument("--processor-depth", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--device-cache", action="store_true")
    parser.add_argument("--cache-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--cache-reserve-gib", type=float, default=32)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    if (
        min(
            args.updates,
            args.hours,
            args.decoder_width,
            args.latent_width,
            args.processor_depth,
            args.sampling_steps,
        )
        <= 0
    ):
        parser.error("Positive budgets and architecture sizes required")
    if args.phase != "qualify" and args.qualification is None:
        parser.error("A real-grid qualification is required")
    if (args.phase == "observation") != (args.pretrained is not None):
        parser.error("Only the observation phase requires --pretrained")
    producer = os.environ["SAMUDRA_CODE_COMMIT"]
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.root / "checkpoints/om4-source/selected.pt"
    selection = json.loads((args.root / "checkpoints/selection.json").read_text())
    reference_path = (
        args.root / "checkpoints" / selection["selected"] / "selection-reference.json"
    )
    reference_hash = digest(reference_path)
    contract = dict(
        selection_reference_sha256=reference_hash,
        producer=producer,
        architecture="persistent-latent-diffusion",
        initialization="scratch",
        decoder_width=args.decoder_width,
        latent_width=args.latent_width,
        latent_shape=[45, 90],
        processor_depth=args.processor_depth,
        sampling_steps=args.sampling_steps,
        device_cache=args.device_cache,
        cache_device=args.cache_device,
        cache_reserve_gib=args.cache_reserve_gib,
        readers=args.readers,
        history=19,
        native_leads=6,
        source_sha256=digest(source),
        data_sha256=digest(args.root / "DATA_READY.json"),
        observation_manifest_sha256=digest(args.root / "data/observations/SHA256SUMS"),
        observation_statistics_sha256=digest(
            args.root / "data/observations/statistics.npz"
        ),
        learning_rate=1e-4,
        replay_every=4,
        replay_weight=0.1,
        training_members=2,
        validation_members=8,
        recurrence="No decoded states, interior targets or readout noise enter recurrence",
        uncertainty="Independent conditional readouts; not coherent ensemble trajectories",
        native_context="Target midpoint, anchor plus (step+1)*5 days",
    )
    if args.qualification:
        qualification = json.loads(args.qualification.read_text())
        if qualification["contract"] != contract or not qualification["qualified"]:
            raise ValueError("Real-grid qualification contract differs")
    signature = dict(
        **contract,
        phase=args.phase,
        seed=args.seed,
        pretrained_sha256=digest(args.pretrained) if args.pretrained else None,
        max_updates=args.updates,
        max_seconds=args.hours * 3600,
    )
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != signature:
        raise ValueError("Existing run protocol differs")
    atomic_json(signature, protocol_path)
    if args.phase != "qualify":
        marker = (
            "PRETRAIN_COMPLETE.json"
            if args.phase == "om4"
            else "OBSERVATION_COMPLETE.json"
        )
        finished_path = args.output / marker
        if finished_path.exists():
            finished = json.loads(finished_path.read_text())
            if finished["state"]["complete"]:
                if finished["best_checkpoint_sha256"] != digest(
                    args.output / "best.pt"
                ):
                    raise ValueError("Completed selected checkpoint changed")
                print(
                    json.dumps(dict(event="stage_already_complete", phase=args.phase)),
                    flush=True,
                )
                return
    loader_args = SimpleNamespace(
        arm="D",
        phase="reconstruction",
        seed=args.seed,
        readers=args.readers,
        data_root=str(args.root / "data/om4_onedeg_v3"),
        output=str(args.output / "loader"),
        name=f"latent-D-{args.seed}-{args.phase}",
        wandb_mode="offline",
        normalization="instance",
        observation_normalization_root=str(args.root / "data/observations"),
        fresh_evolution=True,
        initial_checkpoint=str(source),
        wave1_root="",
        val_origins=12,
        device_cache=args.device_cache,
        cache_device=args.cache_device,
        device_cache_reserve_gib=args.cache_reserve_gib,
        batch_size=1,
    )
    wave = InitializerWave(loader_args)
    try:
        # Fresh initializer/processor/decoder. The loader's physical model is not used.
        torch.manual_seed(args.seed)
        core = ObservationTransfer(wave.names, "instance")
        model = LatentOceanForecast(
            core.initializer,
            core.adapter,
            stochastic=True,
            latent_width=args.latent_width,
            processor_depth=args.processor_depth,
            decoder_width=args.decoder_width,
            sampling_steps=args.sampling_steps,
        ).to(wave.device)
        del core
        model.adapter.requires_grad_(args.phase != "om4")
        if args.pretrained:
            complete = json.loads(
                (args.pretrained.parent / "PRETRAIN_COMPLETE.json").read_text()
            )
            if not complete["state"]["complete"] or complete[
                "best_checkpoint_sha256"
            ] != digest(args.pretrained):
                raise ValueError("Incomplete or changed pretraining checkpoint")
            saved = torch.load(args.pretrained, map_location="cpu", weights_only=False)
            parent = saved["protocol"]["signature"]
            if (
                any(parent[k] != v for k, v in contract.items())
                or parent["seed"] != args.seed
            ):
                raise ValueError("Pretraining contract or seed differs")
            model.load_state_dict(saved["model"], strict=True)
            del saved
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )
        wave.prepare(wave.trainset)
        wave.prepare(wave.valset)
        data = Samples(args.root / "data/observations", wave.device)
        data.use_observation_normalization()
        training, validation = data.paths("train"), data.paths("validation")
        if (len(training), len(validation)) != (243, 9):
            raise ValueError("Observation cohorts differ")

        def observation_loss(step):
            sample = data.load(training[batch_at(step, len(training), 1, args.seed)[0]])
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
                    members=2,
                )
                return forecast_observation_crps(data, predictions, sample)

        def objective(step):
            if args.phase == "observation":
                value = observation_loss(step)
                if step % 4 == 0:
                    value = value + 0.1 * native_loss(
                        model,
                        wave,
                        wave.trainset,
                        batch_at(step // 4, len(wave.trainset), 1, args.seed + 1),
                        args.seed + 1000000 + step,
                    )
                return value
            return native_loss(
                model,
                wave,
                wave.trainset,
                batch_at(step, len(wave.trainset), 1, args.seed),
                args.seed + step,
            )

        def validate(step):
            if args.phase == "observation":
                ref = reference_path
                if digest(ref) != reference_hash:
                    raise ValueError("Frozen validation reference changed")
                metrics, score = evaluate_point_metrics(
                    model, data, validation, json.loads(ref.read_text()), members=8
                )
                atomic_json(
                    dict(step=step, score=score, metrics=metrics),
                    args.output / f"validation-{step}.json",
                )
                return score
            return sum(
                float(
                    native_loss(model, wave, wave.valset, [i], args.seed + 100000 + i)
                )
                for i in wave.val_ids
            ) / len(wave.val_ids)

        if args.phase == "qualify":
            model.train()
            measurements = []
            for phase in ("om4", "observation"):
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                value = (
                    native_loss(model, wave, wave.trainset, [0], 1729)
                    if phase == "om4"
                    else observation_loss(0)
                )
                value.backward()
                norms = {
                    name: gradient_norm(module)
                    for name, module in (
                        ("encoder", model.encoder),
                        ("processor", model.processor),
                        ("decoder", model.decoder),
                    )
                }
                if not torch.isfinite(value) or not all(
                    math.isfinite(v) and v > 0 for v in norms.values()
                ):
                    raise ValueError("Nonfinite loss or missing latent-path gradients")
                optimizer.step()
                torch.cuda.synchronize()
                measurements.append(
                    dict(
                        phase=phase,
                        loss=float(value.detach()),
                        gradient_norms=norms,
                        seconds=time.monotonic() - started,
                        peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                    )
                )
                wave.emit(dict(event="latent_qualification", **measurements[-1]))
            # Exercise the actual resume path, including optimizer and RNG restore.
            budget = dict(
                max_updates=2,
                max_seconds=600,
                checkpoint_every=1,
                validate_every=2,
                emit=wave.emit,
            )

            def small_validation(step):
                return native_loss(model, wave, wave.valset, [wave.val_ids[0]], 100001)

            first = bounded_fit(
                model,
                optimizer,
                objective,
                small_validation,
                args.output / "resume",
                signature,
                max_new_updates=1,
                **budget,
            )
            assert first["step"] == 1 and not first["complete"]
            state = bounded_fit(
                model,
                optimizer,
                objective,
                small_validation,
                args.output / "resume",
                signature,
                **budget,
            )
            assert state["step"] == 2 and state["complete"]
            model.eval()
            with torch.no_grad():
                expected = float(small_validation(2))
            atomic_torch(model.state_dict(), args.output / "reload.pt")
            model.load_state_dict(
                torch.load(
                    args.output / "reload.pt",
                    map_location=wave.device,
                    weights_only=True,
                ),
                strict=True,
            )
            with torch.no_grad():
                actual = float(small_validation(2))
            if expected != actual:
                raise ValueError("Strict reload changed fixed-noise loss")
            atomic_json(
                dict(
                    qualified=True,
                    contract=contract,
                    measurements=measurements,
                    resume_state=state,
                    strict_reload_loss=actual,
                ),
                args.output / "QUALIFIED.json",
            )
            return
        state = bounded_fit(
            model,
            optimizer,
            objective,
            validate,
            args.output,
            signature,
            max_updates=args.updates,
            max_seconds=args.hours * 3600,
            max_new_seconds=args.invocation_hours * 3600
            if args.invocation_hours
            else None,
            checkpoint_every=args.checkpoint_every,
            validate_every=args.validate_every,
            emit=wave.emit,
        )
        marker = (
            "PRETRAIN_COMPLETE.json"
            if args.phase == "om4"
            else "OBSERVATION_COMPLETE.json"
        )
        atomic_json(
            dict(state=state, best_checkpoint_sha256=digest(args.output / "best.pt")),
            args.output / marker,
        )
    finally:
        if wave.run:
            wave.run.finish()


if __name__ == "__main__":
    main()
