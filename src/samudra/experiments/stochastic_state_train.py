# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched short pilots with stochastic member states and deterministic decoding.

One GPU, effective batch eight, alternating OM4/observation updates. No diffusion
objective or independent physical-field sampling is used by these experiments.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from samudra.experiments.diffusion_spatial import (
    completion_crps,
    forecast_crps,
    interior_crps,
    spatial_crps,
)
from samudra.experiments.diffusion_structure_losses import excess_latent_loss
from samudra.experiments.initializer_wave import InitializerWave
from samudra.experiments.missingness import corrupt_sample, structured_visibility
from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_state import advance_season
from samudra.experiments.task_schedule import sample_indices

ARMS = ("deterministic", "initial", "fresh", "ar1")


def counts(step):
    """OM4 is first; every completed pair exposes each task once."""
    return {"om4": (step + 1) // 2, "observation": step // 2}


def learning_rate(step, total, base=1e-4, minimum=1e-6):
    start = int(total * 0.8)
    if step < start:
        return base
    fraction = (step - start) / max(1, total - start - 1)
    return minimum + (base - minimum) * (1 + math.cos(math.pi * fraction)) / 2


def rng(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def make_model(names, device, arm, decoder_width=64):
    from samudra.experiments.missingness import identity_adapters
    from samudra.experiments.observation_model import ObservationTransfer
    from samudra.experiments.stochastic_state import StochasticStateForecast

    core = ObservationTransfer(names, "instance")
    model = StochasticStateForecast(
        core.initializer,
        core.adapter,
        arm=arm,
        latent_width=128,
        processor_depth=4,
        decoder_width=decoder_width,
    )
    model.input_adapters = identity_adapters(19 * 7 + 5)
    model.processor.input_adapters = identity_adapters(2 * 128 + 8)
    return model.to(device)


def model_from_checkpoint(saved, data):
    contract = saved["contract"]
    model = make_model(
        data.grid["names"].tolist(),
        data.device,
        contract["arm"],
        contract["decoder_width"],
    )
    model.load_state_dict(saved["model"], strict=True)
    return model


def coherent_reconstruction(model, data, sample, generator, members=2):
    """Use first target-bin surface then evolve the rest of the same month.

    Unlike the legacy independently reinitialized monthly reconstruction, a
    member's initial draw and state persist across all bins. Only the first
    current-month surface bin enters this auxiliary task; later surface values
    and all interior values are labels, never encoder inputs.
    """
    adapted = model.adapt(sample["atmosphere"])
    history = model.history
    initial, known, anchor = model.encode_native(
        sample["surface"][:, 1 : history + 1],
        adapted[:, 1 : history + 1],
        sample["contexts"][:, history],
        data.mask,
        sample["validity"][:, 1 : history + 1],
        task="observation",
    )
    states = model.rollout_members(
        initial,
        adapted[:, history + 1 :],
        sample["contexts"][:, history + 1 :],
        generator=generator,
        members=members,
        task="observation",
    )
    weights = sample["month_weights"]
    if len(states) != len(weights):
        raise ValueError("Reconstruction does not cover exactly the target month")
    monthly = None
    for i, state in enumerate(states):
        fields = model.decode_members(
            state, data.mask, known if i == 0 else None, anchor if i == 0 else None
        )[:, :, -1]
        term = fields * weights[i]
        monthly = term if monthly is None else monthly + term
    return monthly, excess_latent_loss(states, states[0])


def observation_parts(model, data, original, seed):
    """Separate graphs permit forecast and reconstruction backprop independently."""
    sample = corrupt_sample(original, seed)
    adapted = model.adapt(sample["atmosphere"])
    latent, known, anchor = model.encode_native(
        sample["surface"][:, :19],
        adapted[:, :19],
        sample["contexts"][:, 18],
        data.mask,
        sample["validity"][:, :19],
        task="observation",
    )
    states = model.rollout_members(
        latent,
        adapted[:, 19:],
        sample["contexts"][:, 19:],
        generator=rng(data.device, seed + 2000000),
        members=2,
        task="observation",
    )
    initial = model.decode_members(states[0], data.mask, known, anchor)
    predictions = torch.stack(
        [model.decode_members(state, data.mask)[:, :, -1] for state in states[1:]], 2
    )
    bound = excess_latent_loss(states, states[0])
    forecast = forecast_crps(data, predictions, sample)
    completion = completion_crps(
        data,
        initial[:, :, :, model.surface],
        sample["completion_target"],
        sample["completion_valid"],
    )
    yield (
        "forecast_completion",
        forecast + 0.1 * completion + 0.1 * bound,
        {
            "forecast": float(forecast.detach()),
            "completion": float(completion.detach()),
            "bound": float(bound.detach()),
        },
    )
    del predictions, initial, forecast, completion, states, latent, bound, adapted
    monthly, bound = coherent_reconstruction(
        model, data, sample, rng(data.device, seed + 3000000)
    )
    reconstruction = interior_crps(data, monthly, sample)
    yield (
        "reconstruction",
        0.1 * reconstruction + 0.1 * bound,
        {
            "reconstruction": float(reconstruction.detach()),
        },
    )


def native_objective(model, wave, index, coverage, seed):
    """Physical normalized channel CRPS, with every member evolved separately."""
    surface, past, context, truth, forcing, targets = wave.model_sample(
        wave.trainset, [index]
    )
    if forcing.shape[1] != 6:
        raise ValueError(
            "Pilot OM4 training requires exactly six five-day forecast steps"
        )
    available = wave.mask[model.surface].bool().expand_as(surface)
    visible = structured_visibility(available & coverage.bool(), seed)
    initial, known, anchor = model.encode_native(
        torch.where(visible, surface, 0), past, context, wave.mask, visible, task="om4"
    )
    contexts = torch.stack(
        [advance_season(context, (i + 1) * 5) for i in range(forcing.shape[1])], 1
    )
    states = model.rollout_members(
        initial,
        forcing,
        contexts,
        generator=rng(wave.device, seed + 4000000),
        members=2,
        task="om4",
    )
    decoded_initial = model.decode_members(states[0], wave.mask, known, anchor)

    def score(prediction, target, support):
        values, present = spatial_crps(prediction, target, support, wave.weights, 1.0)
        return values[present].mean()

    interior_mask = wave.mask.clone()
    interior_mask[model.surface] = 0
    reconstruction = score(decoded_initial, truth, interior_mask)
    losses = [
        score(
            model.decode_members(state, wave.mask)[:, :, -1], targets[:, i], wave.mask
        )
        for i, state in enumerate(states[1:])
    ]
    loss = (
        torch.stack(losses).mean()
        + 0.1 * reconstruction
        + 0.1 * excess_latent_loss(states, states[0])
    )
    hidden = available[:, -2:] & ~visible[:, -2:]
    if bool(hidden.any()):
        values, present = spatial_crps(
            decoded_initial[:, :, :, model.surface],
            truth[:, :, model.surface],
            hidden,
            wave.weights[model.surface],
            1.0,
        )
        loss = loss + 0.1 * values[present].mean()
    return loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--updates-per-task", type=int, default=1000)
    parser.add_argument("--decoder-width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-new-updates", type=int)
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--compile-decoder", action="store_true")
    parser.add_argument("--wandb-project", default="default")
    parser.add_argument("--wandb-entity", default="ocean_emulators")
    parser.add_argument(
        "--wandb-mode", choices=("online", "disabled"), default="online"
    )
    args = parser.parse_args()
    if args.smoke:
        args.wandb_mode = "disabled"
    elif args.wandb_mode != "online":
        raise ValueError("W&B online required outside smoke")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("Pilot trainer is single GPU, not DDP")
    if args.updates_per_task < 1 or args.checkpoint_every < 1:
        raise ValueError("Positive update counts required")
    device = torch.device("cuda")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    data = Samples(
        args.root / "data/observations",
        device,
        surface_fill="zero",
        global_observations=True,
    )
    data.use_observation_normalization()
    training, validation = data.paths("train"), data.paths("validation")
    if (len(training), len(validation), len(data.paths("test"))) != (243, 9, 96):
        raise ValueError("Observation cohorts changed")
    if not bool((data.area > 0).all()):
        raise ValueError("Full-latitude area weighting required")
    contract = dict(
        version=1,
        arm=args.arm,
        seed=args.seed,
        effective_batch=8,
        training_members=2,
        updates_per_task=args.updates_per_task,
        schedule="strict alternating OM4 then observation",
        decoder_width=args.decoder_width,
        latent_width=128,
        processor_depth=4,
        native_loss="channel-balanced physical normalized fair CRPS + 0.5 spatial-increment CRPS",
        observation_loss="legacy group-balanced fair CRPS + 0.5 spatial-increment CRPS",
        forecast_weight=1.0,
        reconstruction_weight=0.1,
        completion_weight=0.1,
        reconstruction="one coherent trajectory initialized at first target-month surface bin",
        global_observations=True,
        surface_fill="zero",
        replay=False,
        latent_bound=0.1,
        deterministic_decoder=True,
        noise_scale=0.05,
        noise_channels=8,
        noise_shape=[6, 12],
        ar1_rho=0.9,
        learning_rate=1e-4,
        minimum_lr=1e-6,
        cooldown_fraction=0.2,
        weight_decay=0.01,
        compile_decoder=args.compile_decoder,
        observation_manifest=digest(data.root / "SHA256SUMS"),
        grid=digest(data.root / "grid.npz"),
        stats=digest(data.root / "statistics.npz"),
        reference=digest(args.reference),
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
    )
    protocol = args.output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != contract:
        raise ValueError("Output protocol differs")
    atomic_json(contract, protocol)
    wave = InitializerWave(
        SimpleNamespace(
            arm="D",
            phase="reconstruction",
            seed=args.seed,
            readers=4,
            data_root=str(args.root / "data/om4_onedeg_v3"),
            output=str(args.output / "native-loader"),
            name="stochastic-state-loader",
            wandb_mode="disabled",
            normalization="instance",
            observation_normalization_root=str(data.root),
            fresh_evolution=True,
            initial_checkpoint="",
            wave1_root="",
            val_origins=12,
            device_cache=False,
            cache_device="cpu",
            device_cache_reserve_gib=32,
            batch_size=1,
        )
    )
    del wave.model
    if wave.run is not None:
        wave.run.finish()
        wave.run = None
    gc.collect()
    wave.prepare(wave.trainset)
    if list(wave.names) != list(data.grid["names"]):
        raise ValueError("Native channel order differs")
    # Loader setup may consume RNG; matched model initialization begins here.
    torch.manual_seed(args.seed)
    model = make_model(
        data.grid["names"].tolist(), device, args.arm, args.decoder_width
    )
    wave.initializer = SimpleNamespace(surface=model.surface)
    gc.collect()
    torch.cuda.empty_cache()
    forcing_stats = wave.source.statistics(wave.bundle.data_layout.boundary_var_names)

    def array_digest(value):
        return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()

    native_contract = dict(
        config=wave.config.model_dump(mode="json"),
        channels=list(wave.names),
        native_mean_sha256=array_digest(wave.native_mean.cpu()),
        native_std_sha256=array_digest(wave.native_std.cpu()),
        forcing_mean_sha256=array_digest(forcing_stats.mean),
        forcing_std_sha256=array_digest(forcing_stats.std),
        converted_mean_sha256=array_digest(wave.mean.cpu()),
        converted_std_sha256=array_digest(wave.std.cpu()),
        train_windows=len(wave.trainset),
        source_root=str(args.root / "data/om4_onedeg_v3"),
    )
    native_path = args.output / "native-data.json"
    if native_path.exists() and json.loads(native_path.read_text()) != native_contract:
        raise ValueError("Native data/config/statistics changed on resume")
    atomic_json(native_contract, native_path)
    if args.compile_decoder:
        model.decoder.compile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    last = args.output / "last.pt"
    completed = 0
    run_id = None
    if last.exists() != args.resume:
        raise ValueError(
            "Existing checkpoint requires --resume; --resume requires checkpoint"
        )
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["contract"] != contract or saved["counts"] != counts(saved["step"]):
            raise ValueError("Resume contract or exposure differs")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        completed = saved["step"]
        run_id = saved["wandb_id"]
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        del saved
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"stochastic-state-{args.arm}",
        config=contract,
        mode=args.wandb_mode,
        id=run_id if args.wandb_mode == "online" else None,
        resume="must" if run_id and args.wandb_mode == "online" else None,
        dir=str(args.output),
    )
    if args.wandb_mode == "online" and not run.url:
        raise RuntimeError("Online W&B run URL required before training")
    atomic_json(
        dict(id=run.id, url=run.url, mode=args.wandb_mode), args.output / "wandb.json"
    )
    total = 2 * args.updates_per_task
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)

    def save(snapshot=False):
        payload = dict(
            contract=contract,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            step=completed,
            counts=counts(completed),
            wandb_id=run.id,
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
        )
        atomic_torch(payload, last)
        if snapshot:
            atomic_torch(payload, args.output / f"step-{completed:05d}.pt")
        atomic_json(
            dict(step=completed, counts=counts(completed), complete=completed == total),
            args.output / "state.json",
        )

    if not last.exists():
        save(snapshot=True)
    start, first = time.monotonic(), completed
    while completed < total:
        started = time.monotonic()
        task = "om4" if completed % 2 == 0 else "observation"
        count = counts(completed)[task]
        seed = args.seed + (100000 if task == "om4" else 0)
        indices = sample_indices(
            len(training) if task == "observation" else len(wave.trainset),
            seed,
            count,
            8,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(completed, total)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        measured: dict[str, float] = {}
        value = 0.0
        for micro, index in enumerate(indices):
            noise_seed = seed + 1000000 + count * 8 + micro
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if task == "observation":
                    parts = observation_parts(
                        model, data, data.load(training[index]), noise_seed
                    )
                    for label, loss, detail in parts:
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(label)
                        (loss / 8).backward()
                        value += float(loss.detach()) / 8
                        for key, number in detail.items():
                            measured[key] = measured.get(key, 0) + number / 8
                else:
                    coverage = data.load(training[(count * 8 + micro) % len(training)])[
                        "validity"
                    ][:, :19]
                    loss = native_objective(model, wave, index, coverage, noise_seed)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("native loss")
                    (loss / 8).backward()
                    value += float(loss.detach()) / 8
        gradients = {
            name: math.sqrt(
                sum(
                    float(p.grad.detach().float().square().sum())
                    for p in module.parameters()
                    if p.grad is not None
                )
            )
            for name, module in (
                ("encoder", model.encoder),
                ("processor", model.processor),
                ("decoder", model.decoder),
                ("initial_noise", model.initial_noise),
                ("transition_noise", model.transition_noise),
            )
        }
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        completed += 1
        event = dict(
            event="update",
            step=completed,
            task=task,
            counts=counts(completed),
            loss=value,
            parts=measured,
            gradient_norm=float(norm),
            gradient_norms=gradients,
            seconds=time.monotonic() - started,
            learning_rate=optimizer.param_groups[0]["lr"],
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        print(json.dumps(event), flush=True)
        with (args.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        run.log(
            {
                f"train/{task}/loss": value,
                "train/seconds": event["seconds"],
                "train/learning_rate": event["learning_rate"],
                "exposure/observation": counts(completed)["observation"],
                "exposure/om4": counts(completed)["om4"],
                **{f"train/{task}/{key}": number for key, number in measured.items()},
            },
            step=completed,
        )
        milestone = completed % 2 == 0 and completed // 2 in (
            125,
            250,
            500,
            1000,
            args.updates_per_task,
        )
        due = stop or time.monotonic() - start >= args.hours * 3600
        due |= (
            args.max_new_updates is not None
            and completed - first >= args.max_new_updates
        )
        if (
            due
            or milestone
            or completed % args.checkpoint_every == 0
            or completed == total
        ):
            save(snapshot=milestone or completed == total)
        if due:
            break
    run.finish()
    print(
        json.dumps(
            dict(
                event="invocation_finished",
                step=completed,
                counts=counts(completed),
                complete=completed == total,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
