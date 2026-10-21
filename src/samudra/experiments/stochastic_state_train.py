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


def observation_parts(model, data, original, seed, capture=None):
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
    if capture is not None:
        capture(states)
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


def native_objective(model, wave, index, coverage, seed, capture=None):
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
    if capture is not None:
        capture(states)
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


def qualify_replay(model, data, wave, intervals, output):
    """Backpropagate each calendar continuation to day 365 without ocean resets."""
    from samudra.experiments.stochastic_state_replay import (
        ReplayPool,
        replay_parts,
        window_layout,
    )

    origin = "1993-05"
    pool = ReplayPool()
    captured = []

    def fresh_capture(states):
        pool.add_next(origin, origin, states, 0)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _, loss, _ in observation_parts(
            model,
            data,
            data.load(data.root / "train" / (origin + ".npz")),
            87291,
            capture=fresh_capture,
        ):
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Fresh replay qualification")
            loss.backward()
    for depth in range(1, 12):
        entry = pool.entries[depth][-1]
        model.zero_grad(set_to_none=True)

        def capture(states):
            pool.add_next(entry["origin"], entry["month"], states, 0, origin_step=0)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = []
            for _, loss, _ in replay_parts(
                model, data, intervals, entry, 87291 + depth, capture=capture
            ):
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Long replay qualification")
                loss.backward()
                values.append(float(loss.detach()))
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        ):
            raise FloatingPointError("Nonfinite long-replay gradients")
        end = window_layout(entry["origin"], entry["month"])[1] * 5
        captured.append(
            dict(
                month=entry["month"],
                start_days=entry["bin_index"] * 5,
                end_days=end,
                loss=sum(values),
            )
        )
    from samudra.experiments.stochastic_state_native_replay import (
        OM4ReplayPool,
        native_replay_objective,
    )

    native = OM4ReplayPool(len(wave.trainset))
    coverage = data.load(data.root / "train" / (origin + ".npz"))["validity"][:, :19]

    def native_fresh_capture(states):
        native.add_next(0, 0, states, 0)

    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = native_objective(
            model, wave, 0, coverage, 91723, capture=native_fresh_capture
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Fresh OM4 qualification")
        loss.backward()
    native_windows = []
    for lead in range(6, 72, 6):
        entry = native.entries[lead][-1]
        model.zero_grad(set_to_none=True)

        def native_capture(states):
            native.add_next(0, lead, states, 0, origin_step=0)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = native_replay_objective(
                model, wave, entry, coverage, 91723 + lead, capture=native_capture
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Long OM4 replay qualification")
            loss.backward()
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
        ):
            raise FloatingPointError("Nonfinite OM4 replay gradients")
        native_windows.append(
            dict(
                start_days=lead * 5, end_days=(lead + 6) * 5, loss=float(loss.detach())
            )
        )
    atomic_json(
        dict(
            native_windows=native_windows,
            native_max_end_days=native_windows[-1]["end_days"],
            windows=captured,
            max_end_days=captured[-1]["end_days"],
            observed_ocean_resets=False,
        ),
        output / "replay-longlead.json",
    )


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
    parser.add_argument("--replay-cache", type=Path)
    parser.add_argument("--qualify-replay", action="store_true")
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
    if args.qualify_replay and (not args.smoke or not args.replay_cache):
        raise ValueError("Replay qualification requires smoke and replay cache")
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
        replay=bool(args.replay_cache),
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
    pool = intervals = None
    replay_rng = np.random.default_rng(1729001)
    replay_examples = fresh_examples = 0
    native_replay_examples = 0
    native_replay_rng = np.random.default_rng(1729002)
    native_pool = None
    if args.replay_cache:
        from samudra.experiments.stochastic_state_replay import (
            ReplayIntervals,
            ReplayPool,
        )

        if args.arm == "ar1":
            raise ValueError("Replay of AR1 requires cached innovation history")
        proof = json.loads((args.replay_cache / "COMPLETE.json").read_text())
        if proof["exact_original_months"] != 243:
            raise ValueError("Replay cache differs from original 243 training months")
        for path, sha in proof["contracts"].items():
            if digest(path) != sha:
                raise ValueError("Replay data contract changed: " + path)
        for name, sha in proof["files"].items():
            if (args.qualify_replay or name != "values.npy") and digest(
                args.replay_cache / name
            ) != sha:
                raise ValueError("Replay payload hash changed: " + name)
        if (args.replay_cache / "values.npy").stat().st_size != proof["bytes"]:
            raise ValueError("Replay cache size changed")
        pool = ReplayPool()
        intervals = ReplayIntervals(args.replay_cache)
        contract.update(
            replay_fraction_of_obs=0.5,
            replay_max_age=250,
            replay_entries_per_lead_month=8,
            replay_max_months=11,
            native_replay_fraction=0.5,
            native_replay_max_days=360,
            native_replay_chunk_days=30,
            replay_proof_sha256=digest(args.replay_cache / "COMPLETE.json"),
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
    if args.replay_cache:
        from samudra.experiments.stochastic_state_native_replay import OM4ReplayPool

        native_pool = OM4ReplayPool(len(wave.trainset))
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
        if pool is not None:
            pool.load_state_dict(saved["replay_pool"])
            replay_rng.bit_generator.state = saved["replay_rng"]
            replay_examples = saved["replay_examples"]
            fresh_examples = saved["fresh_examples"]
            assert native_pool is not None
            native_pool.load_state_dict(saved["native_replay_pool"])
            native_replay_rng.bit_generator.state = saved["native_replay_rng"]
            native_replay_examples = saved["native_replay_examples"]
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
        if pool is not None:
            assert native_pool is not None
            payload.update(
                replay_pool=pool.state_dict(),
                replay_rng=replay_rng.bit_generator.state,
                replay_examples=replay_examples,
                fresh_examples=fresh_examples,
                native_replay_pool=native_pool.state_dict(),
                native_replay_rng=native_replay_rng.bit_generator.state,
                native_replay_examples=native_replay_examples,
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
        replay_leads, replay_ages = [], []
        native_leads, native_ages = [], []
        for micro, index in enumerate(indices):
            noise_seed = seed + 1000000 + count * 8 + micro
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if task == "observation":
                    path = training[index]
                    entry = (
                        pool.draw(replay_rng, completed)
                        if pool is not None and micro % 2
                        else None
                    )

                    def capture(states):
                        if pool is not None:
                            pool.add_next(
                                entry["origin"] if entry else path.stem,
                                entry["month"] if entry else path.stem,
                                states,
                                completed,
                                origin_step=entry["origin_step"] if entry else None,
                            )

                    if entry is None:
                        parts = observation_parts(
                            model, data, data.load(path), noise_seed, capture=capture
                        )
                        fresh_examples += 1
                    else:
                        from samudra.experiments.stochastic_state_replay import (
                            replay_parts,
                        )

                        parts = replay_parts(
                            model, data, intervals, entry, noise_seed, capture=capture
                        )
                        replay_examples += 1
                        replay_leads.append(entry["bin_index"] * 5)
                        replay_ages.append(completed - entry["origin_step"])
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
                    native_entry = (
                        native_pool.draw(native_replay_rng, completed)
                        if native_pool is not None and micro % 2
                        else None
                    )

                    def native_capture(states):
                        if native_pool is not None:
                            native_pool.add_next(
                                native_entry["origin_index"]
                                if native_entry
                                else int(index),
                                native_entry["start_lead"] if native_entry else 0,
                                states,
                                completed,
                                origin_step=native_entry["origin_step"]
                                if native_entry
                                else None,
                            )

                    if native_entry is None:
                        loss = native_objective(
                            model,
                            wave,
                            index,
                            coverage,
                            noise_seed,
                            capture=native_capture,
                        )
                    else:
                        from samudra.experiments.stochastic_state_native_replay import (
                            native_replay_objective,
                        )

                        loss = native_replay_objective(
                            model,
                            wave,
                            native_entry,
                            coverage,
                            noise_seed,
                            capture=native_capture,
                        )
                        native_replay_examples += 1
                        native_leads.append(native_entry["start_lead"] * 5)
                        native_ages.append(completed - native_entry["origin_step"])
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
            replay_examples=replay_examples,
            fresh_examples=fresh_examples,
            replay_lead_days=replay_leads,
            replay_age_updates=replay_ages,
            native_replay_examples=native_replay_examples,
            native_replay_lead_days=native_leads,
            native_replay_age_updates=native_ages,
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
    if args.qualify_replay:
        qualify_replay(model, data, wave, intervals, args.output)
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
