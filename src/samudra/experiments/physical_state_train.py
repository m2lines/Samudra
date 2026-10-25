# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU physical-state diffusion pilot. No jobs are submitted by this CLI."""

import argparse
import copy
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch

from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from samudra.experiments.task_schedule import sample_indices


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path)
    result.add_argument(
        "--data",
        type=Path,
        help="Observation samples; defaults to ROOT/data/observations",
    )
    result.add_argument("--output", type=Path)
    result.add_argument("--reference", type=Path)
    result.add_argument("--describe", action="store_true")
    result.add_argument(
        "--forcing-convention",
        choices=("shared-era5",),
        default="shared-era5",
        help="The same eight normalized ERA5 channels condition both tasks directly.",
    )
    result.add_argument(
        "--om4",
        type=Path,
        help="Directory containing OM4.zarr and means/stds; defaults to ROOT/data/om4_onedeg_v3",
    )
    result.add_argument(
        "--packed-om4", type=Path, help="Optional exactly verified frame pack"
    )
    result.add_argument("--shared-era5", type=Path, help="Verified interval cache")
    result.add_argument(
        "--shared-era5-proof-sha256", help="Pinned COMPLETE.json SHA256"
    )
    result.add_argument("--om4-warmup-updates", type=int, default=100)
    result.add_argument("--obs-cooldown-updates", type=int, default=100)
    result.add_argument("--ema-half-life", type=float, default=100)
    result.add_argument("--updates-per-task", type=int, default=500)
    result.add_argument("--sampling-steps", type=int, default=8)
    result.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    result.add_argument("--noise-correlation", type=float, default=0.5)
    result.add_argument("--microbatch", type=int, default=1)
    result.add_argument("--no-checkpoint-denoiser", action="store_true")
    result.add_argument("--compile", action="store_true")
    result.add_argument("--fused-adam", action="store_true")
    result.add_argument("--replay-cache", type=Path)
    result.add_argument("--replay-per-lead", type=int, default=2)
    result.add_argument("--effective-batch", type=int, default=8)
    result.add_argument("--members", type=int, default=2)
    result.add_argument("--seed", type=int, default=1729)
    result.add_argument("--checkpoint-every", type=int, default=50)
    result.add_argument("--max-new-updates", type=int)
    result.add_argument("--hours", type=float, default=8)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--wandb-project", default="default")
    result.add_argument("--wandb-entity", default="ocean_emulators")
    return result


def validate_args(args):
    for name in (
        "updates_per_task",
        "effective_batch",
        "checkpoint_every",
        "sampling_steps",
        "microbatch",
        "replay_per_lead",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if (
        args.om4_warmup_updates < 0
        or args.obs_cooldown_updates < 0
        or args.ema_half_life <= 0
    ):
        raise ValueError(
            "Nonnegative phase lengths and positive EMA half-life required"
        )
    if args.sampling_steps < 2:
        raise ValueError("Heun sampling requires at least two noise levels")
    if args.members < 2 or args.hours <= 0 or any(width < 1 for width in args.widths):
        raise ValueError("Require at least two members, positive widths and runtime")
    if not 0 <= args.noise_correlation <= 1:
        raise ValueError("Noise correlation must be between zero and one")
    if args.max_new_updates is not None and args.max_new_updates < 1:
        raise ValueError("max-new-updates must be positive")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise ValueError("This pilot is single GPU; DDP is not implemented")
    if (
        args.forcing_convention == "shared-era5"
        and not args.describe
        and (args.shared_era5 is None or args.shared_era5_proof_sha256 is None)
    ):
        raise ValueError(
            "Shared ERA5 requires --shared-era5 and --shared-era5-proof-sha256"
        )
    if not args.describe and any(
        getattr(args, name) is None for name in ("root", "output", "reference")
    ):
        raise ValueError("Training requires --root, --output and --reference")


def configuration(args):
    return dict(
        version=4,
        architecture="samudra-physical-diffusion",
        widths=args.widths,
        sampling_steps=args.sampling_steps,
        noise_correlation=args.noise_correlation,
        forcing_convention=args.forcing_convention,
        updates_per_task=args.updates_per_task,
        om4_warmup_updates=args.om4_warmup_updates,
        obs_cooldown_updates=args.obs_cooldown_updates,
        ema_half_life=args.ema_half_life,
        ema_reset_at_cooldown=True,
        adam_preserved_at_phase_boundaries=True,
        forcing_channels=8,
        forcing_adapter=False,
        effective_batch=args.effective_batch,
        microbatch=args.microbatch,
        checkpoint_denoiser=not args.no_checkpoint_denoiser,
        fused_adam=args.fused_adam,
        training_members=args.members,
        seed=args.seed,
        precision="bf16 autocast with fp32 parameters",
        schedule="OM4-only warmup; alternating OM4/observation; observation-only cooldown",
        learning_rate=1e-4,
        minimum_lr=1e-6,
        weight_decay=0.01,
        replay=bool(args.replay_cache),
        replay_fraction=0.5 if args.replay_cache else 0.0,
        replay_sources=["observation", "om4"] if args.replay_cache else [],
        replay_per_lead=args.replay_per_lead,
        replay_max_age=250,
        replay_max_months=11,
        native_fresh_cache_rollouts_per_update=1,
        latent_bound=False,
        global_observations=True,
        data_reader_origin_commit="4150c5b",
        data_reader_origin="monthly_model_data.ModelMonths and shared_era5.SharedEra5; calendar helper extracted unchanged",
        physical_normalization="fixed OM4 grid mean/std for both tasks",
        om4_window_policy="409 contiguous calendar months; history >=1975; target <=2013-07",
        optimizer="AdamW",
        compiled=args.compile,
    )


def total_updates(contract):
    return (
        contract["om4_warmup_updates"]
        + 2 * contract["updates_per_task"]
        + contract["obs_cooldown_updates"]
    )


def counts(step, contract):
    if not 0 <= step <= total_updates(contract):
        raise ValueError("Completed updates outside schedule")
    warm = min(step, contract["om4_warmup_updates"])
    mixed = min(max(0, step - warm), 2 * contract["updates_per_task"])
    cool = step - warm - mixed
    return dict(om4=warm + (mixed + 1) // 2, observation=mixed // 2 + cool)


def phase_settings(step, contract):
    """Zero-based next update; cooldown matches monthly-thread linear schedule."""
    if not 0 <= step < total_updates(contract):
        raise ValueError("Next update outside schedule")
    warm, mixed = contract["om4_warmup_updates"], 2 * contract["updates_per_task"]
    base, minimum = contract["learning_rate"], contract["minimum_lr"]
    if step < warm:
        return "om4_warmup", "om4", base
    if step < warm + mixed:
        return "mixed", "om4" if (step - warm) % 2 == 0 else "observation", base
    fraction = (step - warm - mixed + 1) / contract["obs_cooldown_updates"]
    return "observation_cooldown", "observation", base + (minimum - base) * fraction


@torch.no_grad()
def update_ema(ema, model, half_life):
    for target, source in zip(ema.parameters(), model.parameters(), strict=True):
        target.lerp_(source, 1 - 2 ** (-1 / half_life))
    for target, source in zip(ema.buffers(), model.buffers(), strict=True):
        target.copy_(source)


def make_model(names, device, config):
    from samudra.experiments.samudra_physical_diffusion import SamudraPhysicalDiffusion

    initializer = SimpleNamespace(
        channels=len(names),
        history=19,
        surface=[names.index("thetao_0"), names.index("zos")],
    )
    return SamudraPhysicalDiffusion(
        initializer,
        widths=tuple(config["widths"]),
        sampling_steps=config["sampling_steps"],
        noise_correlation=config["noise_correlation"],
        checkpoint_denoiser=config.get("checkpoint_denoiser", True),
        vectorize_members=True,
    ).to(device)


def model_from_checkpoint(saved, data, *, use_ema=True):
    model = make_model(data.grid["names"].tolist(), data.device, saved["contract"])
    model.load_state_dict(saved["ema"] if use_ema else saved["model"], strict=True)
    return model


def checkpoint_payload(model, optimizer, contract, completed, run_id, ema, replay=None):
    return dict(
        contract=contract,
        model=model.state_dict(),
        ema=ema.state_dict(),
        optimizer=optimizer.state_dict(),
        step=completed,
        counts=counts(completed, contract),
        replay=replay,
        wandb_id=run_id,
        cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
    )


def restore_checkpoint(saved, model, optimizer, contract, ema):
    if saved["contract"] != contract or saved["counts"] != counts(
        saved["step"], contract
    ):
        raise ValueError("Resume protocol or task exposure changed")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    ema.load_state_dict(saved["ema"], strict=True)
    torch.set_rng_state(saved["cpu_rng"])
    if saved["cuda_rng"] is not None:
        torch.cuda.set_rng_state(saved["cuda_rng"])
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    return saved["step"], saved["wandb_id"]


def main():
    args = parser().parse_args()
    validate_args(args)
    contract = configuration(args)
    if args.describe:
        names = [
            f"{variable}_{level}"
            for variable in ("uo", "vo", "thetao", "so")
            for level in range(19)
        ] + ["zos"]
        model = make_model(names, torch.device("cpu"), contract)
        print(
            json.dumps(
                dict(contract, parameters=sum(p.numel() for p in model.parameters())),
                indent=2,
            )
        )
        return
    from samudra.experiments.monthly_model_data import ModelMonths
    from samudra.experiments.observation_training import Samples
    from samudra.experiments.physical_state_batching import (
        collate_examples,
        compatible_groups,
    )
    from samudra.experiments.physical_state_objectives import (
        native_objective,
        observation_batch_objective,
    )
    from samudra.experiments.physical_state_replay import (
        PhysicalReplayPool,
        ReplayIntervals,
        native_replay_objective,
        observation_replay_objective,
        sample_native_trajectory,
        states_from_predictions,
    )
    from samudra.experiments.shared_era5 import SharedEra5

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; --describe is CPU-only")
    device = torch.device("cuda")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    data = Samples(
        args.data or args.root / "data/observations",
        device,
        surface_fill="zero",
        global_observations=True,
    )
    training = data.paths("train")
    if (len(training), len(data.paths("validation")), len(data.paths("test"))) != (
        243,
        9,
        96,
    ):
        raise ValueError("Observation cohorts changed")
    if not bool((data.area > 0).all()):
        raise ValueError("Full-latitude area weighting required")
    contract.update(
        observation_manifest=digest(data.root / "SHA256SUMS"),
        grid=digest(data.root / "grid.npz"),
        stats=digest(data.root / "statistics.npz"),
        reference=digest(args.reference),
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
    )
    om4_path = args.om4 or args.root / "data/om4_onedeg_v3"
    reader = ModelMonths(
        om4_path / "OM4.zarr", "om4", data.grid, om4_path, packed=args.packed_om4
    )
    if len(reader.windows) != 409:
        raise ValueError("Expected exact latent-thread 409 OM4 calendar windows")
    forcing_cache = None
    if args.forcing_convention == "shared-era5":
        forcing_cache = SharedEra5(
            args.shared_era5,
            reader,
            data,
            args.shared_era5_proof_sha256,
            full_readback=True,
        )

    assert forcing_cache is not None

    def array_digest(value):
        return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()

    windows = [
        dict(
            start=int(start),
            stop=int(stop),
            weights=[float(w) for w in weights],
            duration=int(duration),
            history_start=str(reader.bounds[start - 19, 0]),
            target_start=str(reader.bounds[start, 0]),
            target_end=str(reader.bounds[stop - 1, 1]),
        )
        for start, stop, weights, duration in reader.windows
    ]
    native_contract = dict(
        channels=reader.names,
        window_count=len(windows),
        windows=windows,
        excluded_gap_months=reader.excluded_gap_months,
        physical_mean_sha256=array_digest(data.grid["mean"]),
        physical_std_sha256=array_digest(data.grid["std"]),
        forcing_mean_sha256=array_digest(reader.forcing_mean),
        forcing_std_sha256=array_digest(reader.forcing_std),
        source_root=str(om4_path),
        source_metadata_sha256=digest(om4_path / "OM4.zarr/.zmetadata"),
        means_metadata_sha256=digest(om4_path / "OM4_means.zarr/.zmetadata"),
        stds_metadata_sha256=digest(om4_path / "OM4_stds.zarr/.zmetadata"),
        packed_proof_sha256=digest(args.packed_om4 / "COMPLETE.json")
        if args.packed_om4
        else None,
        shared_era5_root=str(args.shared_era5) if forcing_cache is not None else None,
        shared_era5_proof_sha256=args.shared_era5_proof_sha256,
    )
    native_path = args.output / "native-data.json"
    if native_path.exists() and json.loads(native_path.read_text()) != native_contract:
        raise ValueError("Native data/config/statistics changed")
    atomic_json(native_contract, native_path)
    contract["native_data_sha256"] = digest(native_path)
    pools = None
    intervals = None
    replay_counts = {"om4": 0, "observation": 0}
    fresh_counts = {"om4": 0, "observation": 0}
    if args.replay_cache:
        proof = json.loads((args.replay_cache / "COMPLETE.json").read_text())
        if proof["exact_original_months"] != 243:
            raise ValueError("Observation replay cache changed training months")
        for path, expected in proof["contracts"].items():
            if digest(path) != expected:
                raise ValueError("Observation replay data contract changed")
        for name, expected in proof["files"].items():
            if digest(args.replay_cache / name) != expected:
                raise ValueError("Observation replay payload checksum changed")
        contract["replay_proof_sha256"] = digest(args.replay_cache / "COMPLETE.json")
        intervals = ReplayIntervals(args.replay_cache)
        pools = {
            source: PhysicalReplayPool(
                source, per_lead=args.replay_per_lead, seed=args.seed + 90000 + i
            )
            for i, source in enumerate(("om4", "observation"))
        }
    protocol = args.output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != contract:
        raise ValueError("Output protocol differs")
    atomic_json(contract, protocol)
    torch.manual_seed(args.seed)
    model = make_model(data.grid["names"].tolist(), device, contract)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if args.compile:
        model.enable_compile()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.01, fused=args.fused_adam
    )
    last = args.output / "last.pt"
    if last.exists() != args.resume:
        raise ValueError("Existing checkpoint requires --resume and vice versa")
    completed, run_id = 0, None
    if args.resume:
        saved = torch.load(last, map_location="cpu", weights_only=False)
        completed, run_id = restore_checkpoint(
            saved,
            model,
            optimizer,
            contract,
            ema,
        )
        if pools is not None:
            for source, pool in pools.items():
                pool.load_state_dict(saved["replay"]["pools"][source])
            replay_counts = saved["replay"]["replay_counts"]
            fresh_counts = saved["replay"]["fresh_counts"]
        del saved
    import wandb

    mode: Literal["disabled", "online"] = "disabled" if args.smoke else "online"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="physical-state-diffusion",
        config=contract,
        mode=mode,
        id=run_id if mode == "online" else None,
        resume="must" if run_id and mode == "online" else None,
        dir=str(args.output),
    )
    if mode == "online" and not run.url:
        raise RuntimeError("Online W&B URL required")
    atomic_json(dict(id=run.id, url=run.url, mode=mode), args.output / "wandb.json")
    total, stop = total_updates(contract), False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)

    def save(snapshot=False):
        replay_state = (
            None
            if pools is None
            else dict(
                pools={source: pool.state_dict() for source, pool in pools.items()},
                replay_counts=replay_counts,
                fresh_counts=fresh_counts,
            )
        )
        payload = checkpoint_payload(
            model, optimizer, contract, completed, run.id, ema, replay_state
        )
        atomic_torch(payload, last)
        if snapshot:
            atomic_torch(payload, args.output / f"step-{completed:05d}.pt")
        atomic_json(
            dict(
                step=completed,
                counts=counts(completed, contract),
                complete=completed == total,
            ),
            args.output / "state.json",
        )

    if not args.resume:
        save(snapshot=True)
    start, first = time.monotonic(), completed
    while completed < total and not stop:
        started = time.monotonic()
        phase, task, rate = phase_settings(completed, contract)
        if completed == args.om4_warmup_updates + 2 * args.updates_per_task:
            ema.load_state_dict(model.state_dict())
        count = counts(completed, contract)[task]
        seed = args.seed + (100000 if task == "om4" else 0)
        indices = sample_indices(
            len(training) if task == "observation" else len(reader.windows),
            seed,
            count,
            args.effective_batch,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        model.train()
        value = 0.0
        replay_leads, replay_ages = [], []
        selected = [
            pools[task].draw(completed) if pools is not None and micro % 2 else None
            for micro in range(len(indices))
        ]
        fresh_slots = [micro for micro, entry in enumerate(selected) if entry is None]
        fresh_samples = []
        for micro in fresh_slots:
            index = int(indices[micro])
            if task == "observation":
                fresh_samples.append(data.load(training[index]))
            else:
                batch = reader.sample(index, device)
                batch["atmosphere"] = forcing_cache.load(reader, index, data)
                fresh_samples.append(batch)
        native_cached = False
        for group in compatible_groups(fresh_samples, args.microbatch):
            originals = [fresh_samples[i] for i in group]
            slots = [fresh_slots[i] for i in group]
            seeds = [
                seed + 1000000 + count * args.effective_batch + slot for slot in slots
            ]
            generator = torch.Generator(device=device).manual_seed(seeds[0])

            def capture(prediction, initial):
                if pools is not None:
                    for offset, slot in enumerate(slots):
                        month = training[int(indices[slot])].stem
                        states = states_from_predictions(
                            initial[:, offset : offset + 1],
                            prediction[:, offset : offset + 1],
                        )
                        pools[task].add_observation(month, month, states, completed)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = (
                    observation_batch_objective(
                        model,
                        data,
                        originals,
                        generator,
                        members=args.members,
                        capture=capture,
                        seeds=seeds,
                    )
                    if task == "observation"
                    else native_objective(
                        model, data, collate_examples(originals), generator
                    )
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite fresh {task} loss")
            (loss * len(group) / args.effective_batch).backward()
            value += float(loss.detach()) * len(group) / args.effective_batch
            fresh_counts[task] += len(group)
            del loss
            if pools is not None and task == "om4" and not native_cached:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    states = sample_native_trajectory(
                        model, originals[0], generator, members=args.members
                    )
                pools[task].add_native(
                    reader, int(indices[slots[0]]), states, completed
                )
                native_cached = True
                del states
        for slot, entry in enumerate(selected):
            if entry is None:
                continue
            assert pools is not None and intervals is not None
            generator = torch.Generator(device=device).manual_seed(
                seed + 1000000 + count * args.effective_batch + slot
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if task == "observation":
                    loss, states = observation_replay_objective(
                        model, data, intervals, entry, generator
                    )
                    pools[task].add_observation(
                        entry["origin"],
                        entry["month"],
                        states,
                        completed,
                        origin_step=entry["origin_step"],
                    )
                    lead = entry["bin_index"] * 5
                else:
                    loss, states = native_replay_objective(
                        model, data, reader, forcing_cache, entry, generator
                    )
                    pools[task].add_native(
                        reader,
                        entry["window_index"],
                        states,
                        completed,
                        origin_index=entry["origin_index"],
                        origin_step=entry["origin_step"],
                    )
                    lead = entry["lead_days"]
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite replay {task} loss")
            (loss / args.effective_batch).backward()
            value += float(loss.detach()) / args.effective_batch
            replay_counts[task] += 1
            replay_leads.append(int(lead))
            replay_ages.append(int(completed - entry["origin_step"]))
            del loss, states
        gradient_norms = {}
        for name, module in (
            ("initializer", model.initializer),
            ("processor", model.processor),
        ):
            terms = [
                parameter.grad.detach().float().square().sum()
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            gradient_norms[name] = (
                float(torch.stack(terms).sum().sqrt()) if terms else 0.0
            )
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        update_ema(ema, model, args.ema_half_life)
        torch.cuda.synchronize()
        completed += 1
        event = dict(
            step=completed,
            task=task,
            phase=phase,
            replay_examples=dict(replay_counts),
            fresh_examples=dict(fresh_counts),
            replay_lead_days=replay_leads,
            replay_age_updates=replay_ages,
            counts=counts(completed, contract),
            loss=value,
            gradient_norm=float(norm),
            gradient_norms=gradient_norms,
            seconds=time.monotonic() - started,
            learning_rate=optimizer.param_groups[0]["lr"],
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
        )
        with (args.output / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        run.log(event, step=completed)
        print(json.dumps(event), flush=True)
        if completed % args.checkpoint_every == 0 or completed == total:
            save(snapshot=True)
        if time.monotonic() - start >= args.hours * 3600 or (
            args.max_new_updates is not None
            and completed - first >= args.max_new_updates
        ):
            break
    save(snapshot=True)
    run.finish()


if __name__ == "__main__":
    main()
