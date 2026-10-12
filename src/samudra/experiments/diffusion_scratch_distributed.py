# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fresh direct/residual training with exact batch-eight distributed gradients.

Run under torchrun. The replay buffer is sharded by logical microbatch, not by
whatever gradients happen to be present on a rank. Checkpoints include all shards
and RNGs; reducing the union of used parameters keeps AdamW semantics identical to
serial accumulation, including when replay does not exercise the initializer.
"""

import argparse
import datetime
import gc
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
import torch.distributed as dist

from samudra.experiments.diffusion_correlated_pilot import build_wave
from samudra.experiments.diffusion_direct import install_direct_processor
from samudra.experiments.diffusion_global import (
    make_model,
    native_objective,
    observation_parts,
)
from samudra.experiments.diffusion_interventions import LatentReplay
from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.task_schedule import TaskSchedule, sample_indices


def owned_replay_slots(rank, world):
    if world not in (1, 2, 4, 8) or not 0 <= rank < world:
        raise ValueError("Rank/world must shard effective batch eight")
    return [micro - 4 for micro in range(rank, 8, world) if micro >= 4]


def validate_rank_states(states, world, require_populated=False):
    """Reject missing, duplicate, or incorrectly assigned replay chains on resume."""
    if len(states) != world or [s["rank"] for s in states] != list(range(world)):
        raise ValueError("Checkpoint rank states differ from world size/order")
    for state in states:
        if state["owned_slots"] != owned_replay_slots(state["rank"], world):
            raise ValueError("Checkpoint replay ownership changed")
        if not set(state["replay"]).issubset(state["owned_slots"]):
            raise ValueError("Checkpoint replay shard has a foreign chain")
        if require_populated and set(state["replay"]) != set(state["owned_slots"]):
            raise ValueError("Checkpoint lost an active replay chain")


def learning_rate(completed, total, peak=1e-4, minimum=1e-6):
    """Preserve scratch LR, with cosine cooldown over the final 12.5%."""
    if not 0 <= completed < total:
        raise ValueError("Invalid schedule position")
    cooldown = total - total // 8
    if completed < cooldown:
        return peak
    progress = (completed - cooldown) / max(1, total - 1 - cooldown)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2


def sum_gradients(parameters):
    """One FP32 vector reduction plus a presence mask; inputs already scaled /8.

    Globally unused parameters remain None (not zero) so AdamW neither updates
    their moments nor applies weight decay. A locally unused parameter receives
    the other ranks' gradient when it is used anywhere in the global batch.
    """
    parameters = list(parameters)
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("Expected nonempty FP32 master parameters")
    device = parameters[0].device
    present = torch.tensor(
        [p.grad is not None for p in parameters], device=device, dtype=torch.int32
    )
    gradient = torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in parameters
        ]
    )
    if dist.is_initialized():
        dist.all_reduce(present, op=dist.ReduceOp.MAX)
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
    offset = 0
    for p, used in zip(parameters, present.cpu().tolist(), strict=True):
        p.grad = gradient[offset : offset + p.numel()].view_as(p) if used else None
        offset += p.numel()
    return gradient


def capture_rank_state(rank, world, replay, device):
    return dict(
        rank=rank,
        owned_slots=owned_replay_slots(rank, world),
        replay=replay.state_dict(),
        cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
    )


def restore_rank_state(state, replay, device):
    replay.load_state_dict(state["replay"])
    torch.set_rng_state(state["cpu_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def gather_rank_states(local, world, rank):
    if world == 1:
        return [local]
    states = [None] * world if rank == 0 else None
    dist.gather_object(local, states, dst=0)
    return states


def emit(event, output):
    print(json.dumps(event), flush=True)
    with (output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("direct", "residual"), required=True)
    parser.add_argument("--updates", type=int, default=20320)
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--invocation-hours", type=float, default=11.5)
    args = parser.parse_args()
    if args.updates < 4 or args.updates % 2 or args.checkpoint_every < 1:
        raise ValueError("Require even total updates and positive checkpoint cadence")
    if args.stop_at is not None and not 0 < args.stop_at <= args.updates:
        raise ValueError("stop-at must be within the complete run budget")
    learning_rate(0, args.updates)
    schedule = TaskSchedule(args.updates // 2, args.updates // 2, "mixed")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    owned_replay_slots(rank, world)
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group(
            "nccl", timeout=datetime.timedelta(minutes=30), device_id=device
        )
    # Qualification and interrupted invocations use exactly the production budget
    # and schedule; --stop-at never silently changes LR or sample selection.
    output = args.root / ("smoke" if args.smoke else "runs") / args.arm
    output.mkdir(parents=True, exist_ok=True)
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    data = Samples(
        args.root / "data/observations",
        device,
        surface_fill="zero",
        global_observations=True,
    )
    data.use_observation_normalization()
    training = data.paths("train")
    if (len(training), len(data.paths("validation")), len(data.paths("test"))) != (
        243,
        9,
        96,
    ):
        raise ValueError("Observation cohorts changed")
    if not bool((data.area > 0).all()):
        raise ValueError("Expected full-latitude loss weights")
    reference_path = args.root / "checkpoints/data-reference-contract.json"
    reference_contract = json.loads(reference_path.read_text())
    # The metadata receipt is the only reference input. No pretrained weights or
    # optimizer are opened, even temporarily, by this from-scratch driver.
    for key, path in (
        ("observation_manifest", data.root / "SHA256SUMS"),
        ("stats", data.root / "statistics.npz"),
        ("grid", data.root / "grid.npz"),
        ("reference", args.root / "checkpoints/selection-reference.json"),
    ):
        if digest(path) != reference_contract[key]:
            raise ValueError(f"Changed data contract: {key}")
    spec = dict(
        version=1,
        campaign="scratch-distributed",
        arm=args.arm,
        processor_prediction=args.arm,
        initialization="fresh same-seed random weights",
        seed=1729,
        updates=args.updates,
        world_size=world,
        effective_batch=8,
        task_schedule="quadratic mixed",
        learning_rate=1e-4,
        cooldown_start=args.updates - args.updates // 8,
        minimum_lr=1e-6,
        weight_decay=0.01,
        clip=1.0,
        sampling_steps=32,
        members_train=2,
        noise_correlation=0.5,
        spatial_weight=0.5,
        global_observations=True,
        compile_decoder=True,
        autocast="bfloat16",
        master_parameters="float32",
        latent_bound_weight=0.1 if args.arm == "residual" else 0.0,
        replay="micros 4..7 own chains 0..3; burn-in 30/90/180/300d; six differentiable steps; four uses maximum",
        checkpoint_world_size_fixed=True,
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
        data_reference_sha256=digest(reference_path),
        reference_contract=reference_contract,
    )
    contract = {k: v for k, v in reference_contract.items() if k != "cooldown"}
    # Preserve the data fingerprints consumed by existing evaluators, while
    # replacing historical training claims rather than inheriting parent counts.
    contract.update(
        seed=1729,
        schedule=f"quadratic mixed {args.updates // 2}/{args.updates // 2}",
        producer=spec["producer"],
        compile_decoder=True,
        scratch_protocol=spec,
    )
    protocol = output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != spec:
        raise ValueError("Scratch protocol changed")
    last = output / "last.pt"
    if last.exists() != args.resume:
        raise ValueError("Use --resume exactly when a saved checkpoint exists")
    if rank == 0:
        atomic_json(spec, protocol)
    wave = build_wave(
        SimpleNamespace(
            root=args.root, output=output / f"rank-{rank}", export_only=True
        ),
        data,
    )
    del wave.model
    del wave.initializer
    gc.collect()
    torch.manual_seed(1729)
    np.random.seed(1729)
    random.seed(1729)
    model = make_model(data.grid["names"].tolist(), device)
    if args.arm == "direct":
        install_direct_processor(model)
    wave.initializer = SimpleNamespace(surface=model.surface)
    wave.prepare(wave.trainset)
    if list(wave.names) != list(data.grid["names"]):
        raise ValueError("Native channel order changed")
    if wave.run:
        wave.run.finish()
        wave.run = None
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.accumulated_recompile_limit = 256
    model.decoder.compile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    replay = LatentReplay(wave, bound=args.arm == "residual")
    completed = 0
    if args.resume:
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["intervention"] != spec or saved["contract"] != contract:
            raise ValueError("Resume contract changed")
        completed = saved["step"]
        if (
            saved["counts"] != schedule.counts(completed)
            or not 0 <= completed <= args.updates
        ):
            raise ValueError("Invalid checkpoint step/counts")
        validate_rank_states(
            saved["rank_states"], world, require_populated=completed > 0
        )
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rank_state(saved["rank_states"][rank], replay, device)
        del saved
    run = None
    if rank == 0 and not args.smoke:
        import wandb

        run = wandb.init(
            project="default",
            entity="ocean_emulators",
            group="diffusion-scratch-distributed",
            name=args.arm,
            id=f"diffscratch-{args.root.name}-{args.arm}",
            resume="allow",
            dir=str(output),
            config=spec,
            mode="online",
        )
        if run.disabled or not run.url:
            raise RuntimeError("Online W&B initialization failed")
        atomic_json(dict(url=run.url, id=run.id), output / "wandb.json")
        emit(dict(event="wandb", url=run.url), output)
    if world > 1:
        dist.barrier()

    def save(snapshot=False):
        states = gather_rank_states(
            capture_rank_state(rank, world, replay, device), world, rank
        )
        if rank == 0:
            validate_rank_states(states, world, require_populated=completed > 0)
            payload = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                contract=contract,
                intervention=spec,
                processor_prediction=args.arm,
                step=completed,
                counts=schedule.counts(completed),
                rank_states=states,
            )
            atomic_torch(payload, last)
            if snapshot:
                snapshot_path = output / f"step-{completed:05d}.pt"
                atomic_torch(
                    {
                        k: v
                        for k, v in payload.items()
                        if k not in ("optimizer", "rank_states")
                    },
                    snapshot_path,
                )
                # Publish the evaluation receipt only after the atomic file is complete.
                atomic_json(
                    dict(
                        checkpoint=str(snapshot_path),
                        sha256=digest(snapshot_path),
                        step=completed,
                        counts=schedule.counts(completed),
                    ),
                    output / f"EVAL_READY-{completed:05d}.json",
                )
            atomic_json(
                dict(
                    step=completed,
                    completed=completed,
                    counts=schedule.counts(completed),
                    complete=completed == args.updates,
                ),
                output / "state.json",
            )
        if world > 1:
            dist.barrier()

    if not args.resume:
        save(snapshot=True)
    started = time.monotonic()
    while completed < args.updates:
        # Synchronize requests before starting another expensive update, including
        # signals delivered to only one Slurm rank. Every rank follows this path.
        due = stop or time.monotonic() - started >= args.invocation_hours * 3600
        due |= args.stop_at is not None and completed >= args.stop_at
        flag = torch.tensor(int(due), device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if bool(flag):
            save()
            break
        task = schedule.task(completed)
        count = schedule.counts(completed)[task]
        lr = learning_rate(completed, args.updates)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        model.train()
        update_started = time.monotonic()
        sample_seed = 1729 + (100000 if task == "om4" else 0)
        indices = sample_indices(
            len(training) if task == "observation" else len(wave.trainset),
            sample_seed,
            count,
            8,
        )
        measurements = torch.zeros(4, device=device, dtype=torch.float64)
        replay_records = []
        for micro in range(rank, 8, world):
            seed = sample_seed + 1000000 + count * 8 + micro
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if task == "observation":
                    sample = data.load(training[indices[micro]])
                    for label, loss, parts in observation_parts(
                        model, data, sample, seed
                    ):
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(label)
                        (loss / 8).backward()
                        measurements[0] += loss.detach().double() / 8
                        for pos, name in enumerate(
                            ("forecast", "completion", "reconstruction"), 1
                        ):
                            measurements[pos] += parts.get(name, 0.0) / 8
                else:
                    coverage = data.load(training[(count * 8 + micro) % len(training)])[
                        "validity"
                    ][:, :19]
                    if micro >= 4:
                        loss = replay.loss(model, micro - 4, count, coverage, seed)
                        replay_records.append(dict(slot=micro - 4, **replay.last_info))
                    else:
                        loss = native_objective(
                            model, wave, indices[micro], coverage, seed
                        )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite native objective")
                    (loss / 8).backward()
                    measurements[0] += loss.detach().double() / 8
        sum_gradients(model.parameters())
        module_norms = {
            name: float(
                torch.linalg.vector_norm(
                    torch.stack(
                        [
                            p.grad.detach().float().norm()
                            for p in module.parameters()
                            if p.grad is not None
                        ]
                    )
                )
            )
            for name, module in (
                ("initializer", model.encoder),
                ("processor", model.processor),
                ("decoder", model.decoder),
            )
        }
        if not all(math.isfinite(v) and v > 0 for v in module_norms.values()):
            raise FloatingPointError(
                f"Missing or nonfinite global gradients: {module_norms}"
            )
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        if world > 1:
            dist.all_reduce(measurements, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        timing = torch.tensor(time.monotonic() - update_started, device=device)
        if world > 1:
            dist.all_reduce(timing, op=dist.ReduceOp.MAX)
        completed += 1
        # Keep each chain's actual exposure visible without serializing latents.
        if replay_records:
            with (output / f"replay-rank-{rank}.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(dict(step=completed, records=replay_records)) + "\n"
                )
        if rank == 0:
            values = measurements.cpu().tolist()
            event = dict(
                event="update",
                step=completed,
                task=task,
                task_update=count + 1,
                counts=schedule.counts(completed),
                loss=values[0],
                forecast=values[1],
                completion=values[2],
                reconstruction=values[3],
                gradient_norm=float(norm),
                learning_rate=lr,
                seconds=float(timing),
                world_size=world,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                gradient_norms=module_norms,
            )
            emit(event, output)
            if run:
                run.log(
                    {
                        f"train/{k}": v
                        for k, v in event.items()
                        if isinstance(v, (int, float))
                    },
                    step=completed,
                )
        milestone = (
            task == "observation"
            and count + 1 in (2000, 4000, 6000, 8000, args.updates // 2)
        ) or completed in (4000, spec["cooldown_start"], args.updates)
        if completed % args.checkpoint_every == 0 or milestone:
            save(snapshot=milestone)
    if rank == 0:
        result = dict(
            complete=completed == args.updates,
            step=completed,
            counts=schedule.counts(completed),
            invocation_seconds=time.monotonic() - started,
            world_size=world,
        )
        atomic_json(
            result, output / ("COMPLETE.json" if result["complete"] else "PAUSED.json")
        )
        emit(dict(event="invocation_finished", **result), output)
        if run:
            run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
