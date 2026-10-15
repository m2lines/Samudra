# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Observation-only fork with preserved Adam state and a newly reset weight EMA."""

import argparse
import datetime
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from samudra.experiments.diffusion_global import make_model, observation_parts
from samudra.experiments.diffusion_scratch_distributed import (
    emit,
    gather_rank_states,
    owned_replay_slots,
    sum_gradients,
)
from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.task_schedule import sample_indices


def learning_rate(completed, total, peak=1e-4, minimum=1e-6):
    if total < 2 or not 0 <= completed < total:
        raise ValueError("Invalid cooldown schedule position")
    return peak + (minimum - peak) * completed / (total - 1)


def reset_ema(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def update_ema(ema, model, decay):
    if not 0 <= decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    state = model.state_dict()
    if state.keys() != ema.keys():
        raise ValueError("EMA state keys changed")
    parameters = dict(model.named_parameters())
    for key, value in state.items():
        if key in parameters and value.is_floating_point():
            ema[key].lerp_(value.detach(), 1 - decay)
        else:
            ema[key].copy_(value)


def validate_parent(saved, world):
    if saved.get("processor_prediction") != "residual":
        raise ValueError("Expected residual parent")
    if saved["counts"] != {"om4": 7721, "observation": 4000} or saved["step"] != 11721:
        raise ValueError("Expected immutable 4k-observation full parent checkpoint")
    if "optimizer" not in saved or len(saved["rank_states"]) != world:
        raise ValueError("Require full parent Adam and matching rank RNG states")
    if [state["rank"] for state in saved["rank_states"]] != list(range(world)):
        raise ValueError("Parent rank ordering changed")


def capture_rng(rank, device):
    return dict(
        rank=rank,
        cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device),
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
    )


def restore_rng(state, device):
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"], device)
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=2 ** (-1 / 100))
    parser.add_argument("--stop-at", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--invocation-hours", type=float, default=23)
    args = parser.parse_args()
    learning_rate(0, args.updates)
    if not 0 <= args.ema_decay < 1 or args.checkpoint_every < 1:
        raise ValueError("Invalid EMA/cadence")
    if args.stop_at is not None and not 0 < args.stop_at <= args.updates:
        raise ValueError("Invalid stop-at")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    owned_replay_slots(rank, world)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group(
            "nccl", timeout=datetime.timedelta(minutes=30), device_id=device
        )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pt"
    if last.exists() != args.resume:
        raise ValueError("Use --resume exactly when last.pt exists")
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    validate_parent(parent, world)
    spec = dict(
        version=1,
        campaign="scratch-obs-cooldown",
        parent_sha256=digest(args.parent),
        parent_counts=parent["counts"],
        parent_step=parent["step"],
        updates=args.updates,
        world_size=world,
        effective_batch=8,
        ema_decay=args.ema_decay,
        ema_reset="raw parent weights",
        optimizer="preserve AdamW",
        learning_rate=1e-4,
        minimum_lr=1e-6,
        schedule="linear observation-only",
        sampling_steps=32,
        members_train=2,
        autocast="bfloat16",
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
    )
    protocol = output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != spec:
        raise ValueError("Cooldown protocol changed")
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
        raise ValueError("Expected full-latitude weights")
    reference = json.loads(
        (args.root / "checkpoints/data-reference-contract.json").read_text()
    )
    for key, path in (
        ("observation_manifest", data.root / "SHA256SUMS"),
        ("stats", data.root / "statistics.npz"),
        ("grid", data.root / "grid.npz"),
        ("reference", args.root / "checkpoints/selection-reference.json"),
    ):
        if digest(path) != reference[key] or parent["contract"][key] != reference[key]:
            raise ValueError(f"Data contract changed: {key}")
    contract = dict(parent["contract"], observation_cooldown=spec)
    model = make_model(data.grid["names"].tolist(), device)
    model.load_state_dict(parent["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    optimizer.load_state_dict(parent["optimizer"])
    ema = reset_ema(model)
    if rank == 0 and not args.resume:
        adam_steps = [
            float(state["step"])
            for state in optimizer.state.values()
            if "step" in state
        ]
        if not adam_steps:
            raise ValueError("Parent Adam state is empty")
        atomic_json(
            dict(
                parent_sha256=spec["parent_sha256"],
                optimizer_states=len(optimizer.state),
                adam_step_min=min(adam_steps),
                adam_step_max=max(adam_steps),
                ema_reset_exact=all(
                    torch.equal(ema[k], v) for k, v in model.state_dict().items()
                ),
                world_size=world,
                counts=parent["counts"],
            ),
            output / "FORK.json",
        )
    completed = 0
    rng_state = parent["rank_states"][rank]
    del parent
    if args.resume:
        saved = torch.load(last, map_location="cpu", weights_only=False)
        completed = saved["cooldown_step"]
        if saved["intervention"] != spec or saved["contract"] != contract:
            raise ValueError("Resume protocol changed")
        if not 0 <= completed <= args.updates or saved["counts"] != {
            "om4": 7721,
            "observation": 4000 + completed,
        }:
            raise ValueError("Invalid resume counts")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        ema = {k: v.to(device) for k, v in saved["ema"].items()}
        if len(saved["rank_states"]) != world or [
            s["rank"] for s in saved["rank_states"]
        ] != list(range(world)):
            raise ValueError("Resume rank RNG ownership changed")
        rng_state = saved["rank_states"][rank]
        del saved
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.accumulated_recompile_limit = 256
    model.decoder.compile()
    run = None
    if rank == 0:
        import wandb

        atomic_json(spec, protocol)
        run = wandb.init(
            project="default",
            entity="ocean_emulators",
            group="diffusion-scratch-obs-cooldown",
            name=output.name,
            id=f"diffcool-{output.name}",
            resume="allow",
            dir=str(output),
            config=spec,
            mode="online",
        )
        if run.disabled or not run.url:
            raise RuntimeError("Online W&B initialization failed")
        atomic_json(dict(url=run.url, id=run.id), output / "wandb.json")
    # Logging initialization must not perturb the saved rank RNG stream.
    restore_rng(rng_state, device)
    del rng_state
    stop = False

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def save(snapshot=False):
        states = gather_rank_states(capture_rng(rank, device), world, rank)
        if rank == 0:
            counts = dict(om4=7721, observation=4000 + completed)
            payload = dict(
                model=model.state_dict(),
                ema=ema,
                optimizer=optimizer.state_dict(),
                contract=contract,
                intervention=spec,
                processor_prediction="residual",
                step=11721 + completed,
                cooldown_step=completed,
                counts=counts,
                rank_states=states,
            )
            atomic_torch(payload, last)
            if snapshot:
                for name, weights in (("raw", payload["model"]), ("ema", ema)):
                    path = output / (
                        f"final-{name}.pt"
                        if completed == args.updates
                        else f"cooldown-{completed:05d}-{name}.pt"
                    )
                    atomic_torch(
                        {
                            **{
                                k: v
                                for k, v in payload.items()
                                if k not in ("optimizer", "rank_states", "ema")
                            },
                            "model": weights,
                            "weight_variant": name,
                        },
                        path,
                    )
                    atomic_json(
                        dict(
                            checkpoint=str(path),
                            sha256=digest(path),
                            counts=counts,
                            step=11721 + completed,
                            cooldown_step=completed,
                            weight_variant=name,
                        ),
                        output / f"EVAL_READY-{completed:05d}-{name}.json",
                    )
            atomic_json(
                dict(
                    step=11721 + completed,
                    cooldown_step=completed,
                    counts=counts,
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
        due = stop or time.monotonic() - started >= args.invocation_hours * 3600
        due |= args.stop_at is not None and completed >= args.stop_at
        flag = torch.tensor(int(due), device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if bool(flag):
            save(snapshot=True)
            break
        count = 4000 + completed
        lr = learning_rate(completed, args.updates)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        model.train()
        update_started = time.monotonic()
        indices = sample_indices(len(training), 1729, count, 8)
        measurements = torch.zeros(4, device=device, dtype=torch.float64)
        for micro in range(rank, 8, world):
            sample = data.load(training[indices[micro]])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for label, loss, parts in observation_parts(
                    model, data, sample, 1729 + 1000000 + count * 8 + micro
                ):
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(label)
                    (loss / 8).backward()
                    measurements[0] += loss.detach().double() / 8
                    for pos, name in enumerate(
                        ("forecast", "completion", "reconstruction"), 1
                    ):
                        measurements[pos] += parts.get(name, 0.0) / 8
        sum_gradients(model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        update_ema(ema, model, args.ema_decay)
        if world > 1:
            dist.all_reduce(measurements, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        timing = torch.tensor(time.monotonic() - update_started, device=device)
        if world > 1:
            dist.all_reduce(timing, op=dist.ReduceOp.MAX)
        completed += 1
        if rank == 0:
            values = measurements.cpu().tolist()
            event = dict(
                event="update",
                step=11721 + completed,
                cooldown_step=completed,
                counts=dict(om4=7721, observation=4000 + completed),
                task="observation",
                loss=values[0],
                forecast=values[1],
                completion=values[2],
                reconstruction=values[3],
                gradient_norm=float(norm),
                learning_rate=lr,
                seconds=float(timing),
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
        milestone = completed in (250, 500, args.updates)
        if completed % args.checkpoint_every == 0 or milestone:
            save(snapshot=milestone)
    if rank == 0:
        atomic_json(
            dict(
                complete=completed == args.updates,
                cooldown_step=completed,
                counts=dict(om4=7721, observation=4000 + completed),
            ),
            output / ("COMPLETE.json" if completed == args.updates else "PAUSED.json"),
        )
        if run:
            run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
