# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Budget-limited, source-homogeneous DDP training for the velocity transfer screen."""

import argparse
import contextlib
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from samudra.experiments.velocity_transfer.data import VelocitySource, eligible_anchors
from samudra.experiments.velocity_transfer.model import (
    TrainingRollout,
    VelocitySamudra,
    rollout,
    weighted_loss,
)


def choose_task(variant: str, rng: np.random.Generator, fraction: float) -> str:
    if variant == "D0":
        return "duacs"
    if variant == "D5":
        return (
            str(rng.choice(["om4_1deg", "om4_quarterdeg"]))
            if fraction < 0.4
            else "duacs"
        )
    tasks, weights = {
        "D1": (["duacs", "om4_1deg"], [0.6, 0.4]),
        "D2": (["duacs", "om4_quarterdeg"], [0.6, 0.4]),
        "D3": (["duacs", "om4_1deg", "om4_quarterdeg"], [0.6, 0.2, 0.2]),
        "D4": (["duacs", "om4_1deg", "om4_quarterdeg"], [0.6, 0.2, 0.2]),
    }[variant]
    return str(rng.choice(tasks, p=weights))


def precision(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )


@torch.no_grad()
def validate(model, source, device, rank=0, world=1, count=12):
    """Fixed validation dates; compare physical vector RMSE on identical valid cells."""
    anchors = eligible_anchors(source.times, "validation")
    if not len(anchors):
        raise ValueError("No validation windows")
    anchors = anchors[
        np.linspace(0, len(anchors) - 1, min(count, len(anchors)), dtype=int)
    ]
    totals = torch.zeros((3, 6), dtype=torch.float64, device=device)
    if world > 1:
        # Evaluate the exact running statistics that rank zero checkpoints.
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    model.eval()
    for anchor in anchors[rank::world]:
        batch = source.batch(int(anchor), 6, device)
        with precision(device):
            predictions = rollout(model, batch, "duacs", 6)
        weights = batch["target_valid"].float() * batch["area"][:, None]
        err = (predictions.float() - batch["targets"]) * batch["std"][:, None]
        persistent_err = (batch["history"][:, -1:, :, :, :] - batch["targets"]) * batch[
            "std"
        ][:, None]
        totals[0] += (err.square() * weights).sum(dim=(0, 2, 3, 4))
        totals[1] += (persistent_err.square() * weights).sum(dim=(0, 2, 3, 4))
        totals[2] += weights.sum(dim=(0, 2, 3, 4))
    if world > 1:
        dist.all_reduce(totals)
    if (totals[2] <= 0).any():
        raise ValueError("Empty validation target")
    rmse, persistence = (totals[:2] / totals[2]).sqrt().cpu().numpy()
    metrics = {}
    for step in (1, 2, 4, 6):
        metrics[f"validation/vector_rmse_{step * 5}d"] = float(rmse[step - 1])
        metrics[f"validation/persistence_rmse_{step * 5}d"] = float(
            persistence[step - 1]
        )
        metrics[f"validation/skill_{step * 5}d"] = float(
            1 - rmse[step - 1] / persistence[step - 1]
        )
    model.train()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=[f"D{i}" for i in range(6)], required=True)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--gpu-hours", type=float, default=48)
    parser.add_argument("--max-updates", type=int, default=1_000_000)
    parser.add_argument("--validate-every", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--read-threads", type=int, default=4)
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 96, 128, 192])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--wandb-mode", choices=("online", "disabled"), default="online"
    )
    args = parser.parse_args()
    if int(os.environ.get("SLURM_RESTART_COUNT", "0")) > 0:
        args.resume = True
    if (
        min(
            args.gpu_hours,
            args.max_updates,
            args.validate_every,
            args.read_threads,
            args.validation_count,
            args.checkpoint_every,
        )
        <= 0
    ):
        parser.error(
            "Budgets, thread counts, and validation intervals must be positive"
        )
    start = time.monotonic()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    task_rng = np.random.default_rng(args.seed)
    sample_rng = np.random.default_rng(args.seed + 1000 + rank)
    from samudra_rust_loader import FlatOm4ReadPool

    pool = FlatOm4ReadPool(args.read_threads)
    source_names = ["duacs"]
    if args.variant != "D0":
        source_names += (
            ["om4_1deg"]
            if args.variant == "D1"
            else ["om4_quarterdeg"]
            if args.variant == "D2"
            else ["om4_1deg", "om4_quarterdeg"]
        )
    sources = {
        name: VelocitySource(args.data_root / name, pool) for name in source_names
    }
    anchors = {
        name: eligible_anchors(source.times, "train")
        for name, source in sources.items()
    }
    if any(not len(indices) for indices in anchors.values()):
        raise ValueError("A source has no complete training windows")
    model = VelocitySamudra(args.widths, use_geometry=args.variant != "D4").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        world_size=world,
        code_commit=os.environ.get("WANDB_GIT_COMMIT"),
        rust_read_threads_per_rank=args.read_threads,
        cpu_workers=0,
    )
    checkpoint_path = args.output / "checkpoint.pt"
    update, previous_elapsed = 0, 0.0
    exposure = dict.fromkeys(source_names, 0)
    best = float("inf")
    if args.resume:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for key in (
            "variant",
            "seed",
            "widths",
            "data_root",
            "world_size",
            "gpu_hours",
        ):
            if state["config"][key] != config[key]:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        update, previous_elapsed, exposure, best = (
            state["update"],
            state["elapsed"],
            state["exposure"],
            state["best"],
        )
        task_rng.bit_generator.state = state["task_rng"]
        sample_rng.bit_generator.state = state["rank_rng"][rank]["sample"]
        torch.set_rng_state(state["rank_rng"][rank]["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["rank_rng"][rank]["cuda"], device)
    elif rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    tracker = None
    if rank == 0:
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        provenance = {
            key: os.environ.get(key)
            for key in (
                "SAMUDRA_CODE_COMMIT",
                "SAMUDRA_CODE_REPO_URL",
                "SAMUDRA_CODE_LAYER_SHA256",
                "SAMUDRA_CONTAINER_GIT_COMMIT",
                "SAMUDRA_CONTAINER_IMAGE_REF",
                "SAMUDRA_CONTAINER_SIF_PATH",
            )
        }
        (args.output / "run-provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        import wandb

        tracker = wandb.init(
            project="samudra-velocity-transfer",
            entity="ocean_emulators",
            name=args.output.name,
            group="om4-local-duacs-2026-09",
            config=config,
            mode=args.wandb_mode,
            id=args.output.name,
            resume="allow",
            dir=str(args.output),
        )
    training_model = TrainingRollout(model)
    wrapped = (
        DistributedDataParallel(
            training_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        if world > 1
        else training_model
    )
    budget_seconds = args.gpu_hours * 3600 / world

    def elapsed():
        seconds = torch.tensor(
            previous_elapsed + time.monotonic() - start,
            device=device,
            dtype=torch.float64,
        )
        if world > 1:
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        return seconds.item()

    def checkpoint(seconds, filename="checkpoint.pt"):
        rng = {
            "sample": sample_rng.bit_generator.state,
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }
        rank_rng: list[Any] = [None] * world
        if world > 1:
            dist.all_gather_object(rank_rng, rng)
        else:
            rank_rng[0] = rng
        if rank == 0:
            state = {
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "elapsed": seconds,
                "exposure": exposure,
                "best": best,
                "task_rng": task_rng.bit_generator.state,
                "rank_rng": rank_rng,
            }
            temporary = args.output / f"{filename}.tmp"
            torch.save(state, temporary)
            temporary.replace(args.output / filename)

    def log(record):
        if rank == 0:
            print(json.dumps(record), flush=True)
            with (args.output / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if tracker is not None:
                tracker.log(record, step=update)

    model.train()
    seconds = elapsed()
    if not args.resume:
        checkpoint(seconds)
    while update < args.max_updates and seconds < budget_seconds * 0.97:
        fraction = min(seconds / budget_seconds, 1)
        task = choose_task(args.variant, task_rng, fraction)
        source = sources[task]
        anchor = int(sample_rng.choice(anchors[task]))
        regional = task == "om4_quarterdeg"
        crop = source.random_crop(sample_rng) if regional else None
        steps = 1 if regional else 2
        batch = source.batch(anchor, steps, device, crop)
        lr = (
            args.learning_rate
            * min((update + 1) / 100, 1)
            * (0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            predictions = wrapped(batch, source.kind, steps, not regional)
            loss = weighted_loss(predictions, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss for {task}, anchor {anchor}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        update += 1
        exposure[task] += world
        seconds = elapsed()
        if update == 1 or update % 10 == 0:
            log(
                {
                    "update": update,
                    "task": task,
                    "train/loss": float(loss.detach()),
                    "train/grad_norm": float(norm),
                    "learning_rate": lr,
                    "gpu_hours": seconds * world / 3600,
                    "peak_rss_gib_rank0": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    / 1024**2,
                    "peak_cuda_gib_rank0": torch.cuda.max_memory_allocated(device)
                    / 1024**3
                    if device.type == "cuda"
                    else 0,
                    **{f"exposure/{k}": v for k, v in exposure.items()},
                }
            )
            if rank == 0:
                print(
                    f"Training Epoch: [0] [{update}/{args.max_updates}] loss={float(loss.detach()):.6f}",
                    flush=True,
                )
        if update % args.validate_every == 0 and seconds < budget_seconds * 0.9:
            metrics = validate(
                model, sources["duacs"], device, rank, world, args.validation_count
            )
            seconds = elapsed()
            log(metrics)
            score = metrics["validation/vector_rmse_10d"]
            if score < best:
                best = score
                checkpoint(seconds, "best.pt")
            checkpoint(seconds)
        elif update % args.checkpoint_every == 0:
            checkpoint(seconds)
    checkpoint(elapsed())
    log(
        {
            "complete": True,
            "update": update,
            "gpu_hours": elapsed() * world / 3600,
            "best_validation_rmse_10d": best if np.isfinite(best) else None,
        }
    )
    if tracker is not None:
        tracker.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
