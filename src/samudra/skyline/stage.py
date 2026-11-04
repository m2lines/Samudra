# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Single-node DDP continuation with corrected coverage and allocation budgets."""

import argparse
import hashlib
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from numcodecs import blosc  # type: ignore[import-untyped]
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .coverage import corrected_mean, sample_coverage
from .data import OceanSource
from .model import ModelConfig, PointTransformer, Vocabulary
from .run import FrameStream, emit, evaluate, save_atomic, to_device
from .utilization import GpuMonitor


def merge_validation(parts):
    """Merge per-date shards without averaging RMSE or dropping support counts."""
    result = dict(parts[0])
    origins = [i for part in parts for i in part["origins"]]
    if len(set(origins)) != len(origins):
        raise ValueError("Validation shards overlap")
    result["origins"] = sorted(origins)
    result["per_depth"] = []
    for z in range(19):
        rows = [p["per_depth"][z] for p in parts]
        counts = np.array([r["count"] for r in rows], dtype=np.float64)
        row = dict(rows[0], count=int(counts.sum()))
        for key in rows[0]:
            if key in {"depth", "depth_m", "count"} or rows[0][key] is None:
                continue
            values = np.array([r[key] for r in rows], dtype=np.float64)
            if key == "outside_support":
                row[key] = int(values.sum())
            elif key == "in_support_density_nll":
                support = counts - np.array([r["outside_support"] for r in rows])
                row[key] = float(np.average(values, weights=support))
            elif key.endswith("rmse_c"):
                row[key] = math.sqrt(float(np.average(values**2, weights=counts)))
            else:
                row[key] = float(np.average(values, weights=counts))
        result["per_depth"].append(row)
    result["mean_depth_nmse"] = float(np.mean([r["nmse"] for r in result["per_depth"]]))
    if result["mean_depth_bin_nll"] is not None:
        result["mean_depth_bin_nll"] = float(
            np.mean([r["bin_nll"] for r in result["per_depth"]])
        )
        result["randomized_pit_counts"] = np.sum(
            [p["randomized_pit_counts"] for p in parts], axis=0
        ).tolist()
    return result


def allocation_used(config, prior, allocation_start):
    return prior + max(0.0, time.time() - allocation_start)


def train_stage(config, output):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != config["world_size"]:
        raise ValueError("DDP world size differs from the pinned experiment")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15), device_id=device)
    torch.set_num_threads(config.get("torch_threads", 4))
    blosc.set_nthreads(1)
    output.mkdir(parents=True, exist_ok=True)
    rank_output = output / f"rank-{rank}"
    rank_output.mkdir(exist_ok=True)
    prior = float(os.environ.get("SKYLINE_PRIOR_SECONDS", "0"))
    allocation_start = float(
        os.environ.get("SKYLINE_ALLOCATION_START", str(time.time()))
    )
    deadline = config["allocation_seconds"] - config["finish_reserve_seconds"]
    source = OceanSource(
        config["source"], config["readers"], config.get("reference_root")
    )
    monitor = GpuMonitor(
        rank_output,
        torch.cuda.get_device_properties(device).uuid,
        int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        config["utilization_guard"],
    )
    try:
        with monitor:
            _train(
                config,
                output,
                rank_output,
                source,
                monitor,
                device,
                rank,
                world,
                prior,
                allocation_start,
                deadline,
            )
    except BaseException:
        if rank == 0:
            (output / "COMPLETE.json").unlink(missing_ok=True)
        raise
    finally:
        source.close()
        dist.destroy_process_group()


def _train(
    config,
    output,
    rank_output,
    source,
    monitor,
    device,
    rank,
    world,
    prior,
    allocation_start,
    deadline,
):
    support = json.loads(Path(config["support"]).read_text())
    for key in ("metadata_sha256", "mean", "std"):
        if support["source"][key] != source.receipt()[key]:
            raise ValueError(f"Training support mismatch: {key}")
    if not support["complete_training_scan"]:
        raise ValueError("Stage requires the complete training support audit")
    seed = config["seed"] + rank * 1009
    torch.manual_seed(config["seed"])
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    model = PointTransformer(ModelConfig(**config["model"])).to(device)
    if model.config.objective not in {"absolute_ce", "regression"}:
        raise ValueError("This continuation compares absolute CE and regression")
    vocab = Vocabulary(support["absolute_bounds"], model.config.bins).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
        fused=True,
    )
    latest = output / "latest.pt"
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    resumed = latest.exists()
    if resumed and not (restart or config.get("resume", False)):
        raise FileExistsError(latest)
    state = torch.load(
        latest if resumed else config["initial_checkpoint"],
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if state["config"]["model"] != config["model"]:
        raise ValueError("Initial model architecture differs")
    model.load_state_dict(state["model"], strict=True)
    vocab.load_state_dict(state["vocabulary"], strict=True)
    step, best_nll, best_nmse, stream_state = 0, math.inf, math.inf, None
    initial_step = state.get("initial_step", state["step"])
    if resumed:
        if state["config"] != config:
            raise ValueError("Continuation configuration changed on resume")
        step, best_nll, best_nmse = state["step"], state["best_nll"], state["best_nmse"]
        optimizer.load_state_dict(state["optimizer"])
        saved = state["rank_states"][rank]
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
        stream_state = saved["stream"]
    del state
    ddp = DistributedDataParallel(
        model, device_ids=[device.index], broadcast_buffers=False
    )
    stream = FrameStream(source, seed + 100, stream_state)
    run = None
    if rank == 0:
        provenance = dict(
            config=config,
            source=source.receipt(),
            initial_step=initial_step,
            source_commit=os.environ.get("SKYLINE_COMMIT"),
            archive_sha256=os.environ.get("SKYLINE_ARCHIVE_SHA256"),
            job=os.environ.get("SLURM_JOB_ID"),
            restart=restart,
            world_size=world,
            effective_batch=world * config["microbatch"],
            gpu=torch.cuda.get_device_name(),
            prior_allocation_seconds=prior,
            allocation_start=allocation_start,
            resumed_step=step,
            parameters=sum(p.numel() for p in model.parameters()),
        )
        (output / f"provenance-restart-{restart}.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        if config["wandb"]:
            import wandb

            run = wandb.init(
                project="temperature-skyline",
                entity="ocean_emulators",
                name=output.name,
                id=hashlib.sha256(str(output).encode()).hexdigest()[:12],
                resume="allow",
                config=provenance,
            )
        emit(dict(event="ddp_start", **provenance))
    last_save = last_validation = allocation_used(config, prior, allocation_start)
    stop_reason = None
    complete = False
    model.train()
    try:
        while True:
            used = allocation_used(config, prior, allocation_start)
            flags = torch.tensor(
                [int(used >= deadline), int(monitor.reason is not None)], device=device
            )
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            if flags[1]:
                stop_reason = "GPU telemetry failed on at least one rank"
            finished = (
                bool(flags[0])
                or stop_reason is not None
                or step >= config["max_updates"]
            )
            if not finished:
                begin = time.monotonic()
                step += 1
                if stream.used >= config["samples_per_frame"]:
                    stream.advance()
                batch = to_device(
                    sample_coverage(
                        source,
                        stream.frame,
                        rng,
                        config["microbatch"],
                        config["crop"],
                        config["coverage"],
                    ),
                    device,
                )
                stream.used += config["microbatch"]
                optimizer.zero_grad(set_to_none=True)
                fraction = min(1.0, used / max(1, deadline))
                factor = min(1.0, step / config["warmup"]) * (
                    0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2
                )
                for group in optimizer.param_groups:
                    group["lr"] = config["learning_rate"] * factor
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = ddp(batch)
                if model.config.objective == "regression":
                    per_example = F.mse_loss(
                        logits.squeeze(-1), batch["target"], reduction="none"
                    )
                else:
                    if vocab.outside(batch["target"], batch["depth"]).any():
                        raise ValueError("Target outside training-only support")
                    per_example = F.cross_entropy(
                        logits,
                        vocab.encode(batch["target"], batch["depth"]),
                        reduction="none",
                    )
                loss = corrected_mean(per_example, batch["importance"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite importance-corrected loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                torch.cuda.synchronize()
                if step == 1 or step % config["log_every"] == 0:
                    stats = torch.stack(
                        (
                            loss.detach(),
                            batch["importance"].mean(),
                            batch["importance"].square().mean(),
                            norm,
                        )
                    )
                    dist.all_reduce(stats)
                    stats /= world
                    if rank == 0:
                        record = dict(
                            step=step,
                            loss=float(stats[0]),
                            importance_mean=float(stats[1]),
                            importance_ess_fraction=float(stats[1].square() / stats[2]),
                            grad_norm=float(stats[3]),
                            lr=optimizer.param_groups[0]["lr"],
                            seconds=time.monotonic() - begin,
                            allocation_seconds=allocation_used(
                                config, prior, allocation_start
                            ),
                            accepted_examples=step * world * config["microbatch"],
                            utilization=monitor.mean,
                            peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                        )
                        emit(record, output / "training.jsonl")
                        if run:
                            run.log(
                                {"train/" + k: v for k, v in record.items()}, step=step
                            )
            used = allocation_used(config, prior, allocation_start)
            # Rank zero decides timer events; ranks can cross a threshold at
            # slightly different times even after a synchronized optimizer step.
            events = torch.tensor(
                [
                    int(finished),
                    int(
                        finished or used - last_validation >= config["validate_seconds"]
                    ),
                    int(finished or used - last_save >= config["save_seconds"]),
                ],
                device=device,
            )
            dist.broadcast(events, src=0)
            if events[1] and not stop_reason:
                local = evaluate(
                    model, vocab, source, config, device, rank_output, step, rank, world
                )
                parts: list = [None] * world
                dist.all_gather_object(parts, local)
                validation = merge_validation(parts)
                if rank == 0:
                    (output / f"validation-{step:07d}.json").write_text(
                        json.dumps(validation, indent=2) + "\n"
                    )
                    weights = dict(
                        model=model.state_dict(),
                        vocabulary=vocab.state_dict(),
                        config=config,
                        step=step,
                        initial_step=initial_step,
                        validation=validation,
                    )
                    if validation["mean_depth_nmse"] < best_nmse:
                        best_nmse = validation["mean_depth_nmse"]
                        save_atomic(weights, output / "best-error.pt")
                    nll = validation["mean_depth_bin_nll"]
                    if nll is not None and nll < best_nll:
                        best_nll = nll
                        save_atomic(weights, output / "best-likelihood.pt")
                    emit(
                        dict(
                            event="validation",
                            step=step,
                            nmse=validation["mean_depth_nmse"],
                            nll=nll,
                        ),
                        output / "training.jsonl",
                    )
                    if run:
                        metrics = {"validation/nmse": validation["mean_depth_nmse"]}
                        if nll is not None:
                            metrics["validation/nll"] = nll
                        metrics.update(
                            {
                                f"validation/thetao_{r['depth']}/rmse_c": r["rmse_c"]
                                for r in validation["per_depth"]
                            }
                        )
                        run.log(metrics, step=step)
                last_validation = allocation_used(config, prior, allocation_start)
                model.train()
                dist.barrier()
            if events[2] or events[1]:
                local_state = dict(
                    numpy_rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(device),
                    stream=stream.state(),
                )
                rank_states: list = [None] * world
                dist.all_gather_object(rank_states, local_state)
                if rank == 0:
                    save_atomic(
                        dict(
                            model=model.state_dict(),
                            vocabulary=vocab.state_dict(),
                            optimizer=optimizer.state_dict(),
                            config=config,
                            step=step,
                            initial_step=initial_step,
                            best_nll=best_nll,
                            best_nmse=best_nmse,
                            rank_states=rank_states,
                            allocation_seconds=allocation_used(
                                config, prior, allocation_start
                            ),
                        ),
                        latest,
                    )
                    emit(dict(event="checkpoint", step=step), output / "training.jsonl")
                dist.barrier()
                last_save = allocation_used(config, prior, allocation_start)
            if events[0]:
                break
        # DDP replicas must agree after all updates, including corrected losses.
        signature = torch.stack([p.detach().float().sum() for p in model.parameters()])
        signatures = [torch.empty_like(signature) for _ in range(world)]
        dist.all_gather(signatures, signature)
        if any(not torch.equal(signature, s) for s in signatures):
            raise RuntimeError("DDP parameter signatures differ")
        if stop_reason:
            raise RuntimeError(stop_reason)
        complete = True
        if rank == 0:
            save_atomic(
                dict(
                    step=step,
                    initial_step=initial_step,
                    allocation_seconds=allocation_used(config, prior, allocation_start),
                    accepted_examples=step * world * config["microbatch"],
                    world_size=world,
                    replicas_agree=True,
                    qualification=config["qualification"],
                ),
                output / "completion.pt",
            )
            (output / "COMPLETE.json").write_text(
                json.dumps(
                    dict(
                        step=step,
                        initial_step=initial_step,
                        allocation_seconds=allocation_used(
                            config, prior, allocation_start
                        ),
                        world_size=world,
                        qualification=config["qualification"],
                        replicas_agree=True,
                    )
                )
                + "\n"
            )
    finally:
        stream.close()
        if run:
            run.finish(exit_code=0 if complete else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_stage(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
