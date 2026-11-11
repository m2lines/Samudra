# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fresh, update-budgeted DDP output-task comparison, with resumable EMA."""

import argparse
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from numcodecs import blosc  # type: ignore[import-untyped]
from torch.nn.parallel import DistributedDataParallel

from .batching import device_batches
from .channel_vocabulary import ChannelVocabulary
from .comparison_eval import evaluate_comparison
from .coverage import sample_coverage
from .ema import ema_weights, restore_ema
from .model import ModelConfig, PointTransformer
from .multitask import ComparisonSource, learning_rate_factor
from .optimization import accumulate_gradients, set_synchronized_learning_rate
from .replicas import verify_replicas
from .run import FrameStream, emit, save_atomic
from .utilization import GpuMonitor


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train(config, output):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != config["world_size"]:
        raise ValueError("World size differs from the pinned comparison")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=20), device_id=device)
    torch.set_num_threads(config["torch_threads"])
    blosc.set_nthreads(1)
    output.mkdir(parents=True, exist_ok=True)
    rank_output = output / f"rank-{rank}"
    rank_output.mkdir(exist_ok=True)
    source = ComparisonSource(
        config["source"],
        config["readers"],
        config.get("reference_root"),
        arm=config["arm"],
    )
    try:
        with GpuMonitor(
            rank_output,
            torch.cuda.get_device_properties(device).uuid,
            int(os.environ.get("SLURM_RESTART_COUNT", "0")),
            config["utilization_guard"],
        ) as monitor:
            train_inner(config, output, source, device, rank, world, monitor)
    finally:
        source.close()
        dist.destroy_process_group()


def train_inner(config, output, source, device, rank, world, monitor):
    # All three layouts instantiate the SAME parameters, in the SAME RNG order.
    # Only the query-channel tensor changes between arms.
    torch.manual_seed(config["seed"])
    model = PointTransformer(ModelConfig(**config["model"])).to(device)
    if not model.config.field_queries or model.config.objective not in {
        "regression",
        "absolute_ce",
    }:
        raise ValueError("Comparison requires field-query regression or absolute CE")
    vocab = (
        ChannelVocabulary(config["vocabulary_bounds"], model.config.bins).to(device)
        if model.config.objective == "absolute_ce"
        else None
    )
    initial_digest = parameter_digest(model)
    categorical_target = config.get("categorical_target", "hard")
    if categorical_target not in {"hard", "two_hot"} or (
        vocab is None and categorical_target != "hard"
    ):
        raise ValueError("Unsupported categorical training target")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
        fused=True,
    )
    seed = config["seed"] + rank * 1009
    rng = np.random.default_rng(seed)
    step, stream_state, state = 0, None, None
    best = {"raw": math.inf, "ema": math.inf}
    latest = output / "latest.pt"
    if latest.exists():
        if not (
            int(os.environ.get("SLURM_RESTART_COUNT", "0"))
            or config.get("resume", False)
        ):
            raise FileExistsError(latest)
        state = torch.load(latest, map_location="cpu", mmap=True, weights_only=False)
        if state["config"] != config or state["source"] != source.receipt():
            raise ValueError("Configuration or data provenance changed on resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, best = state["step"], state["best"]
        saved = state["rank_states"][rank]
        rng.bit_generator.state = saved["rng"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
        stream_state = saved["stream"]
    ema = restore_ema(
        model, config["ema"], None if state is None else state["ema"], step
    )
    del state
    ddp = DistributedDataParallel(
        model, device_ids=[device.index], broadcast_buffers=False
    )
    stream = FrameStream(source, seed + 100, stream_state)
    accumulation = config["accumulation"]
    effective_batch = world * config["microbatch"] * accumulation
    run = None
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    provenance = dict(
        config=config,
        source=source.receipt(),
        initial_parameter_sha256=initial_digest,
        source_commit=os.environ.get("SKYLINE_COMMIT"),
        archive_sha256=os.environ.get("SKYLINE_ARCHIVE_SHA256"),
        job=os.environ.get("SLURM_JOB_ID"),
        restart=restart,
        resumed_step=step,
        parameters=sum(p.numel() for p in model.parameters()),
        gpu=torch.cuda.get_device_name(),
        effective_batch=effective_batch,
        initial_head_statistics=model.initial_head_statistics,
    )
    if rank == 0:
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
    started = time.monotonic()
    last_save = started
    complete = False

    def save_checkpoint():
        nonlocal last_save
        local = dict(
            rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device),
            stream=stream.state(),
        )
        states = [None] * world
        dist.all_gather_object(states, local)
        if rank == 0:
            save_atomic(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    config=config,
                    source=source.receipt(),
                    step=step,
                    best=best,
                    ema=ema.get_state(include_ema_params=True),
                    rank_states=states,
                ),
                latest,
            )
            emit(dict(event="checkpoint", step=step), output / "training.jsonl")
        dist.barrier()
        last_save = time.monotonic()

    def validate():
        metrics = {}
        for kind in ("raw", "ema"):
            with ema_weights(model, ema) if kind == "ema" else nullcontext():
                result = evaluate_comparison(
                    model, source, config, device, step, rank, world, vocab=vocab
                )
                if rank == 0:
                    (output / f"validation-{kind}-{step:07d}.json").write_text(
                        json.dumps(result, indent=2) + "\n"
                    )
                    score = (
                        result[
                            "temperature_full_grid_nll"
                            if config.get("temperature_loss_only", False)
                            else "full_grid_nll"
                        ]
                        if vocab is not None
                        else result["full_grid_nmse"]
                    )
                    if config.get("retain_validation_weights", False):
                        save_atomic(
                            dict(
                                model=model.state_dict(),
                                config=config,
                                step=step,
                                validation=result,
                                weights_kind=kind,
                                ema=ema.get_state(include_ema_params=False),
                            ),
                            output / f"weights-{kind}-{step:07d}.pt",
                        )
                    if score < best[kind]:
                        best[kind] = score
                        save_atomic(
                            dict(
                                model=model.state_dict(),
                                config=config,
                                step=step,
                                validation=result,
                                weights_kind=kind,
                                ema=ema.get_state(include_ema_params=False),
                            ),
                            output / f"best-{kind}.pt",
                        )
                    emit(
                        dict(
                            event="validation",
                            weights=kind,
                            step=step,
                            nmse=result["full_grid_nmse"],
                            nll=result.get("full_grid_nll"),
                            temperature_nmse=result["temperature_full_grid_nmse"],
                        ),
                        output / "training.jsonl",
                    )
                    metrics[f"validation/{kind}/full_grid_nmse"] = result[
                        "full_grid_nmse"
                    ]
                    for key in (
                        "full_grid_nll",
                        "temperature_full_grid_nll",
                        "probability_mean_full_grid_nmse",
                        "temperature_probability_mean_full_grid_nmse",
                    ):
                        if key in result:
                            metrics[f"validation/{kind}/{key}"] = result[key]
                    for field, value in result["per_field_nmse"].items():
                        metrics[f"validation/{kind}/{field}/full_grid_nmse"] = value
                    for row in result["per_channel"]:
                        for key, value in row.items():
                            if key not in {
                                "channel",
                                "channel_index",
                                "depth",
                                "depth_m",
                                "count",
                                "wet_fraction",
                            }:
                                metrics[f"validation/{kind}/{row['channel']}/{key}"] = (
                                    value
                                )
        if run:
            run.log(metrics, step=step)
        model.train()
        dist.barrier()

    try:
        if step == 0:
            # Initialization is recorded explicitly rather than extrapolated.
            validate()
            save_checkpoint()
        model.train()
        while step < config["max_updates"]:
            if "max_allocated_seconds" in config:
                allocated = float(os.environ["SKYLINE_PRIOR_SECONDS"]) + (
                    time.time() - float(os.environ["SKYLINE_ALLOCATION_START"])
                )
                exhausted = torch.tensor(
                    int(allocated >= config["max_allocated_seconds"]), device=device
                )
                dist.all_reduce(exhausted, op=dist.ReduceOp.MAX)
                if exhausted:
                    save_checkpoint()
                    raise RuntimeError(
                        "Pilot allocation budget exhausted; checkpoint preserved"
                    )
            failure = torch.tensor(int(monitor.reason is not None), device=device)
            dist.all_reduce(failure, op=dist.ReduceOp.MAX)
            if failure:
                raise RuntimeError("GPU telemetry failed on at least one rank")
            begin = time.monotonic()
            step += 1

            def next_batch():
                if stream.used >= config["samples_per_frame"]:
                    stream.advance()
                batch = sample_coverage(
                    source,
                    stream.frame,
                    rng,
                    config["microbatch"],
                    config["crop"],
                    config["coverage"],
                )
                stream.used += config["microbatch"]
                if config.get("temperature_loss_only", False):
                    if config["arm"] != "all_fields":
                        raise ValueError(
                            "Matched temperature loss needs all-field queries"
                        )
                    batch["loss_weight"] *= batch["query_field"] == 2
                return batch

            factor = learning_rate_factor(
                step,
                config["warmup"],
                config.get("schedule_updates", config["max_updates"]),
            )
            set_synchronized_learning_rate(
                optimizer, config["learning_rate"] * factor, device
            )
            optimizer.zero_grad(set_to_none=True)
            with device_batches(
                next_batch,
                config["microbatch"],
                accumulation,
                config.get("physical_microbatch", config["microbatch"]),
                device,
                config.get("prefetch_microbatches", False),
            ) as (get_batch, physical_accumulation):
                stats = accumulate_gradients(
                    ddp,
                    get_batch,
                    (
                        "comparison_two_hot"
                        if categorical_target == "two_hot"
                        else "comparison_ce"
                    )
                    if vocab is not None
                    else "comparison",
                    vocab,
                    device,
                    physical_accumulation,
                )
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            ema(model)
            torch.cuda.synchronize()
            if step == 1 or step % config["log_every"] == 0:
                stats = torch.cat((stats, norm.detach().reshape(1)))
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
                        ema_updates=int(ema.num_updates),
                        seconds=time.monotonic() - begin,
                        elapsed_seconds=time.monotonic() - started,
                        accepted_examples=step * effective_batch,
                        utilization=monitor.mean,
                        peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                    )
                    emit(record, output / "training.jsonl")
                    if run:
                        run.log({"train/" + k: v for k, v in record.items()}, step=step)
            validation_due = (
                step in config.get("validate_at", [1])
                or step % config["validate_updates"] == 0
                or step == config["max_updates"]
            )
            if validation_due:
                validate()
            # Only rank zero's timer controls this collective.
            save_due = torch.tensor(
                int(
                    validation_due
                    or time.monotonic() - last_save >= config["save_seconds"]
                ),
                device=device,
            )
            dist.broadcast(save_due, src=0)
            if save_due:
                save_checkpoint()
        if int(ema.num_updates) != step:
            raise RuntimeError("EMA count does not match optimizer updates")
        parameters = {"raw/" + name: value for name, value in model.named_parameters()}
        parameters.update(
            {
                "ema/" + name: value
                for name, value in ema.get_state(True)["ema_params"].items()
            }
        )
        replicas = verify_replicas(parameters, world)
        if not replicas["identical"]:
            raise RuntimeError("Raw/EMA DDP replicas differ")
        if rank == 0:
            (output / "replica-check.json").write_text(
                json.dumps(replicas, indent=2) + "\n"
            )
            (output / "COMPLETE.json").write_text(
                json.dumps(
                    dict(
                        step=step,
                        effective_batch=effective_batch,
                        accepted_examples=step * effective_batch,
                        ema_updates=int(ema.num_updates),
                        replicas_agree=True,
                        qualification=config["qualification"],
                    )
                )
                + "\n"
            )
        complete = True
    finally:
        stream.close()
        if run:
            run.finish(exit_code=0 if complete else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
