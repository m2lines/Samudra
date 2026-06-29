#!/usr/bin/env python3
"""Measure replay-buffer read, GPU, and full-step timing.

Run via torchrun (Trainer needs a (possibly trivial) distributed context):

    torchrun --standalone --nnodes=1 --nproc_per_node=1 \
        scripts/measure_replay_timing.py configs/samudra_llc/train_replay.yaml \
        --read-iters 60 --gpu-iters 60 --step-iters 200 --read-threads 1,2,4,6 \
        -- --batch_size 1

Key design points:
  * T_step is measured TWICE:
      - throughput mode: NO internal CUDA syncs, one sync at the end -> real ms/step
        (this is the number that reflects prefetch/compute overlap).
      - seam mode: WITH internal syncs to attribute time per seam. This PERTURBS the
        pipeline (kills overlap) and will over-report total; use only for attribution.
  * T_gpu uses a real TRAIN batch (with a valid label), cloned and held static.
  * A parallel-read scaling test reveals whether reads release the GIL (i.e. whether a
    ThreadPoolExecutor will actually help) and estimates threads-needed to feed the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

from ocean_emulators.config import TrainConfig
from ocean_emulators.train import Trainer, _ReplayPrefetchPipeline
from ocean_emulators.utils.logging import handle_logging


@dataclass
class TimerStats:
    name: str
    values_ms: list[float]

    def summary(self) -> dict[str, float | int | str]:
        v = self.values_ms
        if not v:
            return {"name": self.name, "count": 0}
        s = sorted(v)
        return {
            "name": self.name,
            "count": len(v),
            "mean_ms": statistics.fmean(v),
            "median_ms": statistics.median(v),
            "min_ms": s[0],
            "max_ms": s[-1],
            "p90_ms": percentile(s, 0.90),
            "p95_ms": percentile(s, 0.95),
            "total_ms": sum(v),
        }


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[idx]


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clone_train_data(trainer: Trainer, prepared):
    data = prepared.data
    cloned = type(data)(data.num_prognostic_channels)
    for input_tensor, label in data.values():
        cloned.append(input_tensor.detach().clone(), label.detach().clone())
    cloned.load_stats = data.load_stats
    cloned.source_indices = list(data.source_indices)
    return cloned


# --------------------------------------------------------------------------- #
# T_read: single-thread real read path
# --------------------------------------------------------------------------- #
def _do_one_read(trainer: Trainer, cursor) -> None:
    dataset = trainer.train_datasets[cursor.dataset_index]
    dataset.get_raw_replay_train_transition(
        dataset_index=cursor.dataset_index,
        source_index=cursor.source_index,
        lead_step=cursor.lead_step,
    )


def measure_read_single(trainer: Trainer, iters: int, warmup: int) -> TimerStats:
    values: list[float] = []
    cursors = [trainer.sample_replay_seed_cursor() for _ in range(iters + warmup)]
    for i, cursor in enumerate(cursors):
        start = time.perf_counter()
        _do_one_read(trainer, cursor)
        dt = (time.perf_counter() - start) * 1000.0
        if i >= warmup:
            values.append(dt)
    return TimerStats("T_read_single_thread", values)


def measure_read_parallel(
    trainer: Trainer, thread_counts: list[int], reads_per_count: int, warmup: int
) -> dict[str, dict]:
    """Measure aggregate read throughput at various thread counts.

    If reads release the GIL, effective per-read time drops as threads increase.
    """
    results: dict[str, dict] = {}
    single_eff = None
    for n in thread_counts:
        cursors = [
            trainer.sample_replay_seed_cursor()
            for _ in range(reads_per_count + n * warmup)
        ]
        # warmup
        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(lambda c: _do_one_read(trainer, c), cursors[: n * warmup]))
            work = cursors[n * warmup :]
            start = time.perf_counter()
            list(ex.map(lambda c: _do_one_read(trainer, c), work))
            wall = (time.perf_counter() - start) * 1000.0
        eff_per_read = wall / max(1, len(work))
        if n == thread_counts[0]:
            single_eff = eff_per_read
        results[f"threads_{n}"] = {
            "threads": n,
            "num_reads": len(work),
            "wall_ms": wall,
            "effective_ms_per_read": eff_per_read,
            "speedup_vs_fewest": (single_eff / eff_per_read) if single_eff else 1.0,
        }
    return results


# --------------------------------------------------------------------------- #
# T_gpu: forward+backward on a cached static TRAIN batch (uses CUDA events)
# --------------------------------------------------------------------------- #
def build_static_train_data(trainer: Trainer, max_lead_steps: int):
    """Pull one real prepared TRAIN batch from the pipeline and clone it."""
    pipeline = _ReplayPrefetchPipeline(
        trainer, start_batch_in_epoch=0, total_batches=2, max_lead_steps=max_lead_steps
    )
    iterator = iter(pipeline)
    prepared = next(iterator)
    data = clone_train_data(trainer, prepared)
    # Release reservation without advancing buffer entries (no pred applied).
    try:
        pipeline.complete(prepared)
    except Exception:
        pass
    pipeline.close()
    return data


def measure_gpu(trainer: Trainer, static_data, *, iters: int, warmup: int) -> TimerStats:
    device = trainer.device
    trainer.model.train(True)
    use_events = device.type == "cuda"
    values: list[float] = []

    for i in range(iters + warmup):
        trainer.optimizer.zero_grad(set_to_none=True)
        if use_events:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
        else:
            t0 = time.perf_counter()

        outputs = trainer.model(static_data)
        pred = outputs[0]
        label = static_data.get_label(0)
        loss = torch.mean(trainer.loss_fn(pred, label))
        loss.backward()

        if use_events:
            end_ev.record()
            torch.cuda.synchronize(device)
            dt = start_ev.elapsed_time(end_ev)
        else:
            dt = (time.perf_counter() - t0) * 1000.0
        if i >= warmup:
            values.append(dt)

    trainer.optimizer.zero_grad(set_to_none=True)
    return TimerStats("T_gpu_cached_fwd_bwd", values)


# --------------------------------------------------------------------------- #
# T_step: throughput (no internal sync) + seam attribution (perturbing)
# --------------------------------------------------------------------------- #
def replay_forward_backward(trainer: Trainer, data, *, loss_scale: int = 1) -> torch.Tensor:
    outputs = trainer.model(data)
    pred = outputs[0]
    label = data.get_label(0)
    loss = torch.mean(trainer.loss_fn(pred, label))
    (loss / loss_scale).backward()
    return pred


def optimizer_micro_step(trainer: Trainer) -> None:
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)


def measure_step_throughput(
    trainer: Trainer, *, iters: int, warmup: int, max_lead_steps: int
) -> TimerStats:
    """Real ms/step: NO internal syncs. One sync after warmup, one at the end."""
    device = trainer.device
    total = iters + warmup
    pipeline = _ReplayPrefetchPipeline(
        trainer, start_batch_in_epoch=0, total_batches=total, max_lead_steps=max_lead_steps
    )
    iterator = iter(pipeline)
    accum = max(1, trainer.gradient_accumulation_steps)
    trainer.model.train(True)
    trainer.optimizer.zero_grad(set_to_none=True)

    # Run warmup, then sync, then time the measured region as a whole.
    measured_start = None
    try:
        for i in range(total):
            if i == warmup:
                sync_if_cuda(device)  # drain warmup before timing
                measured_start = time.perf_counter()
            prepared = next(iterator)
            pred = replay_forward_backward(
                trainer, prepared.data, loss_scale=accum
            ).detach()
            trainer.apply_replay_prefetch_updates(prepared, pred)
            pipeline.complete(prepared)
            if (i + 1) % accum == 0:
                optimizer_micro_step(trainer)
        sync_if_cuda(device)  # drain everything before stopping clock
        measured_wall = (time.perf_counter() - measured_start) * 1000.0
    finally:
        pipeline.close()
        trainer.optimizer.zero_grad(set_to_none=True)

    per_step = measured_wall / max(1, iters)
    # Store as a single-element "stats" plus expose per-step via mean.
    return TimerStats("T_step_throughput_real", [per_step] * iters)


def measure_step_seams(
    trainer: Trainer, *, iters: int, warmup: int, cadence: int, max_lead_steps: int
) -> dict[str, TimerStats]:
    """PERTURBING: syncs between seams to attribute time. Over-reports total."""
    device = trainer.device
    total = iters + warmup
    pipeline = _ReplayPrefetchPipeline(
        trainer, start_batch_in_epoch=0, total_batches=total, max_lead_steps=max_lead_steps
    )
    iterator = iter(pipeline)
    accum = max(1, trainer.gradient_accumulation_steps)
    stats = {
        "seam_total_perturbed": [],
        "seam_next_prepared": [],
        "seam_gpu_fwd_bwd": [],
        "seam_replay_update": [],
        "seam_optimizer": [],
    }
    trainer.model.train(True)
    trainer.optimizer.zero_grad(set_to_none=True)
    try:
        for i in range(total):
            sync_if_cuda(device)
            t_step = time.perf_counter()

            t = time.perf_counter()
            prepared = next(iterator)
            sync_if_cuda(device)
            next_ms = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            pred = replay_forward_backward(trainer, prepared.data, loss_scale=accum).detach()
            sync_if_cuda(device)
            gpu_ms = (time.perf_counter() - t) * 1000.0

            t = time.perf_counter()
            trainer.apply_replay_prefetch_updates(prepared, pred)
            pipeline.complete(prepared)
            sync_if_cuda(device)
            upd_ms = (time.perf_counter() - t) * 1000.0

            opt_ms = 0.0
            if (i + 1) % accum == 0:
                t = time.perf_counter()
                optimizer_micro_step(trainer)
                sync_if_cuda(device)
                opt_ms = (time.perf_counter() - t) * 1000.0

            total_ms = (time.perf_counter() - t_step) * 1000.0
            if i >= warmup:
                stats["seam_total_perturbed"].append(total_ms)
                stats["seam_next_prepared"].append(next_ms)
                stats["seam_gpu_fwd_bwd"].append(gpu_ms)
                stats["seam_replay_update"].append(upd_ms)
                stats["seam_optimizer"].append(opt_ms)
            if cadence > 0 and (i + 1) % cadence == 0:
                print(json.dumps({
                    "iteration": i + 1,
                    "phase": "warmup" if i < warmup else "measure",
                    "total_ms": total_ms, "next_ms": next_ms,
                    "gpu_ms": gpu_ms, "update_ms": upd_ms, "opt_ms": opt_ms,
                }, sort_keys=True), flush=True)
    finally:
        pipeline.close()
        trainer.optimizer.zero_grad(set_to_none=True)
    return {k: TimerStats(k, v) for k, v in stats.items()}


# --------------------------------------------------------------------------- #
def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config")
    p.add_argument("--read-iters", type=int, default=40)
    p.add_argument("--gpu-iters", type=int, default=40)
    p.add_argument("--step-iters", type=int, default=150)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--cadence", type=int, default=50)
    p.add_argument("--read-threads", type=str, default="1,2,4,6",
                   help="Comma-separated thread counts for the parallel-read scaling test.")
    p.add_argument("--max-lead-steps", type=int, default=None)
    p.add_argument("overrides", nargs="*")
    args, unknown = p.parse_known_args()
    overrides = list(args.overrides) + unknown
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    return args, overrides


def main() -> None:
    args, user_overrides = parse_args()
    safe_overrides = [
        "--backend", "cuda",
        "--experiment.wandb.mode", "disabled",
        "--replay.checkpoint_buffer", "false",
        "--emergency_checkpoint_interval_minutes", "0",
    ]
    cfg = TrainConfig.from_yaml_and_cli([args.config, *safe_overrides, *user_overrides])
    cfg.experiment.name = f"{cfg.experiment.name}:timing"
    # Ensure the experiment output dir exists before logging opens a FileHandler there.
    from pathlib import Path
    Path(cfg.experiment.output_dir).mkdir(parents=True, exist_ok=True)
    handle_logging(cfg.debug, cfg.experiment.output_dir)

    trainer = Trainer(cfg)
    max_lead = args.max_lead_steps or trainer.replay_cfg.max_lead_steps[0]
    thread_counts = [int(x) for x in args.read_threads.split(",") if x.strip()]

    try:
        print("Initializing data loaders + replay buffer...", flush=True)
        # Mirror the training loop's replay-mode setup order: data loaders first
        # (this populates trainer.train_datasets), then the replay buffer.
        cur_temporal_stride = trainer.get_current_temporal_stride(trainer.start_epoch)
        trainer.temporal_stride = cur_temporal_stride
        trainer.init_data_loaders(
            max(trainer.replay_cfg.max_lead_steps),
            cur_temporal_stride,
        )
        trainer.init_replay_buffer()
        print(json.dumps({
            "device": str(trainer.device),
            "batch_size": trainer.batch_size,
            "gradient_accumulation_steps": trainer.gradient_accumulation_steps,
            "buffer_size": trainer.replay_cfg.buffer_size,
            "prefetch_horizon": trainer.replay_prefetch_horizon(),
            "max_lead_steps": max_lead,
        }, sort_keys=True), flush=True)

        print("Measuring T_read (single thread)...", flush=True)
        read_stats = measure_read_single(trainer, args.read_iters, args.warmup)

        print("Measuring read scaling across threads...", flush=True)
        read_scaling = measure_read_parallel(
            trainer, thread_counts, reads_per_count=max(args.read_iters, 24),
            warmup=2,
        )

        print("Preparing cached static TRAIN batch for T_gpu...", flush=True)
        static_data = build_static_train_data(trainer, max_lead)
        print("Measuring T_gpu...", flush=True)
        gpu_stats = measure_gpu(trainer, static_data, iters=args.gpu_iters, warmup=args.warmup)

        print("Measuring T_step (throughput, no internal sync)...", flush=True)
        step_throughput = measure_step_throughput(
            trainer, iters=args.step_iters, warmup=args.warmup, max_lead_steps=max_lead
        )

        print("Measuring T_step seams (perturbing, attribution only)...", flush=True)
        seam_stats = measure_step_seams(
            trainer, iters=args.step_iters, warmup=args.warmup,
            cadence=args.cadence, max_lead_steps=max_lead,
        )

        # ---- derived analysis ----
        t_read = read_stats.summary().get("median_ms", float("nan"))
        t_gpu = gpu_stats.summary().get("median_ms", float("nan"))
        t_step = step_throughput.summary().get("mean_ms", float("nan"))
        overlap_eff = (t_gpu / t_step) if t_step and t_step == t_step else float("nan")
        # threads needed so parallel reads keep up with GPU compute:
        threads_needed = (
            math.ceil((t_read * trainer.batch_size) / t_gpu)
            if t_gpu and t_gpu > 0 else float("nan")
        )
        verdict = (
            "GPU-bound (good)" if t_step <= 1.2 * t_gpu
            else "DATA/OVERHEAD-bound (fix I/O or overlap)"
        )

        summary = {
            "T_read_single": read_stats.summary(),
            "read_scaling": read_scaling,
            "T_gpu": gpu_stats.summary(),
            "T_step_throughput": step_throughput.summary(),
            **{k: v.summary() for k, v in seam_stats.items()},
            "derived": {
                "t_read_median_ms": t_read,
                "t_gpu_median_ms": t_gpu,
                "t_step_real_ms": t_step,
                "overlap_efficiency_gpu_over_step": overlap_eff,
                "threads_needed_to_feed_gpu": threads_needed,
                "verdict": verdict,
            },
        }
        print("SUMMARY")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        trainer.finish()


if __name__ == "__main__":
    main()