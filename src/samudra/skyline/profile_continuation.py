# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded, matched-stream profiling of continuation batch delivery."""

import argparse
import gc
import hashlib
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from numcodecs import blosc  # type: ignore[import-untyped]
from torch.nn.parallel import DistributedDataParallel
from torch.profiler import ProfilerActivity, profile, record_function

from .batching import device_batches
from .channel_vocabulary import ChannelVocabulary
from .coverage import sample_coverage
from .ema import restore_ema
from .model import ModelConfig, PointTransformer
from .multitask import ComparisonSource
from .run import FrameStream, save_atomic
from .utilization import GpuMonitor


def run(parent, out, physical, prefetch, updates):
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0)
    torch.cuda.reset_peak_memory_stats()
    state = torch.load(parent, map_location="cpu", mmap=True, weights_only=False)
    cfg = state["config"]
    torch.manual_seed(cfg["seed"])
    model = PointTransformer(ModelConfig(**cfg["model"])).to(device)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
        fused=True,
    )
    optimizer.load_state_dict(state["optimizer"])
    ema = restore_ema(model, cfg["ema"], state["ema"], state["step"])
    vocab = ChannelVocabulary(cfg["vocabulary_bounds"], model.config.bins).to(device)
    saved = state["rank_states"][0]
    rng = np.random.default_rng()
    rng.bit_generator.state = saved["rng"]
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    source = ComparisonSource(
        cfg["source"], cfg["readers"], cfg.get("reference_root"), arm=cfg["arm"]
    )
    counters: dict[str, float] = defaultdict(float)
    old_read, old_sample = source.read, source.sample

    def read(*args, **kwargs):
        begin = time.perf_counter()
        result = old_read(*args, **kwargs)
        counters["data_read_cpu_seconds"] += time.perf_counter() - begin
        return result

    def sample(*args, **kwargs):
        begin = time.perf_counter()
        result = old_sample(*args, **kwargs)
        counters["crop_extract_cpu_seconds"] += time.perf_counter() - begin
        return result

    source.read, source.sample = read, sample  # type: ignore[method-assign]
    stream = FrameStream(source, cfg["seed"] + 100, saved["stream"])
    ddp = DistributedDataParallel(model, device_ids=[0], broadcast_buffers=False)
    accepted = hashlib.sha256()

    def next_cpu():
        if stream.used >= cfg["samples_per_frame"]:
            begin = time.perf_counter()
            stream.advance()
            counters["frame_advance_wait_seconds"] += time.perf_counter() - begin
        begin = time.perf_counter()
        batch = sample_coverage(
            source, stream.frame, rng, cfg["microbatch"], cfg["crop"], cfg["coverage"]
        )
        counters["coverage_including_crop_cpu_seconds"] += time.perf_counter() - begin
        stream.used += cfg["microbatch"]
        if cfg.get("temperature_loss_only", False):
            batch["loss_weight"] *= batch["query_field"] == 2
        for key in (
            "origin",
            "row",
            "col",
            "depth",
            "target",
            "importance",
            "loss_weight",
        ):
            accepted.update(batch[key].numpy().tobytes())
        return batch

    telemetry = dict(
        minimum_percent=50, warmup_seconds=0, window_seconds=10, sample_seconds=1
    )
    timings = []
    losses = []
    kernels = []
    with GpuMonitor(
        out, torch.cuda.get_device_properties(device).uuid, 0, telemetry
    ) as monitor:
        started = time.perf_counter()
        for step in range(updates):
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prof = (
                profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                )
                if step == 10
                else None
            )
            if prof:
                prof.__enter__()
            with device_batches(next_cpu, 32, 8, physical, device, prefetch) as (
                get,
                count,
            ):
                for microstep in range(count):
                    with record_function("input_wait_and_H2D"):
                        batch = get()
                    with ddp.no_sync() if microstep + 1 < count else nullcontext():
                        with (
                            record_function("forward"),
                            torch.autocast("cuda", dtype=torch.bfloat16),
                        ):
                            logits = ddp(batch)
                        with record_function("FP32_twohot_loss"):
                            loss = (
                                (
                                    vocab.two_hot_loss(
                                        logits, batch["target"], batch["query_channel"]
                                    )
                                    * batch["loss_weight"]
                                ).sum(-1)
                                * batch["importance"]
                            ).mean()
                        assert (
                            logits.dtype == torch.bfloat16
                            and loss.dtype == torch.float32
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite loss")
                        with record_function("backward"):
                            (loss / count).backward()
            with record_function("clip_optimizer_EMA"):
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                ema(model)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - begin)
            losses.append(float(loss))
            if prof:
                prof.__exit__(None, None, None)
                prof.export_chrome_trace(str(out / "trace.json"))
                kernels = [
                    dict(
                        name=e.key,
                        cpu_total_us=e.cpu_time_total,
                        device_total_us=e.device_time_total,
                        calls=e.count,
                    )
                    for e in prof.key_averages()
                ]
                (out / "profiler.txt").write_text(
                    prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=50
                    )
                )
            if step % 20 == 0:
                print(out.name, step, timings[-1], float(norm), flush=True)
        elapsed = time.perf_counter() - started
        begin = time.perf_counter()
        save_atomic(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                ema=ema.get_state(include_ema_params=True),
            ),
            out / "probe-state.pt",
        )
        write_seconds = time.perf_counter() - begin
        result = dict(
            physical_microbatch=physical,
            prefetch=prefetch,
            updates=updates,
            elapsed_seconds=elapsed,
            median_update_seconds=float(
                np.median([v for i, v in enumerate(timings) if i > 5 and i != 10])
            ),
            timings=timings,
            counters=dict(counters),
            checkpoint_write_seconds=write_seconds,
            accepted_sha256=accepted.hexdigest(),
            final_rng=rng.bit_generator.state,
            final_stream=stream.state(),
            final_ema_updates=int(ema.num_updates),
            losses=losses,
            kernels=kernels,
            gpu=torch.cuda.get_device_name(),
            bf16=True,
            logits_dtype=str(logits.dtype),
            loss_dtype=str(loss.dtype),
            peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
            monitor_mean=monitor.mean,
        )
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    stream.close()
    source.close()
    del ddp, model, optimizer, ema, state
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=100)
    a = p.parse_args()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    blosc.set_nthreads(1)
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    try:
        for physical, prefetch in [(32, False), (128, False), (64, True)]:
            run(
                a.parent,
                a.output / f"micro{physical}-prefetch{int(prefetch)}",
                physical,
                prefetch,
                a.updates,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
