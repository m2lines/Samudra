#!/usr/bin/env python3
"""Simulate N training ranks' replay reads (CPU only); srun one process per rank.

    srun -n 16 .venv/bin/python scripts/_bench/replay_read.py --stores S --out DIR
    .venv/bin/python scripts/_bench/replay_read.py --summarize DIR [DIR ...]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import queue
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

import ocean_emulators.datasets  # noqa: F401  # same blosc settings as training
from ocean_emulators.chunk_reader import GroupFrameReader
from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS
from ocean_emulators.tiling import block_tile_windows

# Decoded bytes per prognostic tile (205 channels, 752^2, float16).
TILE_BYTES = 205 * 752 * 752 * 2


def rank_and_world() -> tuple[int, int]:
    return int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))


def reader_loop(reader, blocks, args, rng, out: queue.Queue) -> None:
    """Produce one step's reads at a time, as `_load_raw_batch` does."""
    for step in range(args.steps):
        windows = blocks[rng.integers(len(blocks))]
        row = int(rng.integers(args.row_start, args.row_end - 1))
        seed = bool(rng.random() < args.seed_prob)
        start = time.perf_counter()
        frames = reader.prognostic.read(row + 1, windows)
        frames += reader.boundary.read(row, windows)
        if seed:
            frames += reader.prognostic.read(int(rng.integers(args.row_start, args.row_end)), windows)
        if args.bf16:
            # The trainer converts to the replay storage dtype on the reader thread.
            frames = [frame.to(torch.bfloat16) for frame in frames]
        out.put({"step": step, "load_s": time.perf_counter() - start, "seed": seed})


def run(args) -> None:
    rank, world = rank_and_world()
    stores = args.stores.split(",")
    # First half of the ranks on the first store, and so on: one store per host.
    store = stores[rank * len(stores) // world]
    windows = block_tile_windows(1)
    blocks = [windows[first : first + 9] for first in range(0, len(windows), 9)]
    reader = GroupFrameReader(
        store,
        windows=blocks[0],
        prognostic_var_names=PROGNOSTIC_VARS["all"],
        boundary_var_names=BOUNDARY_VARS["all_fw_noeta"],
        threads=args.threads,
    )
    if args.lockstep and world > 1:
        # Long timeout, so a multi-minute NFS stall is measured rather than fatal.
        timeout = datetime.timedelta(hours=2)
        dist.init_process_group("gloo", rank=rank, world_size=world, timeout=timeout)

    # Lookahead is horizon - 1, because the horizon counts the batch training now.
    ready: queue.Queue = queue.Queue(maxsize=max(1, args.horizon - 1))
    rng = np.random.default_rng(args.seed + 104729 * rank)
    if args.start_at:
        time.sleep(max(0.0, args.start_at - time.time()))
    threading.Thread(
        target=reader_loop, args=(reader, blocks, args, rng, ready), daemon=True
    ).start()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"rank{rank:03d}.jsonl", "w") as log:
        for _ in range(args.steps):
            start = time.perf_counter()
            item = ready.get()
            wait = time.perf_counter() - start
            if args.lockstep and world > 1:
                dist.barrier()
            sync = time.perf_counter() - start - wait
            time.sleep(args.step_seconds)
            item.update(
                rank=rank, store=store, wait_s=wait, sync_s=sync,
                step_s=time.perf_counter() - start, t=time.time(),
            )
            log.write(json.dumps(item) + "\n")
            log.flush()
    if dist.is_initialized():
        dist.destroy_process_group()


def stats(values: list[float]) -> str:
    if not values:
        return "-"
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return (
        f"mean {statistics.fmean(values):6.2f}  med {statistics.median(values):6.2f}  "
        f"p95 {pick(0.95):6.1f}  p99 {pick(0.99):6.1f}  max {max(values):7.1f}"
    )


def summarize(dirs: list[str], warmup: int) -> None:
    for directory in dirs:
        rows = [
            json.loads(line)
            for path in sorted(Path(directory).glob("rank*.jsonl"))
            for line in open(path)
        ]
        rows = [row for row in rows if row["step"] >= warmup]
        if not rows:
            print(f"{directory}: no measured steps")
            continue
        ranks = {row["rank"] for row in rows}
        steps = {row["step"] for row in rows}
        # The step time every rank pays in lockstep is the slowest rank's.
        per_step = {}
        for row in rows:
            per_step[row["step"]] = max(per_step.get(row["step"], 0.0), row["step_s"])
        frames = sum(9 * (2 if row["seed"] else 1) for row in rows)
        wall = max(row["t"] for row in rows) - min(row["t"] for row in rows)
        stores = sorted({row["store"] for row in rows})
        print(f"\n{Path(directory).name}: {len(ranks)} rank(s), {len(steps)} measured steps, stores {stores}")
        print(f"  load  {stats([row['load_s'] for row in rows])}")
        print(f"  wait  {stats([row['wait_s'] for row in rows])}")
        print(f"  step  {stats(list(per_step.values()))}  (slowest rank per step)")
        stalled = sum(1 for value in per_step.values() if value > 1.5 * statistics.median(per_step.values()))
        print(f"  {stalled}/{len(per_step)} steps >1.5x median; decoded {frames * TILE_BYTES / 1e9 / max(wall, 1e-9):.2f} GB/s aggregate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stores", help="comma-separated; ranks split evenly across them")
    parser.add_argument("--out", help="directory for rank*.jsonl")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--step-seconds", type=float, default=7.5, help="simulated compute per step")
    parser.add_argument("--threads", type=int, default=5, help="chunk read threads (face_parallel.read_threads)")
    parser.add_argument("--horizon", type=int, default=3, help="replay prefetch horizon")
    parser.add_argument("--seed-prob", type=float, default=0.03, help="extra seed read per step (cap + refresh)")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-end", type=int, default=8808, help="store rows in train_time")
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--start-at", type=float, default=0.0, help="epoch seconds; all ranks start together")
    parser.add_argument("--no-lockstep", dest="lockstep", action="store_false")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--summarize", nargs="+", metavar="DIR")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize, args.warmup)
    else:
        run(args)


if __name__ == "__main__":
    main()
