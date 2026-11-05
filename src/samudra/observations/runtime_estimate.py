# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compute run-time ranges from qualification receipts, excluding queue delay."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def estimate(
    receipt,
    main_updates=4000,
    cooldown_updates=1000,
    *,
    main_start=0,
    validation_freq=250,
    save_freq=25,
):
    """Include validation and checkpoint work; report the observed timing range."""
    records = receipt["training_records"]
    distributed_receipt = "world_size" in receipt
    timing_records = (
        [r for r in records if r.get("stage") == "steady" and r["updates"] >= 25]
        if distributed_receipt
        else records
    )
    sources = list(receipt["exposure"])
    times = {}
    waits = {}
    for source in sources:
        # Drop first update (allocation/kernel warmup); deduplicate resume replay.
        unique = {
            int(r["updates"]): r
            for r in timing_records
            if f"train/{source}/seconds" in r and int(r["updates"]) > 1
        }
        if not unique:
            raise ValueError(f"No post-warmup timing for {source}")
        times[source] = [float(r[f"train/{source}/seconds"]) for r in unique.values()]
        waits[source] = [
            float(r[f"train/{source}/data_seconds"]) for r in unique.values()
        ]
    counts = {source: 0 for source in sources}
    cycle = ("obs", "om4") if "om4" in sources else ("obs",)
    for step in range(main_start, main_updates):
        counts[cycle[step % len(cycle)]] += 1
    counts["obs"] += cooldown_updates
    updates = sum(counts.values())
    validation = max(
        r["timing/validation_seconds"]
        for r in records
        if "timing/validation_seconds" in r
    )
    if distributed_receipt:
        main_endpoint = max(r["updates"] for r in timing_records)
        # Before the endpoint, each measured block writes only latest.pt/ckpt.pt.
        # Endpoint blocks also write milestone/main/best and cannot be treated
        # as one checkpoint. Use the slowest observed single-file write.
        checkpoint = max(
            r["timing/checkpoint_seconds"]
            for r in records
            if "timing/checkpoint_seconds" in r and r["updates"] < main_endpoint
        )
    else:
        checkpoint = max(
            receipt["checkpoint_seconds"],
            receipt.get("full_pool_checkpoint_seconds", 0),
        )
    # Legacy validation includes its improved-best save. The distributed loop
    # times it separately: budget a best save at every validation plus main,
    # milestone and endpoint copies, in addition to scheduled latest saves.
    extra_checkpoints = (
        math.ceil(updates / validation_freq) + 3 if distributed_receipt else 0
    )
    overhead = (
        math.ceil(updates / save_freq) + extra_checkpoints
    ) * checkpoint + math.ceil(updates / validation_freq) * validation
    low = (sum(counts[s] * min(times[s]) for s in sources) + overhead) / 3600
    high = (sum(counts[s] * max(times[s]) for s in sources) + overhead) / 3600
    mean = (
        sum(counts[s] * float(np.mean(times[s])) for s in sources) + overhead
    ) / 3600
    return {
        "gpu": receipt["gpu"],
        "effective_batch": 8,
        "world_size": receipt.get("world_size", 1),
        "source_updates": counts,
        "seconds_per_update": {
            s: {"min": min(t), "median": float(np.median(t)), "max": max(t)}
            for s, t in times.items()
        },
        "median_data_seconds": {s: float(np.median(t)) for s, t in waits.items()},
        "validation_seconds": validation,
        "checkpoint_seconds": checkpoint,
        "hours_observed_range": [low, high],
        "hours_observed_mean": mean,
        "hours_mean_with_30_percent_execution_allowance": 1.3 * mean,
        "hours_with_30_percent_execution_allowance": [1.3 * low, 1.3 * high],
        "allocated_gpu_hours_with_30_percent_execution_allowance": [
            1.3 * low * receipt.get("world_size", 1),
            1.3 * high * receipt.get("world_size", 1),
        ],
        "excluded": "queue delay and unknown preemption frequency; long-run replay mix may differ",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--main-updates", type=int, default=4000)
    parser.add_argument("--cooldown-updates", type=int, default=1000)
    parser.add_argument("--main-start", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            estimate(
                json.loads(args.receipt.read_text()),
                args.main_updates,
                args.cooldown_updates,
                main_start=args.main_start,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
