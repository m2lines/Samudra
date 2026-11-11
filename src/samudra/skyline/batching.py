# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Physical batches assembled from unchanged coverage-sampling draws."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch


@contextmanager
def device_batches(
    next_cpu_batch,
    logical_microbatch,
    accumulation,
    physical_microbatch,
    device,
    prefetch=False,
):
    """Prefetch only within an update; no future RNG/stream state escapes its end.

    Sampling always uses the original logical microbatch. Grouping never changes
    coverage candidate pools, draw order, or the effective optimizer batch.
    """
    if (
        physical_microbatch % logical_microbatch
        or physical_microbatch < logical_microbatch
    ):
        raise ValueError("Physical microbatch must group whole logical microbatches")
    group = physical_microbatch // logical_microbatch
    if accumulation % group:
        raise ValueError("Physical microbatch must divide the effective batch")
    count = accumulation // group

    def assemble():
        pieces = [next_cpu_batch() for _ in range(group)]
        if any(len(p["target"]) != logical_microbatch for p in pieces):
            raise ValueError("Unexpected logical microbatch size")
        result = (
            pieces[0]
            if group == 1
            else {k: torch.cat([p[k] for p in pieces]) for k in pieces[0]}
        )
        if prefetch and device.type == "cuda":
            result = {k: v.pin_memory() for k, v in result.items()}
        return result

    executor = ThreadPoolExecutor(max_workers=1) if prefetch else None
    pending = executor.submit(assemble) if executor else None
    consumed = 0

    def next_device_batch():
        nonlocal pending, consumed
        if consumed >= count:
            raise RuntimeError("Optimizer update requested too many batches")
        batch = pending.result() if pending is not None else assemble()
        consumed += 1
        pending = executor.submit(assemble) if executor and consumed < count else None
        return {k: v.to(device, non_blocking=prefetch) for k, v in batch.items()}

    try:
        yield next_device_batch, count
        if consumed != count:
            raise RuntimeError("Optimizer update did not consume its complete batch")
    finally:
        if executor:
            executor.shutdown(wait=True)
