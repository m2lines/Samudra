# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded parallel CPU preparation with speculative, non-consuming lookahead."""

import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


def next_model_indices(rng, step, main_end, train_size, model_size):
    """Look ahead through the A schedule without changing the checkpointed RNG."""
    probe = np.random.default_rng()
    probe.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    for update in range(step, main_end):
        source = "obs" if update % 2 == 0 else "om4"
        indices = []
        for _ in range(8):
            probe.random()  # Original unused task draw is part of the contract.
            indices.append(
                int(probe.integers(train_size if source == "obs" else model_size))
            )
        if source == "om4":
            return indices
    return None


class ModelPrefetch:
    """At most eight CPU windows, four loaders; no worker touches CUDA or RNG.

    Only the consumed RNG is checkpointed. Outstanding work is disposable and
    rebuilt from that RNG on resume. Results are consumed in sampled order,
    regardless of which worker finishes first.
    """

    def __init__(self, load, workers=4):
        self.load = load
        self.pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="om4-prefetch"
        )
        self.indices = None
        self.futures = []

    def prime(self, indices):
        if indices is None:
            return
        if self.indices is not None:
            if self.indices != indices:
                raise ValueError(
                    "Prefetched sample order differs from checkpointed RNG"
                )
            return
        if len(indices) != 8:
            raise ValueError("Prefetch requires the unchanged eight-example update")
        self.indices = list(indices)
        self.futures = [self.pool.submit(self.load, index) for index in indices]

    def take(self, position, index):
        if self.indices is None or self.indices[position] != index:
            raise ValueError("Training selected a different prefetched example")
        future = self.futures[position]
        value = future.result()  # Surface I/O/verification failures in training.
        self.futures[position] = None  # Release each CPU window after consumption.
        if position == 7:
            self.indices = None
            self.futures = []
        return value

    def reset(self):
        # Used on restore: do not let stale preparation contaminate replay.
        for future in self.futures:
            if future is not None:
                future.result()
        self.indices = None
        self.futures = []

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value
