# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import threading

import numpy as np
import pytest

from samudra.experiments.monthly_prefetch import ModelPrefetch, next_model_indices


def test_lookahead_preserves_rng_and_matches_consumed_stream_through_cooldown():
    rng = np.random.default_rng(1729)
    for step in range(12):
        state = copy.deepcopy(rng.bit_generator.state)
        predicted = next_model_indices(rng, step, 10, 243, 409)
        assert rng.bit_generator.state == state
        indices = []
        for _ in range(8):
            rng.random()
            indices.append(
                int(rng.integers(243 if step % 2 == 0 or step >= 10 else 409))
            )
        if step % 2 and step < 10:
            assert predicted == indices
        if step >= 10:
            assert predicted is None


def test_parallel_workers_preserve_order_and_reset_discards_speculative_work():
    barrier = threading.Barrier(4, timeout=5)

    def load(index):
        barrier.wait()  # Fails if preparation is serial rather than four-worker.
        return index * 2

    loader = ModelPrefetch(load, workers=4)
    try:
        indices = list(range(8))
        loader.prime(indices)
        loader.prime(indices)  # Idempotent, no second batch.
        assert [loader.take(i, index) for i, index in enumerate(indices)] == [
            2 * i for i in indices
        ]
        loader.prime(indices)
        loader.reset()
        loader.prime(indices[::-1])
        assert [loader.take(i, index) for i, index in enumerate(indices[::-1])] == [
            2 * i for i in indices[::-1]
        ]
    finally:
        loader.close()


def test_corruption_and_sample_order_errors_are_not_hidden():
    def fail(index):
        raise ValueError("checksum failed")

    loader = ModelPrefetch(fail)
    try:
        loader.prime(list(range(8)))
        with pytest.raises(ValueError, match="different prefetched"):
            loader.take(0, 7)
        with pytest.raises(ValueError, match="checksum failed"):
            loader.take(0, 0)
    finally:
        loader.close()
