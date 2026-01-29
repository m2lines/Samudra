#!/usr/bin/env python3
"""Test for memory leaks during training iterations."""

import gc
import tracemalloc

import pytest


def get_memory_mb():
    """Get current memory usage in MB."""
    current, peak = tracemalloc.get_traced_memory()
    return current / 1024 / 1024, peak / 1024 / 1024


@pytest.mark.parametrize(
    "data_source,config_name",
    [("remote-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_memory_growth_across_epochs(trainer_pair, caplog):
    """Track memory growth across multiple training epochs."""
    import logging

    caplog.set_level(logging.WARNING)

    tracemalloc.start()

    _, trainer = trainer_pair
    num_epochs = 10

    memory_per_epoch = []

    for epoch in range(num_epochs):
        trainer.train_one_epoch(epoch)
        gc.collect()
        current, peak = get_memory_mb()
        memory_per_epoch.append(current)
        print(f"Epoch {epoch}: current={current:.2f} MB, peak={peak:.2f} MB")

    tracemalloc.stop()

    # Check if memory is growing linearly (indicating a leak)
    # Allow some growth but not unbounded
    first_epoch = memory_per_epoch[0]
    last_epoch = memory_per_epoch[-1]
    growth = last_epoch - first_epoch
    growth_percent = (growth / first_epoch) * 100 if first_epoch > 0 else 0

    print(f"\nMemory growth: {growth:.2f} MB ({growth_percent:.1f}%)")
    print(f"Memory per epoch: {memory_per_epoch}")

    # If memory grows more than 50% over 5 epochs, that's suspicious
    assert growth_percent < 50, (
        f"Memory grew {growth_percent:.1f}% over {num_epochs} epochs - possible leak!"
    )
