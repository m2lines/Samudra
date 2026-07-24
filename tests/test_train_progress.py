# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.datasets import ModelBatch
from samudra.utils.ctx import BatchGrid
from samudra.utils.train_progress import TrainBatchProgress, TrainProgress


def make_train_data(
    *,
    batch_size: int = 2,
    input_channels: int = 3,
    boundary_channels: int = 1,
    output_channels: int = 2,
    input_grid: tuple[int, int] = (3, 4),
    output_grid: tuple[int, int] = (5, 6),
    num_model_steps: int = 2,
) -> ModelBatch:
    ctx = BatchGrid(
        label_mask=torch.ones(output_channels, *output_grid, dtype=torch.bool),
        input_resolution_cpu=(torch.arange(input_grid[0]), torch.arange(input_grid[1])),
        output_resolution_cpu=(
            torch.arange(output_grid[0]),
            torch.arange(output_grid[1]),
        ),
    )
    model_batch = ModelBatch(input_channels, boundary_channels, ctx)
    for _ in range(num_model_steps):
        model_batch.append(
            torch.zeros(batch_size, input_channels, *input_grid),
            torch.zeros(batch_size, boundary_channels, *input_grid),
            torch.zeros(batch_size, output_channels, *output_grid),
        )
    return model_batch


def test_train_batch_progress_counts_global_training_units():
    batch_size = 2
    world_size = 4
    output_channels = 2
    input_grid = (3, 4)
    output_grid = (5, 6)
    num_model_steps = 2
    model_batch = make_train_data(
        batch_size=batch_size,
        output_channels=output_channels,
        input_grid=input_grid,
        output_grid=output_grid,
        num_model_steps=num_model_steps,
    )

    progress = TrainBatchProgress.from_train_data(model_batch, world_size)

    assert progress.sample_windows == batch_size * world_size
    assert progress.model_examples == batch_size * world_size * num_model_steps
    assert (
        progress.output_grid_cells
        == batch_size * world_size * num_model_steps * output_grid[0] * output_grid[1]
    )
    assert (
        progress.target_values
        == batch_size
        * world_size
        * num_model_steps
        * output_channels
        * output_grid[0]
        * output_grid[1]
    )
    assert progress.input_grid_lat == input_grid[0]
    assert progress.input_grid_lon == input_grid[1]
    assert progress.output_grid_lat == output_grid[0]
    assert progress.output_grid_lon == output_grid[1]


def test_train_batch_progress_throughput_metrics_use_batch_seconds():
    model_batch = make_train_data(batch_size=1, output_channels=1, output_grid=(2, 3))
    progress = TrainBatchProgress.from_train_data(model_batch, world_size=2)
    progress.batch_seconds = 0.5

    metrics = progress.to_throughput_metrics()

    assert metrics["throughput/model_examples_per_second"] == 8
    assert metrics["throughput/output_grid_cells_per_second"] == 48

    progress.batch_seconds = 0.0

    assert progress.to_throughput_metrics() == {}


def test_train_progress_batch_context_records_elapsed_progress(monkeypatch):
    model_batch = make_train_data(batch_size=1, output_channels=1, output_grid=(2, 3))
    progress = TrainProgress()
    times = iter([10.0, 12.5])
    monkeypatch.setattr(
        "samudra.utils.train_progress.time.perf_counter", lambda: next(times)
    )

    with progress.batch(
        model_batch, world_size=2, device=torch.device("cpu")
    ) as batch_progress:
        batch_progress.optimizer_stepped = True

    assert batch_progress.batch_seconds == 2.5
    assert progress.sample_windows_seen == 2
    assert progress.model_examples_seen == 4
    assert progress.output_grid_cells_seen == 24
    assert progress.target_values_seen == 24
    assert progress.optimizer_steps == 1
    assert progress.gpu_seconds == 5.0
