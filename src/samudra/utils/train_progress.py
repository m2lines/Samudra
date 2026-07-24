# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import dataclasses
import time
from collections.abc import Iterator
from typing import Any

import torch

from samudra.datasets import TrainData


@dataclasses.dataclass
class TrainBatchProgress:
    """Comparable progress and timing for one global train-loader batch.

    Counts are global across distributed workers so the resulting ``progress/*``
    metrics can be used as W&B x-axes when comparing runs with different batch
    sizes, GPU counts, gradient accumulation, or grid resolutions.

    A sample window is one dataset item from the train loader. A model example
    is one supervised autoregressive training step within that window, so a
    multi-step rollout contributes multiple model examples per sample window.
    Output grid cells are model examples multiplied by output-grid latitude and
    longitude cells. Target values are output grid cells multiplied by output
    channels.

    Input grid dimensions describe the spatial grid of the tensors presented to
    the model at each autoregressive step. They are not a count of input time
    levels or rollout steps. Output grid dimensions describe the spatial grid of
    each predicted/target field.

    ``TrainProgress`` accumulates the per-batch counts, optimizer update count,
    and summed batch wall time across workers. CUDA training synchronizes before
    reading batch wall time so ``progress/gpu_seconds`` and throughput metrics
    include queued GPU work.
    """

    sample_windows: int
    model_examples: int
    output_grid_cells: int
    target_values: int
    input_grid_lat: int
    input_grid_lon: int
    output_grid_lat: int
    output_grid_lon: int
    optimizer_stepped: bool = False
    batch_seconds: float = 0.0

    @classmethod
    def from_train_data(cls, data: TrainData, world_size: int) -> "TrainBatchProgress":
        label = data.get_label(0)
        (
            local_batch_size,
            output_channels,
            output_grid_lat,
            output_grid_lon,
        ) = label.shape
        input_grid_lat = data.ctx.input_resolution_cpu[0].shape[0]
        input_grid_lon = data.ctx.input_resolution_cpu[1].shape[0]

        sample_windows = local_batch_size * world_size
        model_examples = sample_windows * len(data)
        output_grid_cells = model_examples * output_grid_lat * output_grid_lon
        target_values = output_grid_cells * output_channels

        return cls(
            sample_windows=sample_windows,
            model_examples=model_examples,
            output_grid_cells=output_grid_cells,
            target_values=target_values,
            input_grid_lat=input_grid_lat,
            input_grid_lon=input_grid_lon,
            output_grid_lat=output_grid_lat,
            output_grid_lon=output_grid_lon,
        )

    def to_metrics(self) -> dict[str, int]:
        return {
            "progress/batch_sample_windows": self.sample_windows,
            "progress/batch_model_examples": self.model_examples,
            "progress/batch_output_grid_cells": self.output_grid_cells,
            "progress/batch_target_values": self.target_values,
            "progress/input_grid_lat": self.input_grid_lat,
            "progress/input_grid_lon": self.input_grid_lon,
            "progress/output_grid_lat": self.output_grid_lat,
            "progress/output_grid_lon": self.output_grid_lon,
        }

    def to_throughput_metrics(self) -> dict[str, float]:
        """Compute global per-batch throughput metrics when timing is available."""
        if self.batch_seconds <= 0:
            return {}
        return {
            "throughput/model_examples_per_second": (
                self.model_examples / self.batch_seconds
            ),
            "throughput/output_grid_cells_per_second": (
                self.output_grid_cells / self.batch_seconds
            ),
        }


@dataclasses.dataclass
class TrainProgress:
    """Cumulative training progress counters saved in checkpoints."""

    sample_windows_seen: int = 0
    model_examples_seen: int = 0
    output_grid_cells_seen: int = 0
    target_values_seen: int = 0
    optimizer_steps: int = 0
    gpu_seconds: float = 0.0

    @contextlib.contextmanager
    def batch(
        self, data: TrainData, *, world_size: int, device: torch.device
    ) -> Iterator[TrainBatchProgress]:
        batch = TrainBatchProgress.from_train_data(data, world_size)
        _synchronize_cuda_if_needed(device)
        batch_start_time = time.perf_counter()

        yield batch

        _synchronize_cuda_if_needed(device)
        batch.batch_seconds = time.perf_counter() - batch_start_time
        self.update(batch, world_size=world_size)

    def update(self, batch: TrainBatchProgress, *, world_size: int) -> None:
        self.sample_windows_seen += batch.sample_windows
        self.model_examples_seen += batch.model_examples
        self.output_grid_cells_seen += batch.output_grid_cells
        self.target_values_seen += batch.target_values
        if batch.optimizer_stepped:
            self.optimizer_steps += 1
        self.gpu_seconds += batch.batch_seconds * world_size

    def to_metrics(self) -> dict[str, int | float]:
        return {
            "progress/sample_windows_seen": self.sample_windows_seen,
            "progress/model_examples_seen": self.model_examples_seen,
            "progress/output_grid_cells_seen": self.output_grid_cells_seen,
            "progress/target_values_seen": self.target_values_seen,
            "progress/optimizer_steps": self.optimizer_steps,
            "progress/gpu_seconds": self.gpu_seconds,
        }

    def state_dict(self) -> dict[str, int | float]:
        return dataclasses.asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any] | None) -> "TrainProgress":
        if state is None:
            return cls()
        fields = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in state.items() if key in fields})


def _synchronize_cuda_if_needed(device: torch.device) -> None:
    """Synchronize CUDA kernels before reading wall-clock batch timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
