# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""PyTorch batch construction shared by the CPU and GPU loading configs."""

import multiprocessing

import torch
from torch.utils.data import ConcatDataset, DataLoader

from samudra.datasets import (
    HostBatch,
    TorchBatchLoader,
    TorchTrainDataset,
    TrainBatchLoader,
    TrainingWindows,
)
from samudra.utils.samplers import BatchSchedule
from samudra.utils.train import collate_host_batches


def build_torch_batch_loader(
    windows: list[TrainingWindows],
    batch_sampler: BatchSchedule,
    device: torch.device,
    *,
    num_workers: int,
    persistent_workers: bool,
    pin_memory: bool,
    worker_seed: int,
    concurrent_compute: bool,
) -> TrainBatchLoader:
    """Build a PyTorch loader from explicit settings supplied by its config."""
    datasets = [
        TorchTrainDataset(window, concurrent_compute_=concurrent_compute)
        for window in windows
    ]

    host_data: torch.utils.data.Dataset[HostBatch] = ConcatDataset(datasets)
    dataloader = DataLoader(
        host_data,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        pin_memory=pin_memory and device.type == "cuda",
        collate_fn=collate_host_batches,
        multiprocessing_context=(
            multiprocessing.get_context("spawn") if num_workers > 0 else None
        ),
        generator=torch.Generator().manual_seed(worker_seed),
    )
    return TorchBatchLoader(dataloader, datasets, device)
