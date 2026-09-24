# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Construction boundary for model-facing training batch loaders."""

from multiprocessing.context import BaseContext

import torch
from torch.utils.data import ConcatDataset, DataLoader

from samudra.config import BaseDataLoadingConfig
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
    loading: BaseDataLoadingConfig,
    *,
    pin_memory: bool,
    multiprocessing_context: BaseContext | None,
    worker_seed: int,
    concurrent_compute: bool,
) -> TrainBatchLoader:
    """Build one loader while keeping backend policy out of Trainer."""
    datasets = [
        TorchTrainDataset(window, concurrent_compute_=concurrent_compute)
        for window in windows
    ]

    host_data: torch.utils.data.Dataset[HostBatch] = ConcatDataset(datasets)
    dataloader = DataLoader(
        host_data,
        batch_sampler=batch_sampler,
        num_workers=loading.num_pytorch_workers(),
        persistent_workers=(
            loading.persistent_pytorch_workers() and loading.num_pytorch_workers() > 0
        ),
        pin_memory=pin_memory,
        collate_fn=collate_host_batches,
        multiprocessing_context=multiprocessing_context,
        generator=torch.Generator().manual_seed(worker_seed),
    )
    return TorchBatchLoader(dataloader, datasets, device)
