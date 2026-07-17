# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Construction boundary for model-facing training batch loaders."""

from multiprocessing.context import BaseContext

import torch
from torch.utils.data import ConcatDataset, DataLoader

from samudra.config import BaseDataLoadingConfig, RustDataLoadingConfig
from samudra.datasets import BatchLoader, HostBatch, TorchTrainDataset, TrainBatchLoader
from samudra.rust_data import (
    BatchSampler,
    CudaPrefetch,
    HostPrefetch,
    RustTrainDataLoader,
)
from samudra.utils.train import collate_host_batches


def build_train_batch_loader(
    datasets: list[TorchTrainDataset],
    batch_sampler: BatchSampler,
    device: torch.device,
    loading: BaseDataLoadingConfig,
    *,
    pin_memory: bool,
    multiprocessing_context: BaseContext | None,
    worker_seed: int,
) -> TrainBatchLoader:
    """Build one loader while keeping backend policy out of Trainer."""
    if isinstance(loading, RustDataLoadingConfig):
        prefetch = (
            CudaPrefetch()
            if device.type == "cuda" and loading.prefetch_to_device
            else HostPrefetch(pin_memory=pin_memory)
        )
        return RustTrainDataLoader(
            [dataset.shard for dataset in datasets],
            batch_sampler,
            device,
            prefetch_batches=loading.prefetch_batches,
            prefetch=prefetch,
        )

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
    return BatchLoader(dataloader, datasets, device)
