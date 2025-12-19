from collections.abc import Sequence
from itertools import tee
from pathlib import Path

import torch
from xarray_einstats.einops import rearrange  # noqa: F401

from ocean_emulators.datasets import (
    InferenceDataset,
    RawMultiscaleTrainData,
    RawTrainData,
)
from ocean_emulators.utils.data import LoadStats


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def collate_raw_train_data(data: Sequence[RawTrainData]) -> RawTrainData:
    batched_data = RawTrainData(data[0].dataset_id)
    assert all(d.dataset_id == batched_data.dataset_id for d in data), (
        "we don't support heterogenous batches yet"
    )

    steps = len(data[0].raw_data)
    for step in range(steps):
        all_prognostic = torch.stack([d.raw_data[step][0] for d in data])
        all_boundary = torch.stack([d.raw_data[step][1] for d in data])
        batched_data.insert(all_prognostic, all_boundary)

    stats = LoadStats.accumulated(
        [d.load_stats for d in data if d.load_stats is not None]
    )
    batched_data.load_stats = stats

    return batched_data


def collate_raw_multiscale_train_data(
    data: Sequence[RawMultiscaleTrainData],
) -> RawMultiscaleTrainData:
    # When using MultiscaleGroupedBatchSampler, items in a batch have the same multiplex
    # position (idx % multiplex_size) but not necessarily the same exact index.
    # For example, indices [0, 4, 8] all have multiplex position 0 when multiplex_size=4.
    # We use data[0].index as the representative since all items map to the same position.
    batched_data = RawMultiscaleTrainData(data[0].dataset_id, {}, data[0].index)

    # Collect all dataset indices present in any batch item
    all_indices: set[int] = {k for d in data for k in d.datasets.keys()}

    # For each index, collate the RawTrainData from all batch items that have it
    for idx in sorted(all_indices):
        tds = [d.datasets[idx] for d in data if idx in d.datasets]
        if tds:
            batched_data.datasets[idx] = collate_raw_train_data(tds)

    return batched_data


def collate_inference_data(
    data: Sequence[InferenceDataset],
) -> tuple[InferenceDataset, int]:
    # TODO: There is probably a better way to do inference batching
    assert len(data) == 1, "Inference batch size must be 1"
    return data[0][0], data[0][1]


class CheckpointPaths:
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "ckpt.pt"

    def latest_checkpoint_path_with_epoch(self, epoch: int) -> Path:
        return self.checkpoint_dir / f"ckpt_{epoch}.pt"

    @property
    def best_inference_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best_inference_ckpt.pt"

    @property
    def ema_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "ema_ckpt.pt"

    @property
    def best_validation_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best_validation_ckpt.pt"
