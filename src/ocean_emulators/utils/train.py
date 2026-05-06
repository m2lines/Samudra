# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from itertools import tee
from pathlib import Path

import torch
from xarray_einstats.einops import rearrange  # noqa: F401

from ocean_emulators.datasets import InferenceDataset, RawTrainData
from ocean_emulators.utils.data import LoadStats


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def collate_raw_train_data(data: Sequence[RawTrainData]) -> RawTrainData:
    # Carry per-sample indices through the collate step so PCGB can look up
    # D_t per batch element. None on input → None on output (back-compat).
    if all(d.sample_indices is not None for d in data):
        batched_indices: torch.Tensor | None = torch.cat(
            [d.sample_indices for d in data]  # type: ignore[misc]
        )
    else:
        batched_indices = None

    batched_data = RawTrainData(data[0].dataset_id, sample_indices=batched_indices)
    assert all(d.dataset_id == batched_data.dataset_id for d in data), (
        "we don't support heterogenous batches yet"
    )

    steps = len(data[0].raw_data)
    for step in range(steps):
        input_ = torch.stack([d.raw_data[step][0] for d in data])
        boundary = torch.stack([d.raw_data[step][1] for d in data])
        label = torch.stack([d.raw_data[step][2] for d in data])
        batched_data.insert(input_, boundary, label)

    stats = LoadStats.accumulated(
        [d.load_stats for d in data if d.load_stats is not None]
    )
    batched_data.load_stats = stats

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
