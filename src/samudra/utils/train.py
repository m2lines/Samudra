# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Sequence
from itertools import tee
from pathlib import Path

import torch
from xarray_einstats.einops import rearrange  # noqa: F401

from samudra.datasets import HostBatch, InferenceDataset
from samudra.utils.data import LoadStats


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def collate_host_batches(data: Sequence[HostBatch]) -> HostBatch:
    batched_data = HostBatch(data[0].dataset_id)
    assert all(d.dataset_id == batched_data.dataset_id for d in data), (
        "we don't support heterogenous batches yet"
    )

    steps = len(data[0].steps)
    for step in range(steps):
        input_ = torch.stack([d.steps[step][0] for d in data])
        boundary = torch.stack([d.steps[step][1] for d in data])
        label = torch.stack([d.steps[step][2] for d in data])
        batched_data.append(input_, boundary, label)

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
    _PERIODIC_CHECKPOINT_PATTERN = re.compile(r"^ckpt_(\d+)\.pt$")
    EMA_CHECKPOINT_NAME = "ema_ckpt.pt"

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "ckpt.pt"

    def latest_checkpoint_path_with_epoch(self, epoch: int) -> Path:
        return self.checkpoint_dir / f"ckpt_{epoch}.pt"

    def periodic_checkpoint_paths(self) -> dict[int, Path]:
        """Return periodic checkpoints indexed by epoch."""
        return {
            epoch: path
            for path in self.checkpoint_dir.iterdir()
            if path.is_file()
            and (epoch := self.periodic_checkpoint_epoch(path)) is not None
        }

    @classmethod
    def periodic_checkpoint_epoch(cls, checkpoint_path: Path) -> int | None:
        match = cls._PERIODIC_CHECKPOINT_PATTERN.fullmatch(checkpoint_path.name)
        return int(match.group(1)) if match is not None else None

    @property
    def best_inference_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best_inference_ckpt.pt"

    @property
    def ema_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / self.EMA_CHECKPOINT_NAME

    @property
    def best_validation_checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "best_validation_ckpt.pt"
