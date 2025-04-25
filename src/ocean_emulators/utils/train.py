from collections.abc import Sequence
from itertools import tee
from pathlib import Path

import torch
from xarray_einstats.einops import rearrange  # noqa: F401

from ocean_emulators.datasets import InferenceDataset, TrainData
from ocean_emulators.utils.logging import LoadStats


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def collate_train_data(data: Sequence[TrainData]) -> TrainData:
    num_prognostic_channels = data[0].num_prognostic_channels
    steps = len(data[0])

    batched_data = TrainData(num_prognostic_channels)

    stats = LoadStats.accumulated(
        [d.load_stats for d in data if d.load_stats is not None]
    )
    batched_data.load_stats = stats

    for step in range(steps):
        input = torch.stack([d.get_input(step) for d in data])
        label = torch.stack([d.get_label(step) for d in data])
        batched_data.insert(input, label)

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
