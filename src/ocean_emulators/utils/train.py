from itertools import tee
from pathlib import Path
from typing import Sequence, Tuple

import torch
import xarray as xr
from xarray_einstats.einops import rearrange  # noqa: F401

from ocean_emulators.constants import Example, Input, Prognostic
from ocean_emulators.datasets import InferenceDataset, TrainData


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def collate_train_data(data: Sequence[TrainData]) -> TrainData:
    num_prognostic_channels = data[0].num_prognostic_channels
    steps = len(data[0])

    batched_data = TrainData(num_prognostic_channels)

    for step in range(steps):
        input = torch.stack([d.get_input(step) for d in data])
        label = torch.stack([d.get_label(step) for d in data])
        batched_data.insert(input, label)

    return batched_data


def collate_om4(examples: Sequence[Example]) -> TrainData:
    """Combine several deferred Examples into single a `torch.Tensor` example pair."""
    inputs_batch = xr.concat([x for x, _ in examples], dim="step")
    labels_batch = xr.concat([y for _, y in examples], dim="step")

    inputs: Input = inputs_batch.transpose(
        "step",
        "window",
        "variable",
        "lat",
        "lon",
    ).compute()
    labels: Prognostic = labels_batch.transpose(
        "step",
        "window",
        "variable",
        "lat",
        "lon",
    ).compute()

    input_tensor = torch.from_numpy(inputs.to_numpy())
    labels_tensor = torch.from_numpy(labels.to_numpy())

    # TODO(#126): Remove TrainData interface (eventually)
    batch = TrainData(labels.shape[2])  # len(prognostic_vars)
    steps = input_tensor.shape[0]
    for step in range(steps):
        batch.insert(input_tensor[step], labels_tensor[step])

    return batch


def collate_inference_data(
    data: Sequence[InferenceDataset],
) -> Tuple[InferenceDataset, int]:
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
