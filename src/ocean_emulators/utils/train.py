from collections.abc import Callable, Sequence, Sized
from itertools import tee
from pathlib import Path
from typing import cast

import torch
import torch.utils.data
from spdl.pipeline import Pipeline, PipelineBuilder  # type: ignore
from torch.utils.data import SequentialSampler
from xarray_einstats.einops import rearrange  # noqa: F401

from ocean_emulators.datasets import InferenceDataset, TrainData
from ocean_emulators.utils.data import LoadStats


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


def as_spdl_pipeline(
    dataset: torch.utils.data.Dataset["TrainData"],
    *,
    num_workers: int,
    batch_size: int,
    prefetch_factor: int = 2,
    drop_last: bool = False,
    sampler: torch.utils.data.Sampler | None = None,
    collate_fn: Callable = collate_train_data,
) -> Pipeline:
    """Migrates an existing torch.Dataset into a tunable SPDL data loader pipeline."""
    dataset_ = cast(Sized, dataset)
    if sampler is None:
        sampler = SequentialSampler(dataset_)

    samples = list(sampler)
    if drop_last:
        samples = samples[:-1]

    pipeline = (
        PipelineBuilder()
        .add_source(samples)
        .pipe(
            dataset.__getitem__,
            concurrency=num_workers,
            output_order="input",
        )
        .aggregate(batch_size)
        .pipe(collate_fn)
        .add_sink(prefetch_factor)
        .build(num_threads=num_workers)
    )

    # Our current data pipeline requires us to know the length up front.
    setattr(pipeline, "__len__", lambda: len(dataset_))

    return pipeline
