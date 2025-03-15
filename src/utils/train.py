from itertools import tee
from typing import Sequence, Tuple

import torch
import xarray as xr

from datasets import Example, InferenceDataset, TrainData


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def collate_train_data(data: Sequence[TrainData]) -> TrainData:
    output_channels = data[0].output_channels
    steps = len(data[0])

    batched_data = TrainData(output_channels)

    for step in range(steps):
        input = torch.stack([d.get_input(step) for d in data])
        label = torch.stack([d.get_label(step) for d in data])
        batched_data.insert(input, label)

    return batched_data


def collate_om4_batch(examples: Sequence[Example]) -> Example:
    inputs: list[xr.DataArray] = []
    labels: list[xr.DataArray] = []

    for input_, label in examples:
        inputs.append(
            input_.to_array(dim="vars")
            .transpose("step window time vars lat lon")
            .einops.rearrange(
                "step window time vars lat lon -> step window (time vars) lat lon",
            )
        )
        labels.append(
            label.to_array(dim="vars")
            .transpose("step window time vars lat lon")
            .einops.rearrange(
                "step window time vars lat lon -> step window (time vars) lat lon",
            )
        )

    input_batch: xr.DataArray = xr.concat(inputs, dim="step")
    labels_batch: xr.DataArray = xr.concat(labels, dim="step")

    input_tensor = torch.from_numpy(input_batch.to_numpy())
    labels_tensor = torch.from_numpy(labels_batch.to_numpy())

    return input_tensor, labels_tensor


def collate_inference_data(
    data: Sequence[InferenceDataset],
) -> Tuple[InferenceDataset, int]:
    # TODO: There is probably a better way to do inference batching
    assert len(data) == 1, "Inference batch size must be 1"
    return data[0][0], data[0][1]
