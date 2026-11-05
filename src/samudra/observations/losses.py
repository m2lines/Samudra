# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Masked losses and deterministic artificial gaps for surface initialization."""

from collections.abc import Sequence

import torch
from numpy.typing import ArrayLike

from samudra.observations.constants import (
    HISTORY_STEPS,
    INITIAL_STATE_START,
    MODEL_VARIABLES,
    PROFILE_VARIABLES,
    SURFACE_NAMES,
)
from samudra.observations.sample_types import ObservationSample


def structured_visibility(validity: torch.Tensor, seed: int) -> torch.Tensor:
    """Persistent blocks plus an optional latitude band; no global RNG changes.

    validity is [batch, time, variable, y, x]. Artificial hiding changes only
    inputs. Keep the original validity for labels, including in naturally missing
    regions. Each variable gets three blocks, 8–20% of latitude and 12–32% of
    longitude, plus a 50% chance of a latitude band spanning 1/12 of the grid.
    Longitude blocks wrap, latitude bands do not.
    """
    if validity.ndim != 5 or validity.shape[2] != len(SURFACE_NAMES):
        raise ValueError("Expected two surface variables with batch/time dimensions")
    batch, _, channels, h, w = validity.shape
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    hidden = torch.zeros((batch, 1, channels, h, w), dtype=torch.bool)
    for b in range(batch):
        for c in range(channels):
            for _ in range(3):
                height = max(
                    1, int(h * (0.08 + 0.12 * torch.rand((), generator=generator)))
                )
                width = max(
                    1, int(w * (0.12 + 0.20 * torch.rand((), generator=generator)))
                )
                y = int(torch.randint(max(1, h - height + 1), (), generator=generator))
                x = int(torch.randint(w, (), generator=generator))
                columns = (torch.arange(width) + x) % w
                hidden[b, 0, c, y : y + height, columns] = True
            if bool(torch.rand((), generator=generator) < 0.5):
                height = max(1, h // 12)
                y = int(torch.randint(max(1, h - height + 1), (), generator=generator))
                hidden[b, 0, c, y : y + height] = True
    return validity.bool() & ~hidden.to(validity.device)


def corrupt_sample[Sample: ObservationSample](sample: Sample, seed: int) -> Sample:
    """Return separate inputs/targets; never mutate cached or supplied labels."""
    result = sample.copy()
    visible = structured_visibility(sample["validity"], seed)
    result["surface"] = torch.where(visible, sample["surface"], 0)
    result["validity"] = visible
    result["completion_target"] = sample["surface"][
        :, INITIAL_STATE_START:HISTORY_STEPS
    ]
    result["completion_valid"] = (
        sample["validity"][:, INITIAL_STATE_START:HISTORY_STEPS].bool()
        & ~visible[:, INITIAL_STATE_START:HISTORY_STEPS]
    )
    return result


def completion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    latitude: torch.Tensor | ArrayLike,
) -> torch.Tensor:
    """Return scalar mean MSE from [batch, frame, surface_channel, y, x] fields.

    ``valid`` has the field shape and ``latitude`` is the one-dimensional y axis.
    """
    valid = valid.bool()
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Completion predictions, labels and support must align")
    if not torch.isfinite(target[valid]).all():
        raise ValueError("Nonfinite completion label on declared valid support")
    area = torch.cos(torch.deg2rad(torch.as_tensor(latitude, device=prediction.device)))
    weight = valid * area[None, None, None, :, None]
    denominator = weight.sum((-2, -1))
    error = torch.where(valid, prediction.float() - target, 0).square()
    channel = (error * weight).sum((-2, -1)) / denominator.clamp_min(1e-12)
    present = denominator > 0
    if not present.any():
        return prediction.sum() * 0
    return channel[present].mean()


def channel_mse(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Reduce [..., channel, y, x] fields to [..., channel] using broadcast weights."""
    error = torch.where(weights > 0, prediction.float() - target.float(), 0).square()
    return (error * weights).sum((-2, -1)) / weights.sum((-2, -1)).clamp_min(1e-12)


def balanced_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    names: Sequence[str],
    interior_only: bool = False,
) -> torch.Tensor:
    """Return scalar MSE with equal weight per physical variable group."""
    per_channel = channel_mse(prediction, target, weights)
    groups = []
    variables = dict.fromkeys((*PROFILE_VARIABLES, *MODEL_VARIABLES, SURFACE_NAMES[1]))
    for variable in variables:
        indices = [
            i
            for i, name in enumerate(names)
            if (name == variable or name.startswith(variable + "_"))
            and not (interior_only and name in SURFACE_NAMES)
        ]
        if indices:
            groups.append(per_channel[..., indices].mean())
    if not groups:
        raise ValueError("No supported physical fields in the requested loss")
    return torch.stack(groups).mean()
