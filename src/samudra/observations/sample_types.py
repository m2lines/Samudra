# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Array and tensor contracts for prepared monthly samples and replay trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch


class RawSample(TypedDict, total=False):
    """Unnormalized arrays: time/channel/y/x; profiles use variable/depth/y/x."""

    surface: NDArray
    atmosphere: NDArray
    velocity: NDArray
    interior: NDArray
    ohc: NDArray
    month_weights: NDArray
    midpoints: NDArray
    origin: NDArray
    monthly_targets: NDArray
    month_matrix: NDArray
    months: NDArray
    interval_starts: NDArray
    interval_ends: NDArray
    start_lead: int
    end_lead: int
    next_start_lead: int


class ObservationSample(TypedDict, total=False):
    """Batched tensors: batch/time/channel/y/x; monthly targets replace time by month.

    Profile channels flatten variable/depth. Interval bounds and raw arrays remain
    on the CPU; lead indices count five-day forecast steps from the original origin.
    """

    completion_target: torch.Tensor
    completion_valid: torch.Tensor
    surface: torch.Tensor
    atmosphere: torch.Tensor
    contexts: torch.Tensor
    validity: torch.Tensor
    month_weights: torch.Tensor
    interior: torch.Tensor
    raw_surface: torch.Tensor
    raw: RawSample
    name: str
    monthly_targets: torch.Tensor
    month_matrix: torch.Tensor
    months: list[str]
    interval_starts: NDArray
    interval_ends: NDArray
    start_lead: int
    end_lead: int
    next_start_lead: int


class ForecastSample(ObservationSample, total=False):
    """Surface inputs shared by observations and dense model-source supervision."""

    mask: torch.Tensor  # [physical_channel, y, x]
    fields: torch.Tensor  # [batch, history+forecast, physical_channel, y, x]

    initial_fields: torch.Tensor  # [batch, state_history, channel, y, x]
    target_fields: torch.Tensor  # [batch, forecast, channel, y, x]
