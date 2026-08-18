# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Validation diagnostics for fixed-grid patch and decoder-window seams."""

import numpy as np
import torch

from samudra.aggregator.validate.sub_aggregator import ValidateSubAggregator
from samudra.utils.distributed import all_reduce_mean
from samudra.utils.wandb import Metrics


def boundary_jump_ratio(
    error: torch.Tensor,
    spacing: tuple[int, int],
) -> torch.Tensor:
    """Compare error jumps at fixed boundaries with ordinary adjacent jumps.

    ``error`` ends in latitude and longitude dimensions. Land may be represented
    by NaNs; a difference contributes only when both adjacent cells are finite.
    Longitude is periodic, so its last-to-first jump is included as a boundary.
    A value of one means designated boundaries are no rougher than the interior.
    """
    height_spacing, width_spacing = spacing
    height, width = error.shape[-2:]
    if not 0 < height_spacing < height or not 0 < width_spacing < width:
        raise ValueError(
            f"spacing {spacing} must be smaller than error grid {(height, width)}"
        )

    lat_jump = (error[..., 1:, :] - error[..., :-1, :]).abs()
    lat_boundary = torch.zeros(height - 1, dtype=torch.bool, device=error.device)
    lat_boundary[height_spacing - 1 :: height_spacing] = True

    lon_jump = (error[..., :, 1:] - error[..., :, :-1]).abs()
    lon_boundary = torch.zeros(width - 1, dtype=torch.bool, device=error.device)
    lon_boundary[width_spacing - 1 :: width_spacing] = True
    cyclic_lon_jump = (error[..., :, :1] - error[..., :, -1:]).abs()

    boundary_values = torch.cat(
        (
            lat_jump[..., lat_boundary, :].flatten(),
            lon_jump[..., lon_boundary].flatten(),
            cyclic_lon_jump.flatten(),
        )
    )
    interior_values = torch.cat(
        (
            lat_jump[..., ~lat_boundary, :].flatten(),
            lon_jump[..., ~lon_boundary].flatten(),
        )
    )
    boundary_values = boundary_values[torch.isfinite(boundary_values)]
    interior_values = interior_values[torch.isfinite(interior_values)]
    if boundary_values.numel() == 0 or interior_values.numel() == 0:
        raise ValueError("No finite boundary or interior differences were available")
    return boundary_values.mean() / interior_values.mean().clamp_min(1e-12)


class SeamAggregator(ValidateSubAggregator):
    """Average patch/window error-jump ratios over validation batches."""

    def __init__(self, spacings: dict[str, tuple[int, int]], target_time: int):
        self._spacings = spacings
        self._target_time = target_time
        self._totals: dict[str, dict[str, torch.Tensor]] = {}
        self._n_batches = 0

    @torch.no_grad()
    def record_batch(
        self,
        *,
        loss: torch.Tensor = torch.tensor(np.nan),
        target_data,
        gen_data,
        input_data,
        target_data_norm,
        gen_data_norm,
        input_data_norm,
    ):
        del loss, target_data, gen_data, input_data, input_data_norm
        for boundary_name, spacing in self._spacings.items():
            totals = self._totals.setdefault(boundary_name, {})
            for variable, generated in gen_data_norm.items():
                error = generated.select(1, self._target_time) - target_data_norm[
                    variable
                ].select(1, self._target_time)
                ratio = boundary_jump_ratio(error, spacing)
                totals[variable] = totals.get(variable, torch.zeros_like(ratio)) + ratio
        self._n_batches += 1

    @torch.no_grad()
    def get_logs(self, label: str) -> Metrics:
        if not self._n_batches:
            raise ValueError("No validation batches were recorded")
        logs: dict[str, float] = {}
        for boundary_name, totals in self._totals.items():
            ratios = []
            for variable, total in sorted(totals.items()):
                ratio = all_reduce_mean(total / self._n_batches)
                ratios.append(ratio)
                logs[f"{label}/{boundary_name}_jump_ratio/{variable}"] = float(
                    ratio.cpu()
                )
            channel_mean = torch.stack(ratios).mean()
            logs[f"{label}/{boundary_name}_jump_ratio/channel_mean"] = float(
                channel_mean.cpu()
            )
        return logs
