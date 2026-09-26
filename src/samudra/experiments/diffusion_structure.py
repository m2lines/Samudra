# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Wet-cell spatial moments and native-grid increments, not physical gradients."""

import torch


@torch.no_grad()
def field_structure(fields, mask, area):
    """Describe each field independently, preserving all non-spatial axes.

    Longitude is periodic; latitude is not. Adjacent differences require both
    cells to be wet and supported. Increment units are field units squared per
    native-cell difference, without a distance normalization. These quantities
    describe structure, not forecast skill or dynamical plausibility by themselves.
    """
    valid = mask.bool() & (area > 0)
    if not bool(torch.isfinite(fields.masked_select(valid)).all()):
        raise FloatingPointError("Nonfinite field on supported wet cells")
    values = torch.where(valid, fields.float(), 0)
    weight = valid.float() * area

    def reduce(value, weights):
        support = weights.sum((-2, -1))
        average = (value * weights).sum((-2, -1)) / support.clamp_min(1e-12)
        return torch.where(support > 0, average, torch.nan), support

    mean, support = reduce(values, weight)
    variance, _ = reduce((values - mean[..., None, None]).square(), weight)
    dx = values.roll(-1, -1) - values
    dy = values[..., 1:, :] - values[..., :-1, :]
    wx = torch.minimum(weight, weight.roll(-1, -1))
    wy = torch.minimum(weight[..., 1:, :], weight[..., :-1, :])
    zonal, zonal_support = reduce(dx.square(), wx)
    meridional, meridional_support = reduce(dy.square(), wy)
    return dict(
        area=support,
        mean=mean,
        variance=variance,
        zonal_increment_mse=zonal,
        zonal_pair_area=zonal_support,
        meridional_increment_mse=meridional,
        meridional_pair_area=meridional_support,
    )
