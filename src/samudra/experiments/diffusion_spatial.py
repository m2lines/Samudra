# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed wet-neighbor losses; periodic longitude, nonperiodic latitude."""

import torch

from samudra.experiments.diffusion_observations import fair_crps
from samudra.experiments.joint_diffusion import channel_balanced_mse


def neighbor(value, axis):
    if axis == -1:
        return value, value.roll(-1, -1)
    if axis == -2:
        return value[..., :-1, :], value[..., 1:, :]
    raise ValueError("Only spatial axes are supported")


def spatial_mse(prediction, target, weights):
    result = prediction.new_zeros(prediction.shape[:-3])
    for axis in (-1, -2):
        p, q = neighbor(prediction, axis)
        t, u = neighbor(target, axis)
        a, b = neighbor(weights.expand_as(target), axis)
        pair = (a > 0) & (b > 0)
        w = (a + b) * 0.5 * pair
        if bool(pair.any()):
            result = result + channel_balanced_mse(p - q, t - u, w)
    return result


def spatial_crps(members, target, mask, area, scale, coefficient=0.5):
    """Pixel fair CRPS plus the equally scaled x/y increment scores."""
    reduced, present = fair_crps(members, target, mask, area, scale)
    valid = torch.isfinite(target) & mask.bool()
    area = area.expand_as(target)
    for axis in (-1, -2):
        p, q = neighbor(members, axis)
        t, u = neighbor(target, axis)
        a, b = neighbor(valid, axis)
        x, y = neighbor(area, axis)
        if bool((a & b).any()):
            extra, supported = fair_crps(p - q, t - u, a & b, (x + y) / 2, scale)
            reduced = reduced + coefficient * extra * supported
    return reduced, present


def interior_crps(data, monthly, sample, coefficient=0.5):
    physical = monthly.float() * data.std[:, 0] + data.mean[:, 0]
    values, valid = spatial_crps(
        physical[:, :, data.ts_indices],
        sample["interior"],
        data.ts_mask,
        data.area,
        data.ts_scale,
        coefficient,
    )
    return (
        sum(values[:, s][valid[:, s]].mean() for s in (slice(0, 14), slice(14, 28))) / 2
    )


def forecast_crps(data, members, sample, coefficient=0.5):
    monthly = (members * sample["month_weights"][None, None, :, None, None, None]).sum(
        2
    )
    loss = 0.8 * interior_crps(data, monthly, sample, coefficient)
    physical = members.float() * data.std + data.mean
    values, valid = spatial_crps(
        physical[:, :, :, [38, 76]],
        sample["raw_surface"],
        data.mask[[38, 76]],
        data.area,
        data.surface_scale,
        coefficient,
    )
    return loss + 0.1 * sum(values[:, :, c][valid[:, :, c]].mean() for c in range(2))


def completion_crps(data, members, target, valid, coefficient=0.5):
    if not bool(valid.any()):
        return members.sum() * 0
    values, present = spatial_crps(members, target, valid, data.area, 1.0, coefficient)
    return values[present].mean()
