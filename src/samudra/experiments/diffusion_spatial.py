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


def directional_pairs(value, dy, dx):
    """Offset pair, periodic only in longitude."""
    shifted = value.roll(-dx, -1)
    if dy:
        return value[..., :-dy, :], shifted[..., dy:, :]
    return value, shifted


def multiscale_pairs(valid):
    """Four directions at 1/2/4/8 cells; require every traversed cell valid.

    Differences are divided by lag (and sqrt(2) for diagonals). These are
    grid-coordinate derivatives, not physical derivatives per kilometer.
    """
    for lag in (1, 2, 4, 8):
        for y, x in ((0, 1), (1, 0), (1, 1), (1, -1)):
            dy, dx = y * lag, x * lag
            a, b = directional_pairs(valid, dy, dx)
            support = a & b
            for step in range(1, lag):
                middle = valid.roll(-x * step, -1)
                middle = middle[..., y * step : valid.shape[-2] - dy + y * step, :]
                support = support & middle
            yield dy, dx, lag * (2**0.5 if y and x else 1), support


def multiscale_mse(prediction, target, weights):
    result = prediction.new_zeros(prediction.shape[:-3])
    for dy, dx, distance, support in multiscale_pairs(weights.expand_as(target) > 0):
        p, q = directional_pairs(prediction, dy, dx)
        t, u = directional_pairs(target, dy, dx)
        a, b = directional_pairs(weights.expand_as(target), dy, dx)
        result = (
            result
            + channel_balanced_mse(
                (p - q) / distance, (t - u) / distance, (a + b) * 0.5 * support
            )
            / 8
        )  # Mean over four scales, with total directional weight two.
    return result


def spatial_crps(
    members, target, mask, area, scale, coefficient=0.5, multiscale=0.0, structure=""
):
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
    if multiscale:
        for dy, dx, distance, support in multiscale_pairs(valid):
            if not bool(support.any()):
                continue
            p, q = directional_pairs(members, dy, dx)
            t, u = directional_pairs(target, dy, dx)
            x, y = directional_pairs(area, dy, dx)
            extra, supported = fair_crps(
                (p - q) / distance, (t - u) / distance, support, (x + y) / 2, scale
            )
            reduced = reduced + multiscale * extra * supported / 8
    if structure:
        from samudra.experiments.diffusion_structure_losses import structure_crps

        reduced = reduced + 2.0 * structure_crps(
            members, target, mask, area, scale, structure
        )
    return reduced, present


def interior_crps(data, monthly, sample, coefficient=0.5, multiscale=0.0, structure=""):
    physical = monthly.float() * data.std[:, 0] + data.mean[:, 0]
    values, valid = spatial_crps(
        physical[:, :, data.ts_indices],
        sample["interior"],
        data.ts_mask,
        data.area,
        data.ts_scale,
        coefficient,
        multiscale,
        structure,
    )
    return (
        sum(values[:, s][valid[:, s]].mean() for s in (slice(0, 14), slice(14, 28))) / 2
    )


def forecast_crps(data, members, sample, coefficient=0.5, multiscale=0.0, structure=""):
    monthly = (members * sample["month_weights"][None, None, :, None, None, None]).sum(
        2
    )
    loss = 0.8 * interior_crps(
        data, monthly, sample, coefficient, multiscale, structure
    )
    physical = members.float() * data.std + data.mean
    values, valid = spatial_crps(
        physical[:, :, :, [38, 76]],
        sample["raw_surface"],
        data.mask[[38, 76]],
        data.area,
        data.surface_scale,
        coefficient,
        multiscale,
        structure,
    )
    return loss + 0.1 * sum(values[:, :, c][valid[:, :, c]].mean() for c in range(2))


def completion_crps(
    data, members, target, valid, coefficient=0.5, multiscale=0.0, structure=""
):
    if not bool(valid.any()):
        return members.sum() * 0
    values, present = spatial_crps(
        members, target, valid, data.area, 1.0, coefficient, multiscale, structure
    )
    return values[present].mean()
