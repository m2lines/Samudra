# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Alternative linear spatial features, respecting wet support and grid topology."""

import torch
import torch.nn.functional as F

from samudra.experiments.diffusion_spatial import directional_pairs


def box(value, size):
    shape = value.shape
    flat = value.reshape(-1, 1, *shape[-2:])
    radius = size // 2
    flat = F.pad(flat, (radius, radius, 0, 0), mode="circular")
    flat = F.pad(flat, (0, 0, radius, radius))
    return F.avg_pool2d(flat, size, stride=1).reshape(shape)


def features(prediction, target, weights, kind):
    """Yield transformed prediction/truth/weights; no crossing land or poles.

    Block averages retain scale/shape and require the entire square wet.
    Curvature is a second directional difference at lags 1/2/4; all cells
    along both segments must be valid. Each feature has equal outer weight.
    """
    valid = (weights > 0) & torch.isfinite(target)
    target = torch.nan_to_num(target)
    weights = weights.expand_as(target)
    if kind == "block":
        for size in (3, 7, 15):
            support = box(valid.float(), size) > 1 - 1e-6
            yield box(prediction, size), box(target, size), box(weights, size) * support
    elif kind == "curvature":
        for lag in (1, 2, 4):
            for y, x in ((0, 1), (1, 0), (1, 1), (1, -1)):
                dy, dx = lag * y, lag * x

                def second(value):
                    a, b = directional_pairs(value, dy, dx)
                    first, _ = directional_pairs(a, dy, dx)
                    middle, last = directional_pairs(b, dy, dx)
                    return (first - 2 * middle + last) / lag**2

                support, _ = directional_pairs(valid, 2 * dy, 2 * dx)
                support = support.clone()
                for step in range(1, 2 * lag + 1):
                    shifted = valid.roll(-x * step, -1)
                    shifted = shifted[
                        ..., y * step : valid.shape[-2] - 2 * dy + y * step, :
                    ]
                    support &= shifted
                a, b = directional_pairs(weights, 2 * dy, 2 * dx)
                yield second(prediction), second(target), (a + b) * 0.5 * support
    else:
        raise ValueError(kind)


def structure_mse(prediction, target, weights, kind):
    from samudra.experiments.joint_diffusion import channel_balanced_mse

    return torch.stack(
        [
            (
                channel_balanced_mse(p, t, w)
                if bool((w > 0).any())
                else p.sum((-3, -2, -1)) * 0
            )
            for p, t, w in features(prediction, target, weights, kind)
        ]
    ).mean(0)


def structure_crps(members, target, mask, area, scale, kind):
    from samudra.experiments.diffusion_observations import fair_crps

    weights = area.expand_as(target) * mask * torch.isfinite(target)
    values = []
    for p, t, w in features(members, target, weights, kind):
        value, supported = fair_crps(p, t, w > 0, w, scale)
        values.append(value * supported)
    return torch.stack(values).mean(0)


def excess_latent_loss(states, reference, limit=2.0):
    """Soft channel-RMS bound relative to the chain's encoded origin, not clipping."""
    scale = (
        reference.detach()
        .float()
        .square()
        .mean((-2, -1), keepdim=True)
        .sqrt()
        .clamp_min(0.1)
    )
    losses = [
        torch.relu(
            state.float().square().mean((-2, -1), keepdim=True).sqrt() / scale - limit
        )
        .square()
        .mean()
        for state in states
    ]
    return torch.stack(losses).mean()
