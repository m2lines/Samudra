# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fair p=1 variogram score on wet regular-grid spatial increments.

This scores the ensemble expectation of each absolute spatial increment. It is
not a gradient-magnitude penalty: the target increment sets the desired scale.
The sample-variance correction removes the finite-ensemble bias of the squared
sample-mean error. A realized fair score may be negative and is never clamped.

Lags are in computational grid cells, not physical distance. Directions and lags
have equal weight; this does not claim physical isotropy on a latitude grid.
Longitude is periodic and latitude is not. Every cell on each discrete straight
path must have valid target/wet support, including both endpoints.
"""

import torch

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def validate_lags(lags):
    lags = tuple(lags)
    if not lags or any(type(lag) is not int or lag <= 0 for lag in lags):
        raise ValueError("Variogram lags must be positive integers")
    if any(a >= b for a, b in zip(lags, lags[1:])):
        raise ValueError("Variogram lags must be strictly increasing")
    return lags


def _offset(values, dy, dx, distance, rows):
    """Values at a positive-latitude offset and circular-longitude offset."""
    return values[..., dy * distance : dy * distance + rows, :].roll(
        -dx * distance, dims=-1
    )


def variogram_score(members, target, mask, area, lags=(1, 2, 4, 8)):
    """Return scalar fair/plugin/correction scores and their per-lag vectors.

    ``members`` has shape M,B,C,H,W; ``target`` B,C,H,W; ``mask`` C,H,W;
    ``area`` H,W or H,1 (latitude-only weights). Inputs are already normalized
    using fixed channel statistics.
    Each direction first averages endpoint-area-weighted scores over space,
    then equally averages supported batch/channel entries. Supported directions
    and lags are averaged equally. Entirely unsupported inputs return NaN so the
    trainer's finite-loss gate fails loudly without per-update host synchronization.

    For increment samples t and reference a, the fair estimator is
    ``(mean(t)-a)**2 - var(t, unbiased=True)/M``. M=1 uses the conventional
    deterministic squared-increment error and zero correction.
    """
    lags = validate_lags(lags)
    if members.ndim != 5 or members.shape[1:] != target.shape:
        raise ValueError("Expected members M,B,C,H,W and matching target B,C,H,W")
    if members.shape[0] < 1 or mask.shape != target.shape[1:]:
        raise ValueError("Require at least one member and a C,H,W mask")
    height, width = target.shape[-2:]
    if area.shape not in ((height, width), (height, 1)):
        raise ValueError("Area must have shape H,W or H,1")
    area = area.expand(height, width)
    if max(lags) >= min(height, width):
        raise ValueError("Variogram lags must be smaller than both grid dimensions")
    count = members.shape[0]
    with torch.autocast(device_type=members.device.type, enabled=False):
        valid = mask.bool()[None] & torch.isfinite(target)
        values = torch.where(valid[None], members.float(), 0)
        truth = torch.where(valid, target.float(), 0)
        area = area.float()
        totals, lag_support = [], []
        for lag in lags:
            directions, direction_support = [], []
            for dy, dx in DIRECTIONS:
                rows = height - dy * lag
                support = valid[..., :rows, :].clone()
                for distance in range(1, lag + 1):
                    support = support & _offset(valid, dy, dx, distance, rows)
                weight = torch.where(
                    support,
                    0.5 * (area[:rows] + _offset(area, dy, dx, lag, rows)),
                    0,
                )
                denominator = weight.sum((-2, -1))
                supported = denominator > 0
                # Mask endpoints before subtraction/abs so excluded NaNs cannot
                # enter either forward reductions or their gradients.
                increments = (
                    torch.where(support[None], values[..., :rows, :], 0)
                    - torch.where(support[None], _offset(values, dy, dx, lag, rows), 0)
                ).abs()
                reference = (
                    torch.where(support, truth[..., :rows, :], 0)
                    - torch.where(support, _offset(truth, dy, dx, lag, rows), 0)
                ).abs()
                mean = increments.mean(0)
                plugin = (mean - reference).square()
                correction = (
                    (increments - mean).square().sum(0) / (count * (count - 1))
                    if count > 1
                    else torch.zeros_like(plugin)
                )
                pair = torch.stack((plugin, correction))
                spatial = (pair * weight).sum((-2, -1)) / denominator.clamp_min(1e-30)
                average = torch.where(supported[None], spatial, 0).sum((-2, -1))
                directions.append(average / supported.sum().clamp_min(1))
                direction_support.append(supported.any())
            direction_flags = torch.stack(direction_support)
            totals.append(
                torch.stack(directions).sum(0) / direction_flags.sum().clamp_min(1)
            )
            lag_support.append(direction_flags.any())
        by_lag = torch.stack(totals)
        lag_flags = torch.stack(lag_support)
        overall = by_lag.sum(0) / lag_flags.sum()
        plugin_by_lag, correction_by_lag = by_lag.unbind(-1)
        plugin, correction = overall.unbind(-1)
        return dict(
            fair=plugin - correction,
            plugin=plugin,
            correction=correction,
            fair_by_lag=plugin_by_lag - correction_by_lag,
            plugin_by_lag=plugin_by_lag,
            correction_by_lag=correction_by_lag,
        )
