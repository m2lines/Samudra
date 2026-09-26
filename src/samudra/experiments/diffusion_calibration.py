# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Additive, observation-masked ensemble diagnostics for campaign reports."""

import torch


@torch.no_grad()
def ensemble_field_statistics(members, target, mask, area, scale):
    """Return wet-area weighted sums; pool sums before computing reported averages.

    Shape is (member,*target.shape), ending in channel,y,x (with optional batch
    and lead axes). Scores use standardized observation units. Ranks split ties
    uniformly across all possible ranks. Quantile coverage is the raw finite-
    ensemble diagnostic; small ensembles cannot resolve ideal tail probabilities.
    """
    count = members.shape[0]
    if count < 2 or members.shape[1:] != target.shape:
        raise ValueError("Need at least two ensemble members matching the target")
    valid = torch.isfinite(target) & mask.bool()
    weights = valid.float() * area
    support = weights.sum((-2, -1))
    if not bool((support > 0).any()):
        raise ValueError("Empty observation support")
    values = torch.where(valid[None], members.float(), 0) / scale
    truth = torch.where(valid, target, 0) / scale
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite member on observed support")

    def integrate(value):
        return (value * weights).sum((-2, -1))

    pair = torch.zeros_like(truth)
    for first in range(count):
        for second in range(first):
            pair += (values[first] - values[second]).abs()
    error = (values - truth).abs().mean(0)
    result = dict(
        members=count,
        weight=support,
        cells=valid.sum((-2, -1)),
        mean_squared_error=integrate((values.mean(0) - truth).square()),
        ensemble_variance=integrate(values.var(0, unbiased=True)),
        empirical_crps=integrate(error - pair / count**2),
        fair_crps=integrate(error - pair / (count * (count - 1))),
    )
    for coverage in (0.8, 0.9):
        tail = (1 - coverage) / 2
        bounds = torch.quantile(values, values.new_tensor([tail, 1 - tail]), dim=0)
        result[f"coverage_{int(100 * coverage)}"] = integrate(
            (truth >= bounds[0]) & (truth <= bounds[1])
        )
        result[f"interval_width_{int(100 * coverage)}"] = integrate(
            bounds[1] - bounds[0]
        )
    below = (values < truth).sum(0)
    equal = (values == truth).sum(0)
    result["rank_weights"] = torch.stack(
        [
            integrate(((below <= rank) & (rank <= below + equal)) / (equal + 1))
            for rank in range(count + 1)
        ]
    )
    return result


@torch.no_grad()
def observation_ensemble_statistics(data, members, sample):
    """Apply the same monthly/five-day operators as observation supervision."""
    physical = members.float() * data.std + data.mean
    monthly = (physical * sample["month_weights"][None, None, :, None, None, None]).sum(
        2
    )
    return dict(
        interior=ensemble_field_statistics(
            monthly[:, :, data.ts_indices],
            sample["interior"],
            data.ts_mask,
            data.area,
            data.ts_scale,
        ),
        surface=ensemble_field_statistics(
            physical[:, :, :, [38, 76]],
            sample["raw_surface"],
            data.mask[[38, 76]],
            data.area,
            data.surface_scale,
        ),
    )
