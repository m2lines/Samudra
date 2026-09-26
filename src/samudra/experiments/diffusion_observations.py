# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Partial observation supervision for independently sampled ocean trajectories."""

import torch


def fair_crps(members, target, mask, area, scale):
    """Channel-reduced fair CRPS in standardized observation units.

    Members have shape (ensemble, *target.shape). The unbiased pair correction
    requires independent draws; unlike empirical-ensemble CRPS it does not prefer
    collapsed distributions merely because the training ensemble is small.
    Individual estimates can be negative. Missing observations never become labels.
    """
    count = members.shape[0]
    if count < 2 or members.shape[1:] != target.shape:
        raise ValueError("Need at least two members with the target shape")
    valid = torch.isfinite(target) & mask.bool()
    weights = valid * area
    denominator = weights.sum((-2, -1))
    present = denominator > 0
    if not bool(present.any()):
        raise ValueError("Empty observation support")
    truth = torch.where(valid, target, 0)
    values = torch.where(valid[None], members.float(), 0)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("Nonfinite prediction on observed support")
    score = ((values - truth).abs() / scale).mean(0)
    # Sum unordered pairs to avoid allocating an ensemble-squared field tensor.
    for first in range(count):
        for second in range(first):
            score = score - (values[first] - values[second]).abs() / (
                scale * count * (count - 1)
            )
    reduced = (score * weights).sum((-2, -1)) / denominator.clamp_min(1e-12)
    return reduced, present


def forecast_observation_crps(data, members, sample):
    """Score (ensemble,batch,lead,77,y,x) using the frozen upstream operators.

    Aggregate each member over the month *before* scoring interior observations.
    Surface scores retain the supplied five-day observation bins. Relative group
    weights match upstream (0.8 interior, 0.1 SST, 0.1 SSH); CRPS uses standardized
    absolute errors rather than the upstream deterministic squared-error objective.
    Velocity and unobserved deep channels require separate OM4 replay supervision.
    """
    if members.ndim != 6 or members.shape[2] != len(sample["month_weights"]):
        raise ValueError(
            "Expected ensemble,batch,lead,channel,y,x monthly trajectories"
        )
    weights = sample["month_weights"]
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Invalid monthly observation weights")
    if not torch.isclose(weights.sum(), weights.new_tensor(1.0)):
        raise ValueError("Monthly observation weights must sum to one")
    physical = members.float() * data.std + data.mean
    monthly = (physical * weights[None, None, :, None, None, None]).sum(2)
    interior, present = fair_crps(
        monthly[:, :, data.ts_indices],
        sample["interior"],
        data.ts_mask,
        data.area,
        data.ts_scale,
    )
    groups = []
    for region in (slice(0, 14), slice(14, 28)):
        values, supported = interior[:, region], present[:, region]
        if not bool(supported.any()):
            raise ValueError("Missing thermohaline group")
        groups.append(values[supported].mean())
    surface, present = fair_crps(
        physical[:, :, :, [38, 76]],
        sample["raw_surface"],
        data.mask[[38, 76]],
        data.area,
        data.surface_scale,
    )
    result = 0.4 * (groups[0] + groups[1])
    for channel in range(2):
        values, supported = surface[:, :, channel], present[:, :, channel]
        if not bool(supported.any()):
            raise ValueError("Missing forecast surface group")
        result = result + 0.1 * values[supported].mean()
    return result
