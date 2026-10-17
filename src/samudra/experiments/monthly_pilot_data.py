# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Observation batches and objectives for direct monthly latent pilots."""

import torch

from samudra.experiments.monthly_latent import initializer_visibility


def observation_batch(data, sample, task, generator):
    surface = sample["surface"]
    b, t, _, h, w = surface.shape
    fields = surface.new_zeros(b, t, 77, h, w)
    available = torch.zeros_like(fields, dtype=torch.bool)
    fields[:, :, [38, 76]] = surface
    available[:, :, [38, 76]] = sample["validity"].bool()
    # Observation inputs never acquire unobserved interior labels. Missing-family
    # tasks hide one of the two actually available surface fields.
    if task == "variable":
        channel = [38, 76][
            int(torch.randint(2, (), device=fields.device, generator=generator))
        ]
        available[:, :, channel] = False
    elif task == "gap":
        available &= initializer_visibility(data.mask.bool(), "gap", generator)[
            None, None
        ]
    elif task != "surface":
        raise ValueError(task)
    mean, std = data.mean[:, 0], data.std[:, 0]
    target = surface.new_zeros(b, 77, h, w)
    mask = torch.zeros_like(target, dtype=torch.bool)
    target[:, data.ts_indices] = (sample["interior"] - mean[:, data.ts_indices]) / std[
        :, data.ts_indices
    ]
    mask[:, data.ts_indices] = data.ts_mask.bool()[None] & torch.isfinite(
        sample["interior"]
    )
    return dict(fields=fields, available=available, target=target, support=mask)


def observation_loss(model, data, sample, task, generator):
    batch = observation_batch(data, sample, task, generator)
    forcing = model.adapt(sample["atmosphere"])
    states = model.trajectory(
        batch["fields"], batch["available"], forcing, sample["contexts"]
    )
    weights = sample["month_weights"]
    monthly = model.monthly_condition(states[1:], weights)
    # Exact calendar duration is recovered from the named observational month.
    import pandas as pd

    duration = float(pd.Timestamp(sample["name"]).days_in_month)
    interior = model.decoder.loss(
        monthly,
        batch["target"],
        batch["support"],
        data.area,
        generator,
        duration_days=duration,
        position_days=duration / 2,
    )
    # Uniform lead subsampling is an unbiased estimator of the mean native-lead
    # surface loss. Every initializer task retains the same future targets.
    lead = int(
        torch.randint(len(weights), (), generator=generator, device=weights.device)
    )
    target = torch.zeros_like(batch["target"])
    support = torch.zeros_like(batch["support"])
    target[:, [38, 76]] = sample["surface"][:, 19 + lead]
    support[:, [38, 76]] = sample["validity"][:, 19 + lead].bool()
    surface = model.decoder.loss(
        states[1 + lead],
        target,
        support,
        data.area,
        generator,
        position_days=2.5 + 5 * lead,
    )
    # Completion at the initial time uses only available surface truth. It never
    # uses the forecast month's interior average as an initial-state label.
    target0 = torch.zeros_like(target)
    support0 = torch.zeros_like(support)
    target0[:, [38, 76]] = sample["surface"][:, 18]
    support0[:, [38, 76]] = sample["validity"][:, 18].bool()
    reconstruction = model.decoder.loss(
        states[0], target0, support0, data.area, generator, position_days=-2.5
    )
    return 0.8 * interior + 0.2 * surface + 0.1 * reconstruction, dict(
        interior=float(interior.detach()),
        surface=float(surface.detach()),
        reconstruction=float(reconstruction.detach()),
    )
