# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Full-state denoising on OM4; differentiable physical trajectories on OBS."""

import torch

from samudra.experiments.diffusion_spatial import completion_crps, forecast_crps
from samudra.experiments.missingness import corrupt_sample


def observation_objective(model, data, original, generator, members=2, capture=None):
    return observation_batch_objective(
        model, data, [original], generator, members=members, capture=capture
    )


def observation_batch_objective(
    model, data, originals, generator, members=2, capture=None, seeds=None
):
    """Batch network work, then average separately normalized example losses.

    Monthly weights and missing target support remain example-specific. Group
    callers by temporal length; no padding changes the scientific objective.
    """
    if members < 2 or not originals:
        raise ValueError("Require examples and at least two sampled trajectories")
    if seeds is None:
        seeds = [generator.initial_seed() + i for i in range(len(originals))]
    samples = [
        corrupt_sample(s, seed) for s, seed in zip(originals, seeds, strict=True)
    ]
    if len({s["atmosphere"].shape[1] for s in samples}) != 1:
        raise ValueError("Microbatch temporal lengths differ")
    inputs = [
        torch.cat([s[key] for s in samples], 0)
        for key in ("surface", "atmosphere", "contexts", "validity")
    ]
    prediction, initial = model.forecast(
        inputs[0],
        inputs[1],
        inputs[2],
        data.mask,
        inputs[3],
        generator=generator,
        members=members,
    )
    if capture is not None:
        capture(prediction.detach(), initial.detach())
    losses = []
    for i, sample in enumerate(samples):
        losses.append(
            forecast_crps(data, prediction[:, i : i + 1], sample)
            + 0.1
            * completion_crps(
                data,
                initial[:, i : i + 1, :, model.surface],
                sample["completion_target"],
                sample["completion_valid"],
            )
        )
    return torch.stack(losses).mean()


def native_objective(model, data, batch, generator):
    """Equal initializer / mean teacher-forced transition denoising weights.

    All fields are already in the shared physical normalization. ModelMonths
    supplies exact contiguous five-day intervals; no monthly target is treated
    as an instantaneous state. Both tasks use the same eight ERA5 channels.
    """
    fields, mask = batch["fields"], batch["mask"]
    n = model.history
    if fields.shape[1] <= n or n < 2:
        raise ValueError("Need historical states and at least one future target")
    forcing = batch["atmosphere"]
    contexts = batch["contexts"]
    weights = mask * data.area
    initial = model.pretraining_loss(
        fields[:, :n, model.surface],
        forcing[:, :n],
        contexts[:, n - 1],
        fields[:, n - 2 : n],
        mask,
        weights,
        generator,
    )
    transitions = [
        model.transition_loss(
            fields[:, step - 2 : step],
            forcing[:, step],
            contexts[:, step],
            fields[:, step],
            mask,
            weights,
            generator,
        )
        for step in range(n, fields.shape[1])
    ]
    return initial + torch.stack(transitions).mean()
