# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Globally supervised latent diffusion with the deterministic control's data tasks."""

import torch

from samudra.experiments.diffusion_latent_forecast import LatentOceanForecast
from samudra.experiments.diffusion_spatial import (
    completion_crps,
    forecast_crps,
    interior_crps,
)
from samudra.experiments.joint_diffusion import denoising_loss
from samudra.experiments.missingness import (
    corrupt_sample,
    identity_adapters,
    structured_visibility,
)
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.surface_state import advance_season


def make_model(names, device, width=192, steps=32):
    core = ObservationTransfer(names, "instance")
    model = LatentOceanForecast(
        core.initializer,
        core.adapter,
        stochastic=True,
        latent_width=128,
        processor_depth=4,
        decoder_width=width,
        sampling_steps=steps,
    )
    model.input_adapters = identity_adapters(19 * 7 + 5)
    model.processor.input_adapters = identity_adapters(2 * 128 + 8)
    model.decoder.noise_correlation = 0.5
    return model.to(device)


def rng(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def observation_parts(model, data, original, seed, multiscale=0.0):
    """Yield separate differentiable objectives so graphs can be freed sequentially."""
    structure = getattr(model.decoder, "structure_aux", "")
    sample = corrupt_sample(original, seed)
    predictions, initial = model.forecast(
        sample["surface"],
        sample["atmosphere"],
        sample["contexts"],
        data.mask,
        sample["validity"],
        generator=rng(data.device, seed + 2000000),
        members=2,
    )
    forecast = forecast_crps(
        data, predictions, sample, multiscale=multiscale, structure=structure
    )
    completion = completion_crps(
        data,
        initial[:, :, :, model.surface],
        sample["completion_target"],
        sample["completion_valid"],
        multiscale=multiscale,
        structure=structure,
    )
    yield (
        "forecast_completion",
        forecast + 0.1 * completion,
        {
            "forecast": float(forecast.detach()),
            "completion": float(completion.detach()),
        },
    )
    del predictions, initial, forecast, completion
    # Reconstruct each monthly bin from its own surface history. Target interiors
    # are never encoder inputs; calendar averaging precedes every interior score.
    adapted = model.adapt(sample["atmosphere"])
    generator = rng(data.device, seed + 3000000)
    monthly = None
    for step, weight in enumerate(sample["month_weights"]):
        end = 20 + step
        state, known, anchor = model.encode_native(
            sample["surface"][:, end - 19 : end],
            adapted[:, end - 19 : end],
            sample["contexts"][:, end - 1],
            data.mask,
            sample["validity"][:, end - 19 : end],
            task="observation",
        )
        members = torch.stack(
            [
                model.readout(state, data.mask, generator, known, anchor)[:, -1]
                for _ in range(2)
            ]
        )
        value = members * weight
        monthly = value if monthly is None else monthly + value
    loss = interior_crps(
        data, monthly, sample, multiscale=multiscale, structure=structure
    )
    yield "reconstruction", 0.1 * loss, {"reconstruction": float(loss.detach())}


def native_objective(
    model,
    wave,
    index,
    coverage,
    seed,
    multiscale=0.0,
    jitter=0.0,
    horizon=6,
    dataset=None,
):
    surface, past, context, truth, forcing, targets = wave.model_sample(
        wave.trainset if dataset is None else dataset, [index]
    )
    available = wave.mask[model.surface].bool().expand_as(surface)
    visible = structured_visibility(available & coverage.bool(), seed)
    state, _, anchor = model.encode_native(
        torch.where(visible, surface, 0),
        past,
        context,
        wave.mask,
        visible,
        task="om4",
    )
    if jitter:
        scale = (
            state.detach()
            .float()
            .square()
            .mean((-2, -1), keepdim=True)
            .sqrt()
            .clamp_min(1e-3)
        )
        perturbation = torch.randn(
            state.shape, device=state.device, generator=rng(wave.device, seed + 5000000)
        )
        state = state + jitter * scale * perturbation
    if forcing.shape[1] != horizon:
        raise ValueError("Native forcing horizon differs")
    contexts = torch.stack(
        [advance_season(context, (i + 1) * 5) for i in range(horizon)], 1
    )
    states = model.processor.rollout(state, forcing, contexts, task="om4")
    sequence = torch.cat((truth, targets), 1)
    mask, weights = wave.mask.repeat(2, 1, 1), wave.weights.repeat(2, 1, 1)
    generator = rng(wave.device, seed + 4000000)
    losses = []
    for step, latent in enumerate(states):
        # Longer unroll uses the same six forecast denoiser calls, at days 35–60.
        if step != 0 and step <= horizon - 6:
            continue
        target = sequence[:, step : step + 2].flatten(1, 2)
        supervised = weights.clone()
        if step == 0:
            supervised = supervised.reshape(2, 77, *mask.shape[-2:])
            supervised[:, model.surface] = 0
            supervised = supervised.flatten(0, 1)
        loss = denoising_loss(
            model.decoder,
            latent.flatten(1, 2),
            target,
            mask,
            supervised,
            generator,
            known_mask=anchor if step == 0 else None,
            checkpoint_denoiser=model.training,
            spatial_weight=0.5,
            multiscale_weight=multiscale,
        )
        losses.append(loss)
    # Forecast/reconstruction task weights match the comparator. Denoising is
    # channel-balanced, preserving the previous diffusion objective's convention.
    result = torch.stack(losses[1:]).mean() + 0.1 * losses[0]
    hidden = available[:, -2:] & ~visible[:, -2:]
    completion_weights = torch.zeros_like(truth)
    completion_weights[:, :, model.surface] = hidden * wave.weights[model.surface]
    if bool(hidden.any()):
        completion = denoising_loss(
            model.decoder,
            states[0].flatten(1, 2),
            truth.flatten(1, 2),
            mask,
            completion_weights.flatten(1, 2),
            generator,
            known_mask=anchor,
            checkpoint_denoiser=model.training,
            spatial_weight=0.5,
            multiscale_weight=multiscale,
        )
        result = result + 0.1 * completion
    return result
