# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""One-call observation denoising with explicitly imputed complete fields.

This is a surrogate, not exact full-field denoising score matching or CRPS.
Frozen-parent draws supply missing fields and temporal detail solely in the noisy
input. Observed surfaces and monthly interior averages are projected onto those
inputs; losses use the actual observations and their masks exclusively. Inference
continues to use the ordinary diffusion sampler. Joint OM4 denoising is unchanged.
"""

from pathlib import Path

import torch

from samudra.experiments.diffusion_noise import diffusion_noise
from samudra.experiments.diffusion_spatial import (
    directional_pairs,
    multiscale_pairs,
    neighbor,
)
from samudra.experiments.missingness import corrupt_sample

CACHE_VERSION = 1
FIELD_KEYS = ("forecast", "initial", "reconstruction")
SURFACE = [38, 76]


def _generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def build_imputation(model, data, original, seed):
    """Return one frozen-parent draw per task as a CPU tensor dictionary.

    Keys are forecast[B,T,C,H,W], initial[B,2,C,H,W], reconstruction[B,T,2,C,H,W],
    and integer scalar version/seed. Fields use the model's normalized physical
    coordinates. The caller's cache manifest must bind parent, data and code
    hashes to each example name; raw tensors deliberately contain no credentials.
    Only training examples may be cached. Construction never updates parameters.
    """
    if original["name"] not in {Path(p).stem for p in data.paths("train")}:
        raise ValueError("Imputation caches are restricted to training examples")
    previous_training = model.training
    model.eval()
    generator = _generator(data.device, seed)
    try:
        forecast, initial = model.forecast(
            original["surface"],
            original["atmosphere"],
            original["contexts"],
            data.mask,
            original["validity"],
            generator=generator,
            members=1,
        )
        if initial is None:
            raise ValueError("Imputation requires the initial joint readout")
        adapted = model.adapt(original["atmosphere"])
        reconstruction = []
        for step in range(len(original["month_weights"])):
            end = model.history + 1 + step
            state, known, anchor = model.encode_native(
                original["surface"][:, end - model.history : end],
                adapted[:, end - model.history : end],
                original["contexts"][:, end - 1],
                data.mask,
                original["validity"][:, end - model.history : end],
                task="observation",
            )
            reconstruction.append(
                model.readout(state, data.mask, generator, known, anchor)
            )
        result = dict(
            forecast=forecast[0].detach().float().cpu(),
            initial=initial[0].detach().float().cpu(),
            reconstruction=torch.stack(reconstruction, 1).detach().float().cpu(),
            version=torch.tensor(CACHE_VERSION, dtype=torch.int64),
            seed=torch.tensor(seed, dtype=torch.int64),
        )
        if not all(bool(torch.isfinite(result[key]).all()) for key in FIELD_KEYS):
            raise FloatingPointError("Nonfinite frozen-parent imputation")
        return result
    finally:
        model.train(previous_training)


def _month_weights(sample, length):
    weights = sample["month_weights"].float()
    if (
        weights.shape != (length,)
        or not bool(torch.isfinite(weights).all())
        or not bool((weights >= 0).all())
        or not torch.isclose(weights.sum(), weights.new_tensor(1.0), atol=1e-5)
    ):
        raise ValueError("Monthly weights must be finite, nonnegative and sum to one")
    return weights


def project_sequence(sequence, data, original):
    """Minimum-norm monthly correction; never label every frame with its mean.

    Valid five-day surface values are set independently. Monthly T/S projection
    excludes supplied SST according to data.ts_mask. Missing pixels, velocities
    and unsupported depths retain their imputed input values, never loss labels.
    """
    result = sequence.detach().float().clone()
    weights = _month_weights(original, result.shape[1])
    ts = data.ts_indices
    target = (original["interior"] - data.mean[:, 0, ts]) / data.std[:, 0, ts]
    valid = torch.isfinite(target) & data.ts_mask.bool()
    average = (result[:, :, ts] * weights[None, :, None, None, None]).sum(1)
    residual = torch.where(valid, target - average, 0)
    result[:, :, ts] += (
        weights[None, :, None, None, None] / weights.square().sum() * residual[:, None]
    )
    target_surface = (original["raw_surface"] - data.mean[:, :, SURFACE]) / data.std[
        :, :, SURFACE
    ]
    surface_valid = torch.isfinite(target_surface) & data.mask[SURFACE].bool()
    result[:, :, SURFACE] = torch.where(
        surface_valid, target_surface, result[:, :, SURFACE]
    )
    return result


def _project_surface_pairs(pairs, normalized_surface, validity):
    result = pairs.detach().float().clone()
    valid = validity.bool() & torch.isfinite(normalized_surface)
    result[..., SURFACE, :, :] = torch.where(
        valid, normalized_surface, result[..., SURFACE, :, :]
    )
    return result


def _channel_mse(prediction, target, valid, area, scale):
    valid = valid.bool() & torch.isfinite(target)
    weights = valid * area
    error = (
        (torch.where(valid, prediction.float(), 0) - torch.where(valid, target, 0))
        / scale
    ).square()
    denominator = weights.sum((-2, -1))
    return (error * weights).sum((-2, -1)) / denominator.clamp_min(
        1e-12
    ), denominator > 0


def observed_spatial_mse(prediction, target, mask, area, scale, multiscale=0.0):
    """Per-channel observed MSE plus the existing wet-neighbor difference term.

    Loss normalization matches Samples.masked_channel_mse; difference weights
    match spatial_crps (coefficient0.5, periodic longitude, no latitude wrap).
    NaNs outside valid support do not enter either predictions or targets.
    """
    valid = torch.isfinite(target) & mask.bool()
    area = area.expand_as(target)
    result, present = _channel_mse(prediction, target, valid, area, scale)
    for axis in (-1, -2):
        p, q = neighbor(prediction, axis)
        t, u = neighbor(target, axis)
        a, b = neighbor(valid, axis)
        x, y = neighbor(area, axis)
        extra, supported = _channel_mse(p - q, t - u, a & b, (x + y) / 2, scale)
        result = result + 0.5 * extra * supported
    if multiscale:
        for dy, dx, distance, support in multiscale_pairs(valid):
            p, q = directional_pairs(prediction, dy, dx)
            t, u = directional_pairs(target, dy, dx)
            x, y = directional_pairs(area, dy, dx)
            extra, supported = _channel_mse(
                (p - q) / distance,
                (t - u) / distance,
                support,
                (x + y) / 2,
                scale,
            )
            result = result + multiscale * extra * supported / 8
    return result, present


def _interior_mse(data, monthly, original, multiscale=0.0):
    physical = monthly.float() * data.std[:, 0] + data.mean[:, 0]
    values, valid = observed_spatial_mse(
        physical[:, data.ts_indices],
        original["interior"],
        data.ts_mask,
        data.area,
        data.ts_scale,
        multiscale,
    )
    groups = []
    for group in (slice(0, 14), slice(14, 28)):
        if not bool(valid[:, group].any()):
            raise ValueError("Missing observed temperature or salinity group")
        groups.append(values[:, group][valid[:, group]].mean())
    return sum(groups) / 2


def forecast_observed_mse(data, predictions, original, multiscale=0.0):
    weights = _month_weights(original, predictions.shape[1])
    monthly = (predictions * weights[None, :, None, None, None]).sum(1)
    interior = _interior_mse(data, monthly, original, multiscale)
    physical = predictions.float() * data.std + data.mean
    values, valid = observed_spatial_mse(
        physical[:, :, SURFACE],
        original["raw_surface"],
        data.mask[SURFACE],
        data.area,
        data.surface_scale,
        multiscale,
    )
    groups = []
    for channel in range(2):
        if not bool(valid[:, :, channel].any()):
            raise ValueError("Missing observed SST or SSH group")
        groups.append(values[:, :, channel][valid[:, :, channel]].mean())
    return 0.8 * interior + 0.1 * sum(groups)


def _draw_sigma(device, generator):
    # One scalar for the whole temporally aggregated training example. A monthly
    # mean is still scored only once, never as multiple instantaneous labels.
    sigma = (torch.randn((1,), device=device, generator=generator) * 1.2 - 1.2).exp()
    return sigma, ((1 + sigma.square()) / sigma.square()).squeeze(0)


def _denoise_pair(
    model, latent, clean, mask, sigma, generator, known=None, anchor=None
):
    target = clean.flatten(1, 2).detach()
    joint_mask = mask.repeat(2, 1, 1)
    noise = diffusion_noise(
        target.shape,
        joint_mask,
        generator,
        model.decoder.noise_correlation,
        model.decoder.paired_noise_draws,
    )
    noisy = target + sigma[:, None, None, None] * noise
    if anchor is not None:
        noisy = torch.where(anchor.bool(), known, noisy)
    prediction = model.call(
        model.decoder, noisy, sigma, latent.flatten(1, 2), joint_mask
    )
    if anchor is not None:
        prediction = torch.where(anchor.bool(), known, prediction)
    return prediction.reshape_as(clean)


def _cache_fields(imputation, original, data):
    if set(imputation) != {*FIELD_KEYS, "version", "seed"}:
        raise ValueError("Unexpected imputation cache fields")
    if int(imputation["version"]) != CACHE_VERSION:
        raise ValueError("Unsupported imputation cache version")
    batch, length = original["surface"].shape[0], len(original["month_weights"])
    if batch != 1:
        raise ValueError("Fast observation pilot requires microbatch size one")
    state_shape = (len(data.mask), *data.mask.shape[-2:])
    shapes = {
        "forecast": (batch, length, *state_shape),
        "initial": (batch, 2, *state_shape),
        "reconstruction": (batch, length, 2, *state_shape),
    }
    fields = {}
    for key in FIELD_KEYS:
        value = imputation[key]
        if value.requires_grad or tuple(value.shape) != shapes[key]:
            raise ValueError(f"Imputation shape or gradient contract differs: {key}")
        value = value.to(device=data.device, dtype=torch.float32)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Nonfinite imputation: {key}")
        fields[key] = value
    return fields


def observation_parts_fast(model, data, original, seed, imputation, multiscale=0.0):
    """Yield forecast/completion and reconstruction losses with one-call readouts.

    The caller supplies a frozen-parent cache bound to this training example.
    Loss targets always come from original/sample, never from imputation. Current
    latents see the same corrupted surface history and forcing as the full arm.
    Two yielded graphs permit separate backward passes, as observation_parts does.
    """
    if getattr(model.decoder, "structure_aux", ""):
        raise ValueError(
            "Fast observation pilot does not implement structure auxiliaries"
        )
    fields = _cache_fields(imputation, original, data)
    sample = corrupt_sample(original, seed)
    history = model.history
    future = project_sequence(fields["forecast"], data, original)
    initial_clean = _project_surface_pairs(
        fields["initial"],
        original["surface"][:, history - 2 : history],
        original["validity"][:, history - 2 : history],
    )
    adapted = model.adapt(sample["atmosphere"])
    initial, known, anchor = model.encode_native(
        sample["surface"][:, :history],
        adapted[:, :history],
        sample["contexts"][:, history - 1],
        data.mask,
        sample["validity"][:, :history],
        task="observation",
    )
    states = model.processor.rollout(
        initial,
        adapted[:, history:],
        sample["contexts"][:, history:],
        task="observation",
    )
    if len(states) != future.shape[1] + 1:
        raise ValueError("Forecast latent and observation horizons differ")
    generator = _generator(data.device, seed + 2000000)
    sigma, weight = _draw_sigma(data.device, generator)
    initial_prediction = _denoise_pair(
        model, initial, initial_clean, data.mask, sigma, generator, known, anchor
    )
    sequence = torch.cat((initial_clean[:, -1:], future), 1)
    predictions = torch.stack(
        [
            _denoise_pair(
                model,
                states[step + 1],
                sequence[:, step : step + 2],
                data.mask,
                sigma,
                generator,
            )[:, -1]
            for step in range(future.shape[1])
        ],
        1,
    )
    forecast = forecast_observed_mse(data, predictions, original, multiscale)
    completion = initial_prediction.sum() * 0
    if bool(sample["completion_valid"].any()):
        values, valid = observed_spatial_mse(
            initial_prediction[:, :, model.surface],
            sample["completion_target"],
            sample["completion_valid"],
            data.area,
            1.0,
            multiscale,
        )
        completion = values[valid].mean()
    loss = weight * (forecast + 0.1 * completion)
    yield (
        "forecast_completion",
        loss,
        {
            "forecast": float(forecast.detach()),
            "completion": float(completion.detach()),
            "sigma": float(sigma.item()),
            "edm_weight": float(weight.item()),
        },
    )
    del (
        predictions,
        initial_prediction,
        states,
        initial,
        adapted,
        forecast,
        completion,
        loss,
    )
    # Rebuild the adaptation graph after the first yielded loss was backpropagated.
    adapted = model.adapt(sample["atmosphere"])
    clean = fields["reconstruction"].clone()
    clean[:, :, -1] = project_sequence(clean[:, :, -1], data, original)
    clean[:, 1:, 0] = clean[:, :-1, -1].clone()
    generator = _generator(data.device, seed + 3000000)
    sigma, weight = _draw_sigma(data.device, generator)
    monthly = None
    for step, month_weight in enumerate(sample["month_weights"]):
        end = history + 1 + step
        pair = _project_surface_pairs(
            clean[:, step],
            original["surface"][:, end - 2 : end],
            original["validity"][:, end - 2 : end],
        )
        state, known, anchor = model.encode_native(
            sample["surface"][:, end - history : end],
            adapted[:, end - history : end],
            sample["contexts"][:, end - 1],
            data.mask,
            sample["validity"][:, end - history : end],
            task="observation",
        )
        prediction = _denoise_pair(
            model, state, pair, data.mask, sigma, generator, known, anchor
        )[:, -1]
        value = prediction * month_weight
        monthly = value if monthly is None else monthly + value
    if monthly is None:
        raise ValueError("Empty reconstruction month")
    interior = _interior_mse(data, monthly, original, multiscale)
    loss = 0.1 * weight * interior
    yield (
        "reconstruction",
        loss,
        {
            "reconstruction": float(interior.detach()),
            "reconstruction_sigma": float(sigma.item()),
            "reconstruction_edm_weight": float(weight.item()),
        },
    )
