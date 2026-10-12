# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Observed-target support, monthly projection and fast-objective graph checks."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from samudra.experiments.diffusion_observation_fast import (
    FIELD_KEYS,
    build_imputation,
    forecast_observed_mse,
    observation_parts_fast,
    observed_spatial_mse,
    project_sequence,
)


def sample_and_data():
    h, w, count = 4, 6, 2
    mask = torch.ones(77, h, w)
    mask[:, 0, 0] = 0
    mean = torch.full((1, 1, 77, 1, 1), 2.0)
    std = torch.full_like(mean, 3.0)
    ts = list(range(38, 52)) + list(range(57, 71))
    ts_mask = mask[ts].clone()
    ts_mask[0] = 0
    raw = torch.full((1, count, 2, h, w), 8.0)
    raw[0, 0, 0, 1, 2] = float("nan")
    interior = torch.full((1, 28, h, w), 11.0)
    interior[0, 4, 2, 3] = float("nan")
    sample: dict[str, Any] = dict(
        name="2012-01",
        surface=torch.ones(1, 19 + count, 2, h, w),
        atmosphere=torch.ones(1, 19 + count, 3, h, w),
        contexts=torch.zeros(1, 19 + count, 5, h, w),
        validity=mask[[38, 76]].bool().expand(1, 19 + count, 2, h, w).clone(),
        raw_surface=raw,
        interior=interior,
        month_weights=torch.tensor([0.25, 0.75]),
    )
    # History and raw future observations use the same normalized valid values.
    sample["surface"][:, 19:] = torch.nan_to_num((raw - 2) / 3)
    sample["validity"][:, 19:] &= torch.isfinite(raw)
    sample["surface"] *= sample["validity"]
    data = SimpleNamespace(
        device="cpu",
        mask=mask,
        mean=mean,
        std=std,
        ts_indices=ts,
        ts_mask=ts_mask,
        ts_scale=torch.full((1, 28, 1, 1), 4.0),
        surface_scale=torch.tensor([5.0, 6.0]).reshape(1, 1, 2, 1, 1),
        area=torch.tensor([0.2, 0.7, 0.8, 0.3])[:, None],
        paths=lambda split: [Path("train/2012-01.npz")] if split == "train" else [],
    )
    return sample, data


def test_projection_enforces_only_actual_observations_and_preserves_temporal_detail():
    sample, data = sample_and_data()
    base = torch.randn(1, 2, 77, 4, 6)
    before = base.clone()
    original_targets = {key: sample[key].clone() for key in ("interior", "raw_surface")}
    projected = project_sequence(base, data, sample)
    monthly = (projected * sample["month_weights"][None, :, None, None, None]).sum(1)
    target = (sample["interior"] - 2) / 3
    valid = data.ts_mask.bool() & torch.isfinite(target)
    torch.testing.assert_close(monthly[:, data.ts_indices][valid], target[valid])
    raw = (sample["raw_surface"] - 2) / 3
    surface_valid = torch.isfinite(raw) & data.mask[[38, 76]].bool()
    torch.testing.assert_close(
        projected[:, :, [38, 76]][surface_valid], raw[surface_valid]
    )
    # No correction at the deliberately missing monthly cell or velocity channels.
    torch.testing.assert_close(projected[:, :, 42, 2, 3], base[:, :, 42, 2, 3])
    torch.testing.assert_close(projected[:, :, :38], base[:, :, :38])
    assert not torch.allclose(projected[:, 0, 39], projected[:, 1, 39])
    torch.testing.assert_close(base, before)
    for key, target in original_targets.items():
        torch.testing.assert_close(sample[key], target, equal_nan=True)


@pytest.mark.parametrize("weights", [[0.5, 0.6], [-0.2, 1.2], [float("nan"), 1], [1]])
def test_invalid_month_weights_rejected(weights):
    sample, data = sample_and_data()
    sample["month_weights"] = torch.tensor(weights)
    with pytest.raises(ValueError, match="Monthly weights"):
        project_sequence(torch.zeros(1, 2, 77, 4, 6), data, sample)


def test_loss_has_no_velocity_or_missing_interior_targets():
    sample, data = sample_and_data()
    prediction = torch.randn(1, 2, 77, 4, 6, requires_grad=True)
    loss = forecast_observed_mse(data, prediction, sample)
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert prediction.grad[:, :, :38].count_nonzero() == 0
    assert prediction.grad[:, :, 42, 2, 3].count_nonzero() == 0
    other = prediction.detach().clone()
    other[:, :, :38] = 1e6
    other[:, :, 42, 2, 3] = -1e6
    torch.testing.assert_close(
        forecast_observed_mse(data, other, sample), loss.detach()
    )


def test_observed_spatial_mse_excludes_land_edges_and_handles_nan_values():
    target = torch.tensor([[[[0.0, float("nan"), 2.0, float("nan")]]]])
    prediction = torch.tensor(
        [[[[1.0, float("nan"), 3.0, float("nan")]]]], requires_grad=True
    )
    values, present = observed_spatial_mse(
        prediction,
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.ones(1, 4),
        1.0,
    )
    # Both isolated observed cells have unit error; no connecting valid edges.
    torch.testing.assert_close(values, torch.ones_like(values))
    assert present.all()
    values.sum().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 0, 0, 1].item() == 0


class Processor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))

    def rollout(self, initial, forcing, contexts, task="observation"):
        states = [initial]
        for step in range(forcing.shape[1]):
            value = forcing[:, step].mean((1, 2, 3))[:, None, None, None, None]
            states.append(states[-1] + self.weight * value)
        return states


class Decoder(nn.Module):
    noise_correlation = 0.5
    paired_noise_draws = False
    structure_aux = ""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.inputs = []
        self.latents = []

    def forward(self, noisy, sigma, latent, mask):
        self.inputs.append((noisy.detach().clone(), sigma.detach().clone()))
        self.latents.append(latent.detach().clone())
        conditioning = latent.mean((1, 2, 3))[:, None, None, None]
        return (0.7 * noisy + self.weight * conditioning) * mask


class Model(nn.Module):
    history = 19
    channels = 77
    surface = [38, 76]

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        self.encoder.weight.data.fill_(0.3)
        self.adapter = nn.Linear(1, 1, bias=False)
        self.adapter.weight.data.fill_(0.4)
        self.processor = Processor()
        self.decoder = Decoder()
        self.allow_readout = False

    def adapt(self, atmosphere):
        return atmosphere * self.adapter.weight[0, 0]

    def encode_native(self, surface, past, context, mask, validity, task="observation"):
        signal = (surface.mean() + past.mean()) * self.encoder.weight[0, 0]
        latent = signal.expand(surface.shape[0], 2, 1, 2, 2)
        known = surface.new_zeros(surface.shape[0], 2, 77, *surface.shape[-2:])
        known[:, :, self.surface] = surface[:, -2:]
        anchor = torch.zeros_like(known, dtype=torch.bool)
        anchor[:, :, self.surface] = validity[:, -2:]
        return latent, known.flatten(1, 2), anchor.flatten(1, 2)

    def call(self, module, *arguments):
        return module(*arguments)

    def forecast(
        self, surface, atmosphere, contexts, mask, validity, generator, members
    ):
        assert not torch.is_grad_enabled()
        length = atmosphere.shape[1] - self.history
        base = surface.new_ones(1, surface.shape[0], length, 77, *surface.shape[-2:])
        return base, base[:, :, :2]

    def readout(self, state, mask, generator, known, anchor):
        if not self.allow_readout:
            raise AssertionError("Fast objective must not call the full sampler")
        assert not torch.is_grad_enabled()
        return torch.ones(state.shape[0], 2, 77, *mask.shape[-2:])


def make_cache(sample, data):
    return dict(
        forecast=torch.randn(1, 2, 77, 4, 6),
        initial=torch.randn(1, 2, 77, 4, 6),
        reconstruction=torch.randn(1, 2, 2, 77, 4, 6),
        version=torch.tensor(1),
        seed=torch.tensor(5),
    )


def test_fast_objective_two_backwards_reach_initializer_processor_and_decoder():
    sample, data = sample_and_data()
    model = Model()
    cache = make_cache(sample, data)
    preserved = {key: value.clone() for key, value in cache.items()}
    labels = []
    for label, loss, metrics in observation_parts_fast(
        model, data, sample, 1729, cache
    ):
        assert torch.isfinite(loss)
        assert metrics
        labels.append(label)
        loss.backward()
    assert labels == ["forecast_completion", "reconstruction"]
    for module in (model.encoder, model.processor, model.decoder, model.adapter):
        norm = 0.0
        for parameter in module.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            norm += float(parameter.grad.abs().sum())
        assert norm > 0
    # Three forecast calls (initial+two leads), two reconstruction calls.
    assert len(model.decoder.inputs) == 5
    assert all(
        torch.equal(x[1], model.decoder.inputs[0][1]) for x in model.decoder.inputs[:3]
    )
    assert torch.equal(model.decoder.inputs[3][1], model.decoder.inputs[4][1])
    for key in cache:
        torch.testing.assert_close(cache[key], preserved[key])


def test_cache_builder_is_training_only_detached_and_restores_model_mode():
    sample, data = sample_and_data()
    model = Model().train()
    model.allow_readout = True
    cache = build_imputation(model, data, sample, 19)
    assert model.training
    for key in FIELD_KEYS:
        assert cache[key].device.type == "cpu"
        assert not cache[key].requires_grad
        assert torch.isfinite(cache[key]).all()
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(ValueError, match="training examples"):
        build_imputation(model, data, dict(sample, name="2018-01"), 19)


def test_cache_version_and_shape_fail_loudly():
    sample, data = sample_and_data()
    model = Model()
    cache = make_cache(sample, data)
    cache["version"] = torch.tensor(99)
    with pytest.raises(ValueError, match="cache version"):
        next(observation_parts_fast(model, data, sample, 1, cache))
    cache["version"] = torch.tensor(1)
    cache["forecast"] = cache["forecast"][:, :1]
    with pytest.raises(ValueError, match="shape"):
        next(observation_parts_fast(model, data, sample, 1, cache))


def test_future_labels_do_not_change_latent_conditioning():
    sample, data = sample_and_data()
    cache = make_cache(sample, data)
    changed = dict(
        sample,
        interior=sample["interior"] + 100,
        raw_surface=sample["raw_surface"] + 100,
    )
    models = [Model(), Model()]
    list(observation_parts_fast(models[0], data, sample, 43, cache))
    list(observation_parts_fast(models[1], data, changed, 43, cache))
    assert len(models[0].decoder.latents) == len(models[1].decoder.latents) == 5
    for first, second in zip(
        models[0].decoder.latents, models[1].decoder.latents, strict=True
    ):
        torch.testing.assert_close(first, second)
    # The known observations do enter the noised training target, as in DSM.
    assert not torch.equal(
        models[0].decoder.inputs[1][0], models[1].decoder.inputs[1][0]
    )


def test_empty_observed_support_returns_zero_without_nan():
    target = torch.full((1, 2, 4, 6), float("nan"))
    prediction = torch.ones_like(target, requires_grad=True)
    values, present = observed_spatial_mse(
        prediction,
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.ones(4, 6),
        1.0,
    )
    assert not present.any()
    assert torch.isfinite(values).all()
    assert values.count_nonzero() == 0
    values.sum().backward()
    assert prediction.grad is not None
    assert prediction.grad.count_nonzero() == 0
