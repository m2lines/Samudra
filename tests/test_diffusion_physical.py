# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from torch import nn

from samudra.experiments.diffusion_physical import JointPhysicalForecast


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.8))

    def forward(self, states, forcing, context, mask, steps):
        return (states[:, -1] * self.gain + forcing[:, 0, :1]) * mask


class Initializer(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = 4
        self.surface = [0, 3]
        self.history = 3
        self.net = nn.Conv2d(26, 8, 1)


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.initializer = Initializer()
        self.adapter = nn.Conv2d(8, 3, 1)
        self.evolution = Dynamics()

    def adapt(self, atmosphere):
        b, t, c, h, w = atmosphere.shape
        return self.adapter(atmosphere.reshape(b * t, c, h, w)).reshape(b, t, 3, h, w)

    def call(self, function, *args):
        return function(*args)


@pytest.mark.parametrize("stochastic", [False, True])
def test_frozen_dynamics_transmit_loss_to_initializer_without_future_surface_leak(
    stochastic,
):
    torch.set_num_threads(1)
    torch.manual_seed(101)
    model = JointPhysicalForecast(
        Core(), stochastic=stochastic, width=8, sampling_steps=3
    )
    surface = torch.randn(1, 5, 2, 8, 12)
    atmosphere = torch.randn(1, 5, 8, 8, 12)
    context = torch.randn(1, 5, 5, 8, 12)
    mask = torch.ones(4, 8, 12)
    valid = torch.ones_like(surface)
    count = 2 if stochastic else 1

    def run():
        return model.forecast(
            surface,
            atmosphere,
            context,
            mask,
            valid,
            generator=torch.Generator().manual_seed(42),
            members=count,
        )

    forecast, initial = run()
    assert forecast.shape == (count, 1, 2, 4, 8, 12)
    torch.testing.assert_close(
        initial[:, :, :, [0, 3]], surface[:, 1:3].expand(count, -1, -1, -1, -1, -1)
    )
    forecast[:, :, :, 1:3].square().mean().backward()
    assert model.core.evolution.gain.grad is None
    for layer in (model.core.initializer.net, model.core.adapter, model.decoder.head):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    if stochastic:
        assert not torch.equal(initial[0], initial[1])
    surface[:, 3:] = (
        1000  # Future observations cannot enter initialization or dynamics.
    )
    with torch.no_grad():
        changed, _ = run()
    torch.testing.assert_close(changed, forecast, rtol=0, atol=0)


@pytest.mark.parametrize("stochastic", [False, True])
def test_om4_pretraining_uses_native_fluxes_and_supervises_velocity(stochastic):
    torch.set_num_threads(1)
    torch.manual_seed(301)
    model = JointPhysicalForecast(
        Core(), stochastic=stochastic, width=8, sampling_steps=3
    )
    surface = torch.randn(1, 3, 2, 8, 12)
    past = torch.randn(1, 3, 3, 8, 12)
    context = torch.randn(1, 5, 8, 12)
    mask = torch.ones(4, 8, 12)
    truth = torch.randn(1, 2, 4, 8, 12)
    truth[:, :, [0, 3]] = surface[:, -2:]
    weights = mask.clone()
    loss = model.pretraining_loss(
        surface, past, context, truth, mask, weights, torch.Generator().manual_seed(30)
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.core.adapter.weight.grad is None
    assert model.core.evolution.gain.grad is None
    assert model.core.initializer.net.weight.grad is not None
    assert model.decoder.head.weight.grad is not None
    assert model.core.initializer.net.weight.grad.abs().sum() > 0
    assert model.decoder.head.weight.grad.abs().sum() > 0
    # Native forcing sensitivity, without an ERA5 adapter in the path.
    with torch.no_grad():
        first, _, _ = model.encode_native(surface, past, context, mask)
        second, _, _ = model.encode_native(surface, past + 1, context, mask)
    assert not torch.equal(first, second)
    assert torch.equal(weights, mask)
