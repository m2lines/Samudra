# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from samudra.experiments.stochastic_state import StochasticStateForecast


def make_model(arm):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    return StochasticStateForecast(
        SimpleNamespace(channels=4, history=3, surface=[0, 3], net=nn.Conv2d(26, 8, 1)),
        nn.Conv2d(8, 3, 1),
        arm=arm,
        latent_width=8,
        latent_shape=(4, 6),
        conditioning_shape=(8, 12),
        processor_depth=1,
        decoder_width=8,
        noise_shape=(2, 3),
    )


def inputs():
    return (
        torch.randn(1, 2, 8, 4, 6),
        torch.randn(1, 3, 3, 8, 12),
        torch.randn(1, 3, 5, 8, 12),
    )


@pytest.mark.parametrize("arm", ["deterministic", "initial", "fresh", "ar1"])
def test_member_state_reproducibility_history_and_chunking(arm):
    model = make_model(arm).eval()
    initial, forcing, contexts = inputs()

    def rollout(seed):
        return model.rollout_members(
            initial,
            forcing,
            contexts,
            generator=torch.Generator().manual_seed(seed),
            members=3,
        )

    states = rollout(11)
    same = rollout(11)
    other = rollout(12)
    for state, repeat in zip(states, same, strict=True):
        torch.testing.assert_close(state, repeat, rtol=0, atol=0)
    for old, new in zip(states[:-1], states[1:], strict=True):
        torch.testing.assert_close(old[:, :, -1], new[:, :, 0], rtol=0, atol=0)
    if arm == "deterministic":
        torch.testing.assert_close(states[-1], other[-1], rtol=0, atol=0)
        torch.testing.assert_close(states[-1][0], states[-1][1], rtol=0, atol=0)
    else:
        assert not torch.equal(states[-1][0], states[-1][1])
        assert not torch.equal(states[-1], other[-1])
    rng = torch.Generator().manual_seed(11)
    current = model.sample_initial_members(initial, rng, 3)
    noise = None
    for step in range(3):
        current, noise = model.advance_members(
            current,
            forcing[:, step],
            contexts[:, step],
            generator=rng,
            noise_state=noise,
        )
        torch.testing.assert_close(current, states[step + 1], rtol=0, atol=0)
    assert (noise is not None) == (arm == "ar1")


def test_future_member_loss_reaches_initializer_processor_and_noise():
    model = make_model("ar1").train()
    surface = torch.randn(1, 3, 2, 8, 12)
    mask = torch.ones(4, 8, 12)
    latent, known, anchor = model.encode_native(
        surface, torch.randn(1, 3, 3, 8, 12), torch.randn(1, 5, 8, 12), mask
    )
    _, forcing, contexts = inputs()
    rng = torch.Generator().manual_seed(43)
    states = model.rollout_members(latent, forcing, contexts, generator=rng, members=2)
    decoded = model.decode_members(states[-1], mask)
    decoded_again = model.decode_members(states[-1], mask)
    torch.testing.assert_close(decoded, decoded_again, rtol=0, atol=0)
    before = rng.get_state().clone()
    decoded.square().mean().backward()
    assert torch.equal(before, rng.get_state())
    for module in (
        model.encoder.projection,
        model.processor.head,
        model.initial_noise.projection,
        model.transition_noise.projection,
        model.decoder.head,
    ):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    initial = model.decode_members(states[0], mask, known, anchor)
    torch.testing.assert_close(
        initial[:, :, :, [0, 3]], surface[:, -2:].expand(2, -1, -1, -1, -1, -1)
    )


def test_forecast_ignores_future_observations_and_readout_selection():
    model = make_model("fresh").eval()
    surface = torch.randn(1, 5, 2, 8, 12)
    atmosphere = torch.randn(1, 5, 8, 8, 12)
    contexts = torch.randn(1, 5, 5, 8, 12)
    valid = torch.ones_like(surface)
    mask = torch.ones(4, 8, 12)

    def run(decode):
        return model.forecast(
            surface,
            atmosphere,
            contexts,
            mask,
            valid,
            generator=torch.Generator().manual_seed(9),
            members=2,
            decode_initial=decode,
        )

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        forecast, initial = run(True)
        surface[:, 3:] += 1000
        same, omitted = run(False)
    assert forecast.shape == (2, 1, 2, 4, 8, 12)
    assert initial.shape == (2, 1, 2, 4, 8, 12)
    assert omitted is None
    torch.testing.assert_close(forecast, same, rtol=0, atol=0)


def test_arm_parameters_matched():
    a, b = make_model("deterministic"), make_model("ar1")
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
