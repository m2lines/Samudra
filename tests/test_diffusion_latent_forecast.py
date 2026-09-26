# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from samudra.experiments.diffusion_latent_forecast import LatentOceanForecast


def model_for_test(stochastic):
    initializer = SimpleNamespace(
        channels=4, history=3, surface=[0, 3], net=nn.Conv2d(26, 8, 1)
    )
    return LatentOceanForecast(
        initializer,
        nn.Conv2d(8, 3, 1),
        stochastic=stochastic,
        latent_width=8,
        latent_shape=(4, 6),
        conditioning_shape=(8, 12),
        processor_depth=2,
        decoder_width=8,
        sampling_steps=3,
    )


@pytest.mark.parametrize("stochastic", [False, True])
def test_interior_supervision_trains_latent_path_without_teacher_forcing(stochastic):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = model_for_test(stochastic)
    surface, past = torch.randn(1, 3, 2, 8, 12), torch.randn(1, 3, 3, 8, 12)
    context, truth = torch.randn(1, 5, 8, 12), torch.randn(1, 2, 4, 8, 12)
    truth[:, :, [0, 3]] = surface[:, -2:]
    forcing, contexts = torch.randn(1, 3, 3, 8, 12), torch.randn(1, 3, 5, 8, 12)
    targets, mask = torch.randn(1, 3, 4, 8, 12), torch.ones(4, 8, 12)
    states = []
    hook = model.processor.register_forward_hook(
        lambda module, inputs, result: states.append(result.detach().clone())
    )
    loss = model.pretraining_loss(
        surface,
        past,
        context,
        truth,
        forcing,
        contexts,
        targets,
        mask,
        mask,
        torch.Generator().manual_seed(41),
    )
    original_states = states[:3]
    loss.backward()
    for module in (
        model.encoder.encoder,
        model.encoder.projection,
        model.processor.stem,
        model.processor.head,
        model.decoder.head,
    ):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    assert (
        model.adapter.weight.grad is None
    )  # Native OM4 forcings bypass ERA5 adaptation.
    states.clear()
    changed_truth = truth.clone()
    changed_truth[:, :, 1:3] += 100
    with torch.no_grad():
        changed = model.pretraining_loss(
            surface,
            past,
            context,
            changed_truth,
            forcing,
            contexts,
            targets + 100,
            mask,
            mask,
            torch.Generator().manual_seed(41),
        )
    assert not torch.equal(changed, loss)
    for original, altered in zip(original_states, states, strict=True):
        torch.testing.assert_close(original, altered, rtol=0, atol=0)
    hook.remove()


@pytest.mark.parametrize("stochastic", [False, True])
def test_forecast_keeps_decoding_noise_and_future_observations_out_of_recurrence(
    stochastic,
):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    model = model_for_test(stochastic).eval()
    surface = torch.randn(1, 5, 2, 8, 12)
    atmosphere, contexts = torch.randn(1, 5, 8, 8, 12), torch.randn(1, 5, 5, 8, 12)
    mask, valid = torch.ones(4, 8, 12), torch.ones_like(surface)
    valid[:, 2, 0, 1, 1] = 0
    states = []
    hook = model.processor.register_forward_hook(
        lambda module, inputs, result: states.append(result.detach().clone())
    )
    members = 2 if stochastic else 1

    def run(seed):
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            return model.forecast(
                surface,
                atmosphere,
                contexts,
                mask,
                valid,
                generator=torch.Generator().manual_seed(seed),
                members=members,
            )

    forecast, initial = run(12)
    assert forecast.shape == (members, 1, 2, 4, 8, 12)
    assert initial.shape == (members, 1, 2, 4, 8, 12)
    reference_states = states.copy()
    assert len(reference_states) == 2  # One latent path, independent of member count.
    known = valid[:, 1:3].bool().expand(members, -1, -1, -1, -1, -1)
    torch.testing.assert_close(
        initial[:, :, :, [0, 3]][known],
        surface[:, 1:3].expand(members, -1, -1, -1, -1, -1)[known],
        rtol=0,
        atol=0,
    )
    surface[:, 3:] = 1000
    states.clear()
    same, _ = run(12)
    torch.testing.assert_close(forecast, same, rtol=0, atol=0)
    states.clear()
    other, _ = run(13)
    for original, altered in zip(reference_states, states, strict=True):
        torch.testing.assert_close(original, altered, rtol=0, atol=0)
    if stochastic:
        assert not torch.equal(forecast, other)
    hook.remove()
