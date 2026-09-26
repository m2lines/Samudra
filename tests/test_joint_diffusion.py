# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from samudra.experiments.joint_diffusion import (
    JointInteriorDecoder,
    channel_balanced_mse,
    denoising_loss,
)


def test_multivariate_loss_reaches_initializer_at_both_target_sizes():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    initializer = nn.Conv2d(2, 8, 1)
    decoder = JointInteriorDecoder(8, 4, width=8)
    surface = torch.randn(2, 2, 8, 12)
    for factor in (1, 2):
        initializer.zero_grad()
        decoder.zero_grad()
        latent = initializer(surface)
        target = torch.randn(2, 4, 8 * factor, 12 * factor)
        mask = torch.ones_like(target[0])
        mask[:, :2] = 0
        target *= mask
        loss = denoising_loss(
            decoder, latent, target, mask, mask, torch.Generator().manual_seed(1)
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert initializer.weight.grad is not None
        assert initializer.weight.grad.abs().sum() > 0
        for layer in decoder.condition:
            assert isinstance(layer, nn.Conv2d)
            assert layer.weight.grad is not None
            assert layer.weight.grad.abs().sum() > 0
        assert latent.shape == (2, 8, 8, 12)


def test_channel_weights_do_not_change_with_pixel_count_or_wet_area():
    target = torch.zeros(1, 2, 4, 4)
    prediction = torch.stack((torch.ones(4, 4), torch.full((4, 4), 3.0)))[None]
    weights = torch.ones(2, 4, 4)
    weights[1, :3] = 0
    assert channel_balanced_mse(prediction, target, weights).item() == pytest.approx(5)

    def repeat(x):
        return x.repeat_interleave(2, -1).repeat_interleave(2, -2)

    assert channel_balanced_mse(
        repeat(prediction), repeat(target), repeat(weights)
    ).item() == pytest.approx(5)
    with pytest.raises(ValueError, match="No supervised"):
        channel_balanced_mse(prediction, target, weights * 0)


def test_sampler_keeps_known_surfaces_and_allows_observation_gradients():
    from samudra.experiments.joint_diffusion import sample_joint

    torch.manual_seed(17)
    initializer = nn.Conv2d(2, 8, 1)
    decoder = JointInteriorDecoder(8, 4, width=8)
    surface = torch.randn(1, 2, 8, 12)
    latent = initializer(surface)
    mask = torch.ones(4, 8, 12)
    mask[:, :2] = 0
    known = torch.zeros_like(mask, dtype=torch.bool)
    known[0] = True
    values = torch.full_like(mask, 2.0) * mask
    result = sample_joint(
        decoder,
        latent,
        mask,
        torch.Generator().manual_seed(42),
        steps=3,
        known_values=values,
        known_mask=known,
        checkpoint_denoiser=True,
    )
    assert result.shape == (1, 4, 8, 12)
    assert torch.isfinite(result).all()
    assert torch.equal(result[0, 0], values[0])
    assert torch.count_nonzero(result[:, :, :2]) == 0
    # A partial observation operator, without fabricated labels for other fields.
    result[:, 1:3, 2:].mean().square().backward()
    assert initializer.weight.grad is not None
    assert initializer.weight.grad.abs().sum() > 0
    with torch.no_grad():
        repeated = sample_joint(
            decoder,
            initializer(surface),
            mask,
            torch.Generator().manual_seed(42),
            steps=3,
            known_values=values,
            known_mask=known,
        )
    torch.testing.assert_close(result, repeated, rtol=0, atol=0)
    with pytest.raises(ValueError, match="both known"):
        sample_joint(decoder, latent, mask, torch.Generator(), known_mask=known)


def test_channel_balance_handles_different_supported_channels_per_example():
    prediction = torch.tensor([1.0, 3.0, 2.0, 4.0]).reshape(2, 2, 1, 1)
    weights = torch.tensor([1.0, 0.0, 1.0, 1.0]).reshape(2, 2, 1, 1)
    result = channel_balanced_mse(prediction, torch.zeros_like(prediction), weights)
    torch.testing.assert_close(result, torch.tensor([1.0, 10.0]))
