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
