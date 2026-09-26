# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from samudra.experiments.diffusion_latent import (
    LatentHistoryEncoder,
    PersistentLatentProcessor,
)
from samudra.experiments.joint_diffusion import (
    JointInteriorDecoder,
    channel_balanced_mse,
    denoising_loss,
)


@pytest.mark.parametrize("diffusion", [False, True])
def test_future_readout_loss_reaches_initializer_through_persistent_latent_steps(
    diffusion,
):
    torch.set_num_threads(1)
    torch.manual_seed(24)
    encoder = LatentHistoryEncoder(nn.Conv2d(6, 8, 1), 8, width=8, latent_shape=(4, 6))
    processor = PersistentLatentProcessor(
        width=8, depth=2, latent_shape=(4, 6), conditioning_shape=(8, 12)
    )
    decoder = JointInteriorDecoder(16, 4, width=8)
    conditioning = torch.randn(1, 6, 8, 12)
    forcing, context = torch.randn(1, 3, 3, 8, 12), torch.randn(1, 3, 5, 8, 12)
    initial = encoder(conditioning)
    states = processor.rollout(initial, forcing, context)
    assert len(states) == 4
    assert all(state.shape == initial.shape for state in states)
    target, mask = torch.randn(1, 4, 8, 12), torch.ones(4, 8, 12)
    final = states[-1].flatten(1, 2)
    if diffusion:
        loss = denoising_loss(
            decoder, final, target, mask, mask, torch.Generator().manual_seed(5)
        )
    else:
        prediction = decoder(torch.zeros_like(target), torch.ones(1), final, mask)
        loss = channel_balanced_mse(prediction, target, mask).mean()
    loss.backward()
    for module in (
        encoder.encoder,
        encoder.projection,
        processor.stem,
        processor.head,
        decoder.head,
    ):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    with torch.no_grad():
        changed = processor.rollout(initial, forcing + 1, context)
    assert not torch.equal(changed[-1], states[-1])
    with pytest.raises(ValueError, match="forcing shape"):
        processor(initial, torch.randn(1, 3, 16, 24), context[:, 0])
