# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.experiments.initializer_diffusion import (
    ResidualDenoiser,
    edm_loss,
    ensemble_crps,
    sample_edm,
)


def test_crps_matches_pairwise_definition():
    samples = torch.tensor([[-2.0, 1.0], [0.0, 2.0], [3.0, 9.0]])
    truth = torch.tensor([1.0, 3.0])
    expected = (samples - truth).abs().mean(0) - 0.5 * (
        samples[:, None] - samples[None]
    ).abs().mean((0, 1))
    torch.testing.assert_close(ensemble_crps(samples, truth), expected)


def test_denoising_and_sampling_mask_and_seed():
    torch.manual_seed(7)
    model = ResidualDenoiser(3, width=8)
    mask = torch.ones(1, 1, 12, 20)
    mask[..., :3, :4] = 0
    condition = torch.randn(2, 3, 12, 20)
    target = torch.randn(2, 1, 12, 20) * mask
    generator = torch.Generator().manual_seed(11)
    loss = edm_loss(model, target, condition, mask, mask, generator)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.weight.grad is not None
    assert model.head.weight.grad.abs().sum() > 0
    a = sample_edm(model, condition, mask, torch.Generator().manual_seed(12), steps=4)
    b = sample_edm(model, condition, mask, torch.Generator().manual_seed(12), steps=4)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.isfinite(a).all()
    assert torch.count_nonzero(a[..., :3, :4]) == 0
