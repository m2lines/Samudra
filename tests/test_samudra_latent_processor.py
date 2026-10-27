# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.experiments.diffusion_latent import SamudraLatentProcessor
from samudra.experiments.monthly_latent import MonthlyLatentOcean


def test_odd_grid_recurrence_forcing_gradients_and_state_roundtrip():
    torch.set_num_threads(1)
    model = SamudraLatentProcessor(8, widths=(8, 16, 32))
    latent = torch.randn(1, 2, 8, 45, 90, requires_grad=True)
    forcing = torch.randn(1, 3, 180, 360, requires_grad=True)
    context = torch.randn(1, 5, 180, 360)
    output = model(latent, forcing, context)
    assert output.shape == latent.shape
    torch.testing.assert_close(output[:, 0], latent[:, 1], rtol=0, atol=0)
    output[:, 1].square().mean().backward()
    assert forcing.grad is not None and latent.grad is not None
    assert torch.isfinite(forcing.grad).all() and forcing.grad.abs().sum() > 0
    assert torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0
    restored = SamudraLatentProcessor(8, widths=(8, 16, 32))
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(
        restored(latent, forcing, context), output, rtol=0, atol=0
    )
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())


def test_replacing_processor_preserves_other_initial_weights_and_rng():
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    old = MonthlyLatentOcean(8, 8, (16, 32), encoder_width=8)
    old_rng = torch.get_rng_state()
    torch.manual_seed(1729)
    new = MonthlyLatentOcean(8, 8, (16, 32), encoder_width=8, processor_kind="samudra")
    assert torch.equal(torch.get_rng_state(), old_rng)
    for name, value in old.state_dict().items():
        if not name.startswith("processor."):
            assert torch.equal(value, new.state_dict()[name]), name
