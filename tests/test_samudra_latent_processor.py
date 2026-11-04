# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from samudra.experiments.diffusion_latent import SamudraLatentProcessor
from samudra.experiments.monthly_latent import MonthlyLatentOcean


@pytest.mark.parametrize("norm", ["instance", "group"])
@pytest.mark.parametrize("widths", [(8, 16, 32), (8, 16, 24, 32)])
def test_odd_grid_recurrence_forcing_gradients_and_state_roundtrip(widths, norm):
    torch.set_num_threads(1)
    shape = (45, 90) if len(widths) == 3 else (180, 360)
    model = SamudraLatentProcessor(8, widths=widths, latent_shape=shape, norm=norm)
    latent = torch.randn(1, 2, 8, *shape, requires_grad=True)
    forcing = torch.randn(1, 3, 180, 360, requires_grad=True)
    context = torch.randn(1, 5, 180, 360)
    output = model(latent, forcing, context)
    assert output.shape == latent.shape
    torch.testing.assert_close(output[:, 0], latent[:, 1], rtol=0, atol=0)
    output[:, 1].square().mean().backward()
    assert forcing.grad is not None and latent.grad is not None
    assert torch.isfinite(forcing.grad).all() and forcing.grad.abs().sum() > 0
    assert torch.isfinite(latent.grad).all() and latent.grad.abs().sum() > 0
    restored = SamudraLatentProcessor(8, widths=widths, latent_shape=shape, norm=norm)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(
        restored(latent, forcing, context), output, rtol=0, atol=0
    )
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())


@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize("widths", [(8, 16, 32), (8, 16, 24, 32)])
def test_replacing_processor_preserves_other_initial_weights_and_rng(stride, widths):
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    old = MonthlyLatentOcean(8, 8, (16, 32), encoder_width=8, latent_stride=stride)
    old_rng = torch.get_rng_state()
    torch.manual_seed(1729)
    new = MonthlyLatentOcean(
        8,
        8,
        (16, 32),
        encoder_width=8,
        processor_kind="samudra",
        latent_stride=stride,
        processor_widths=widths,
    )
    assert (
        old.processor.latent_shape
        == new.processor.latent_shape
        == (16 // stride, 32 // stride)
    )
    assert torch.equal(torch.get_rng_state(), old_rng)
    for name, value in old.state_dict().items():
        if not name.startswith("processor."):
            assert torch.equal(value, new.state_dict()[name]), name


def test_groupnorm_changes_only_normalization_not_initial_state_or_rng():
    kwargs = dict(
        latent_width=8,
        decoder_width=8,
        shape=(16, 32),
        encoder_width=8,
        latent_stride=1,
        processor_kind="samudra",
        processor_widths=(8, 12, 16, 20),
    )
    torch.manual_seed(1729)
    instance = MonthlyLatentOcean(**kwargs)
    rng = torch.get_rng_state()
    torch.manual_seed(1729)
    group = MonthlyLatentOcean(**kwargs, processor_norm="group")
    assert torch.equal(rng, torch.get_rng_state())
    assert instance.state_dict().keys() == group.state_dict().keys()
    for name, value in instance.state_dict().items():
        assert torch.equal(value, group.state_dict()[name]), name
    norms = [m for m in group.processor.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert norms and all(m.num_groups == 4 for m in norms)
    assert not any(
        isinstance(m, torch.nn.InstanceNorm2d) for m in group.processor.modules()
    )
    assert any(
        isinstance(m, torch.nn.InstanceNorm2d) for m in group.initializer.modules()
    )


def test_v2_groupnorm_widths_and_parameter_count():
    with torch.device("meta"):
        model = SamudraLatentProcessor(
            64, (180, 360), widths=(280, 380, 480, 520), norm="group"
        )
    norms = [m for m in model.modules() if isinstance(m, torch.nn.GroupNorm)]
    assert norms and all(m.num_groups == 4 and m.num_channels % 4 == 0 for m in norms)
    assert sum(p.numel() for p in model.parameters()) == 83292608
