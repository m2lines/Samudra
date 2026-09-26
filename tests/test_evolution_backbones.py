# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from samudra.experiments.evolution_backbones import (
    Samudra2Evolution,
    Samudra2Output,
    build_evolution,
)
from samudra.experiments.normalization import configure_normalization
from samudra.experiments.surface_state import Evolution


def test_default_evolution_preserves_legacy_weights():
    torch.manual_seed(19)
    old = Evolution(3, [128, 192, 256, 384], "ar").state_dict()
    torch.manual_seed(19)
    new = build_evolution(3).state_dict()
    assert old.keys() == new.keys()
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)


def test_samudra2_head_periodic_longitude_and_zero_latitude():
    head = Samudra2Output(1, 1, 3)
    with torch.no_grad():
        head.decoder.weight.fill_(1)
        assert head.decoder.bias is not None
        head.decoder.bias.zero_()
    x = torch.zeros(1, 1, 4, 8, requires_grad=True)
    x.data[0, 0, 0, 0] = 1
    y = head(x)
    assert y.shape == x.shape
    assert y[0, 0, 0, -1] == 1
    assert y[0, 0, -1].sum() == 0
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_samudra2_instance_forward_and_strict_reload():
    torch.set_num_threads(2)
    model = configure_normalization(build_evolution(3, "samudra2"), "instance")
    assert isinstance(model, Samudra2Evolution)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert model.backbone_contract["unet"]["ch_width"] == [280, 380, 480, 520]
    states = torch.randn(1, 2, 3, 64, 128)
    forcing = torch.randn(1, 1, 3, 64, 128)
    context = torch.randn(1, 5, 64, 128)
    mask = torch.ones(3, 64, 128)
    with torch.no_grad():
        training = model(states, forcing, context, mask, 1)
        model.eval()
        evaluation = model(states, forcing, context, mask, 1)
    torch.testing.assert_close(training, evaluation, rtol=0, atol=0)
    assert evaluation.shape == (1, 3, 64, 128)
    assert torch.isfinite(evaluation).all()
    with pytest.raises(RuntimeError):
        build_evolution(3).load_state_dict(model.state_dict(), strict=True)


def test_unknown_backbone_fails():
    with pytest.raises(ValueError, match="Unknown evolution"):
        build_evolution(3, "typo")
