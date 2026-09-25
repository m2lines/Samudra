# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from samudra.experiments.normalization import configure_normalization


def test_instance_norm_has_no_running_state_or_batch_coupling():
    torch.manual_seed(4)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.Sequential(nn.BatchNorm2d(4))
    )
    model.double()
    configure_normalization(model, "instance")
    assert not any(
        isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()
    )
    block = model[1]
    assert isinstance(block, nn.Sequential)
    norm = block[0]
    assert isinstance(norm, nn.InstanceNorm2d)
    assert norm.affine and not norm.track_running_stats
    assert not list(norm.named_buffers())
    x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    training = model.train()(x)
    torch.testing.assert_close(model.eval()(x), training)
    torch.testing.assert_close(model(x[:1]), training[:1])
    assert not any("running_" in k for k in model.state_dict())
    training.square().mean().backward()
    assert norm.weight.grad is not None


def test_default_and_invalid_normalization():
    model = nn.Sequential(nn.BatchNorm2d(2))
    assert configure_normalization(model, "batch") is model
    assert isinstance(model[0], nn.BatchNorm2d)
    with pytest.raises(ValueError):
        configure_normalization(model, "unknown")
