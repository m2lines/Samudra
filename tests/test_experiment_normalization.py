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


def test_om4_model_scaling_preserves_physical_wet_values():
    from types import SimpleNamespace
    from typing import Any

    from samudra.experiments.initializer_wave import InitializerWave

    experiment: Any = InitializerWave.__new__(InitializerWave)
    experiment.native_mean = torch.tensor([2.0, 3.0, 4.0])
    experiment.native_std = torch.tensor([1.0, 2.0, 3.0])
    experiment.mean = torch.tensor([7.0, 8.0, 9.0])
    experiment.std = torch.tensor([2.0, 4.0, 6.0])
    experiment.mask = torch.ones(3, 2, 2)
    experiment.mask[:, 0, 0] = 0
    experiment.initializer = SimpleNamespace(surface=[0, 2])
    experiment.scaling_contract = {"test": True}
    full = torch.randn(1, 2, 3, 2, 2) * experiment.mask
    forcing = torch.randn(1, 2, 3, 2, 2)
    original = (full[:, :, [0, 2]], forcing, None, full, forcing, full)
    experiment.sample = lambda dataset, ids: original
    surface, past, context, truth, future, labels = experiment.model_sample(None, [])
    assert past is forcing and future is forcing
    for converted, native, indices in [
        (surface, original[0], [0, 2]),
        (truth, full, [0, 1, 2]),
        (labels, full, [0, 1, 2]),
    ]:
        observed = (
            converted * experiment.std[indices][None, None, :, None, None]
            + experiment.mean[indices][None, None, :, None, None]
        )
        expected = (
            native * experiment.native_std[indices][None, None, :, None, None]
            + experiment.native_mean[indices][None, None, :, None, None]
        )
        mask = experiment.mask[indices][None, None]
        torch.testing.assert_close(observed * mask, expected * mask)
        assert torch.equal(converted[..., 0, 0], torch.zeros_like(converted[..., 0, 0]))
