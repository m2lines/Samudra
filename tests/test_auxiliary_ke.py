# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import copy

import numpy as np
import pytest
import torch

from samudra.auxiliary_ke import VarianceTargets
from samudra.config import SamudraConfig
from samudra.datasets import ModelBatch
from samudra.stepper import train_batch
from samudra.utils.ctx import BatchGrid


def model(auxiliary):
    config = SamudraConfig.model_validate(
        dict(
            auxiliary_ke=auxiliary,
            checkpointing=None,
            last_kernel_size=3,
            unet=dict(
                ch_width=[8, 12],
                dilation=[1, 1],
                n_layers=[1, 1],
                core_block=dict(
                    block_type="conv_next_block", norm="instance", upscale_factor=2
                ),
            ),
        )
    )
    return config.build(
        prog_channels=4,
        boundary_channels=2,
        out_channels=4,
        input_steps=2,
        grid_sizes=[(16, 32)],
    )


def batch():
    ctx = BatchGrid(
        torch.ones(4, 16, 32, dtype=torch.bool),
        (torch.arange(16), torch.arange(32)),
        (torch.arange(16), torch.arange(32)),
    )
    result = ModelBatch(ctx)
    for _ in range(4):
        result.append(
            torch.randn(2, 4, 16, 32),
            torch.randn(2, 2, 16, 32),
            torch.randn(2, 4, 16, 32),
        )
    return result


def loss(prediction, target, ctx):
    return (prediction - target).square().mean((0, 2, 3))


def test_core_initialization_rng_and_zero_weight_update():
    torch.manual_seed(15)
    control = model(False)
    rng = torch.get_rng_state()
    torch.manual_seed(15)
    treatment = model(True)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in control.state_dict().items():
        assert torch.equal(value, treatment.state_dict()[key])
    data = batch()
    plain = train_batch(control, data, loss)
    data.auxiliary_targets = [torch.randn(2, 2, 16, 32) for _ in range(4)]
    data.auxiliary_weights = torch.ones(16, 32) / (16 * 32)
    result = train_batch(treatment, data, loss)
    torch.testing.assert_close(
        plain.loss_per_channel, result.loss_per_channel, rtol=0, atol=0
    )
    for network, output in ((control, plain), (treatment, result)):
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        output.optimization_loss.backward()
        optimizer.step()
    for key, value in control.state_dict().items():
        torch.testing.assert_close(value, treatment.state_dict()[key], rtol=0, atol=0)


def test_auxiliary_gradients_and_inference_roundtrip():
    network = model(True)
    data = batch()
    data.auxiliary_targets = [torch.randn(2, 2, 16, 32) for _ in range(4)]
    data.auxiliary_weights = torch.ones(16, 32) / (16 * 32)
    data.auxiliary_coefficient = 0.1
    result = train_batch(network, data, loss)
    assert result.auxiliary_loss is not None
    result.auxiliary_loss.backward()
    assert network.auxiliary_head.weight.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in network.unet.parameters()
    )
    # Later auxiliary leads also backpropagate through earlier physical predictions.
    assert network.decoder.weight.grad.abs().sum() > 0
    restored = model(True)
    restored.load_state_dict(copy.deepcopy(network.state_dict()))
    with torch.no_grad():
        for a, b in zip(network(data), restored(data), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_exact_future_dates_and_missing_dates():
    targets = VarianceTargets.__new__(VarianceTargets)
    targets.lookup = {
        f"2001-01-{day:02}": i for i, day in enumerate([3, 8, 13, 18, 23, 28])
    }
    targets.values = torch.arange(6 * 3 * 4).reshape(6, 3, 4).float()
    targets.weights = torch.ones(3, 4) / 12
    targets.coefficient = 0.2
    data = batch()
    data.steps = data.steps[:2]
    data.label_times = [
        np.array([["2001-01-13", "2001-01-18"], ["2001-01-18", "2001-01-23"]]),
        np.array([["2001-01-23", "2001-01-28"], ["2001-01-13", "2001-01-18"]]),
    ]
    targets.attach(data, torch.device("cpu"))
    torch.testing.assert_close(data.auxiliary_targets[0][0], targets.values[[2, 3]])
    torch.testing.assert_close(data.auxiliary_targets[1][0], targets.values[[4, 5]])
    data.label_times[0][0, 0] = "2002-01-01"
    with pytest.raises(KeyError):
        targets.attach(data, torch.device("cpu"))
