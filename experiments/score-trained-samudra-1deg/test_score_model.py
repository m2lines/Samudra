# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from score_model import ScoreModelConfig, crps, depth_difference, pool_field

from samudra.config import UNetBackboneConfig
from samudra.datasets import ModelBatch
from samudra.utils.ctx import BatchGrid
from samudra.utils.loss import loss_fn_from_metric


@pytest.fixture(autouse=True)
def few_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def model(arm, checkpointing="all"):
    torch.manual_seed(15)
    cfg = ScoreModelConfig(
        arm=arm,
        checkpointing=checkpointing,
        pred_residuals=False,
        use_bfloat16=False,
        unet=UNetBackboneConfig.model_validate(
            {
                "ch_width": [8, 12],
                "dilation": [1, 2],
                "n_layers": [1, 1],
                "core_block": {
                    "block_type": "conv_next_block",
                    "kernel_size": 3,
                    "activation": "capped_gelu",
                    "upscale_factor": 2,
                    "norm": "instance",
                },
                "down_sampling_block": "avg_pool",
                "up_sampling_block": "zonally_periodic_upsample",
            }
        ),
    )
    return cfg.build(
        prog_channels=154,
        boundary_channels=6,
        out_channels=154,
        input_steps=2,
        grid_sizes=[(16, 32)],
    )


def batch(steps=2):
    wet = torch.ones(154, 16, 32, dtype=torch.bool)
    wet[:, :2] = False
    ctx = BatchGrid(
        wet,
        (torch.linspace(-85, 85, 16), torch.arange(32)),
        (torch.linspace(-85, 85, 16), torch.arange(32)),
    )
    result = ModelBatch(ctx)
    for _ in range(steps):
        result.append(
            torch.randn(2, 154, 16, 32) * wet,
            torch.randn(2, 6, 16, 32),
            torch.randn(2, 154, 16, 32) * wet,
        )
    return result


@pytest.mark.parametrize("alpha", [0, 0.95, 1])
def test_crps_matches_definition_and_gradients(alpha):
    x = torch.randn(4, 2, 3, requires_grad=True)
    y = torch.randn(2, 3)
    pair = (x[:, None] - x[None, :]).abs().sum((0, 1))
    expected = (x - y).abs().mean(0) - pair * (alpha / 24 + (1 - alpha) / 32)
    actual = crps(x, y, alpha)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(a, b)
    assert actual.min() > -1e-6


def test_pooling_excludes_dry_cells_and_partial_padding():
    x = torch.full((2, 3, 5, 7), 7.0)
    w = torch.ones(3, 5, 7)
    w[:, :2] = 0
    x[:, :, :2] = 1e8
    pooled, weight = pool_field(x, w, 4)
    assert pooled.shape == (2, 3, 2, 2)
    torch.testing.assert_close(pooled, torch.full_like(pooled, 7))
    torch.testing.assert_close(weight.sum(), w.sum())


def test_depth_order_never_crosses_variable_or_time_boundary():
    x = torch.arange(154.0).reshape(1, 154, 1, 1)
    result = depth_difference(x)
    assert result.shape == (1, 144, 1, 1)
    torch.testing.assert_close(result, torch.ones_like(result))


def test_common_backbone_and_checkpointed_noise_gradients():
    b, c, plain = model("B"), model("C"), model("C", None)
    for name, p in b.base.named_parameters():
        torch.testing.assert_close(
            p, dict(c.base.named_parameters())[name], rtol=0, atol=0
        )
    plain.load_state_dict(c.state_dict())
    data = batch()
    for net in (c, plain):
        torch.manual_seed(777)
        loss = net(data, loss_fn=True).mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in net.parameters()
        )
    for a, b in zip(c.parameters(), plain.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-6)
    initial, boundary = data.get_initial_input()
    history = initial[None].expand(4, *initial.shape)
    torch.manual_seed(91)
    first = c.sample_step(history, boundary, data.ctx)
    torch.manual_seed(91)
    second = c.sample_step(history, boundary, data.ctx)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.var(0).mean() > 0
    assert torch.count_nonzero(first[..., :2, :]) == 0


def test_member_context_identity_survives_next_call():
    net = model("B")
    data = batch(1)
    # Exact deterministic identity dynamics tests the folding/history convention.
    net.base.forward_once = lambda p, b, ctx: p + b[:, :1]
    history = torch.arange(8.0).reshape(4, 2, 1, 1, 1).expand(4, 2, 154, 16, 32)
    boundary = torch.tensor([10.0, 100.0]).reshape(2, 1, 1, 1).expand(2, 6, 16, 32)
    p1 = net.sample_step(history, boundary, data.ctx)
    p2 = net.sample_step(p1, boundary, data.ctx)
    torch.testing.assert_close(p2, history + 2 * boundary[None, :, :1])


def test_structured_loss_backpropagates_and_checkpoint_roundtrips(tmp_path):
    net = model("D")
    data = batch()
    net(data, loss_fn=True).mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
    )
    path = tmp_path / "model.pt"
    torch.save(net.state_dict(), path)
    restored = model("D")
    restored.load_state_dict(torch.load(path, weights_only=True))
    for obj in (net, restored):
        torch.manual_seed(15001000)
        value = obj(data, loss_fn=True)
        if obj is net:
            expected = value.detach()
        else:
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_plain_mse_matches_unwrapped_backbone():
    net = model("B")
    data = batch()
    fn = loss_fn_from_metric("mse")

    def loss_fn(x, y):
        return fn(x, y, ctx=data.ctx)

    torch.testing.assert_close(
        net(data, loss_fn), net.base(data, loss_fn), rtol=0, atol=0
    )
