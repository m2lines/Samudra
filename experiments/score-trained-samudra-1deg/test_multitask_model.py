# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from multitask_model import MultitaskModelConfig, requested_score, task_masks
from score_model import crps
from test_score_model import batch, few_threads, model

from samudra.config import UNetBackboneConfig

# Reuse the small real backbone and model-batch fixtures, including dry cells.
__all__ = ["few_threads"]


def completion_model(mode="multitask", checkpointing="all"):
    control = model("C", checkpointing)
    torch.manual_seed(15)
    cfg = MultitaskModelConfig(
        mode=mode,
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
    net = cfg.build(
        prog_channels=154,
        boundary_channels=6,
        out_channels=154,
        input_steps=2,
        grid_sizes=[(16, 32)],
    )
    for key, value in control.state_dict().items():
        torch.testing.assert_close(net.state_dict()[key], value, atol=0, rtol=0)
    return net


def test_native_pointwise_backend_scope_and_checkpoint_compatibility(monkeypatch):
    import native_pointwise

    net = torch.nn.Sequential(torch.nn.Conv2d(3, 5, 1), torch.nn.Conv2d(5, 2, 3))
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    parameters = tuple(net.parameters())
    state = {k: v.clone() for k, v in net.state_dict().items()}
    expected = net(x)
    expected_grad = torch.autograd.grad(expected.square().sum(), (x, *parameters))
    original = torch.nn.Conv2d.forward
    enabled = torch.backends.cudnn.enabled
    calls = []

    def observe_backend(self, value):
        calls.append((self.kernel_size, torch.backends.cudnn.enabled))
        return original(self, value)

    with monkeypatch.context() as patch:
        # Restore both the global method and observer even if assertions fail.
        patch.setattr(torch.nn.Conv2d, "forward", original)
        patch.setattr(native_pointwise, "_original_forward", observe_backend)
        native_pointwise.configure_native_pointwise(True)
        native_pointwise.configure_native_pointwise(True)
        actual = net(x)
        actual_grad = torch.autograd.grad(actual.square().sum(), (x, *parameters))
        assert calls == [((1, 1), False), ((3, 3), enabled)]
        assert torch.backends.cudnn.enabled == enabled
        assert all(p is q for p, q in zip(parameters, net.parameters(), strict=True))
        torch.testing.assert_close(net.state_dict(), state, atol=0, rtol=0)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)


def test_initial_forecast_and_gradients_match_c():
    c, net = model("C"), completion_model("forecast")
    data = batch()
    torch.manual_seed(701)
    expected = c(data, loss_fn=True)
    torch.manual_seed(701)
    actual = net(data, loss_fn=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.mean().backward()
    assert net.condition.weight.grad.abs().sum() > 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
    )


@pytest.mark.parametrize("task", ["surface", "variable", "gap"])
def test_hidden_targets_history_and_land_cannot_leak(task):
    net = completion_model()
    # A trained, nonzero adapter is necessary to make this a meaningful test.
    torch.nn.init.normal_(net.condition.weight, std=0.01)
    data = batch(1)
    history, forcing = data.get_initial_input()
    target = data.get_label(0)
    hk, known, request = task_masks(
        task, data.ctx.label_mask, torch.Generator().manual_seed(55)
    )
    assert not hk.any() and not (known & request).any()
    assert torch.equal(known | request, data.ctx.label_mask)
    changed = torch.where(known, target, torch.full_like(target, 1e6))
    torch.manual_seed(12)
    first = net.complete(
        history[None].expand(4, -1, -1, -1, -1),
        forcing,
        data.ctx,
        target,
        hk,
        known,
        request,
    )
    torch.manual_seed(12)
    second = net.complete(
        torch.full_like(first, -1e6), forcing, data.ctx, changed, hk, known, request
    )
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(
        first[:, :, known], target[:, known][None].expand(4, -1, -1), atol=0, rtol=0
    )
    assert not first[:, :, ~data.ctx.label_mask].any()
    leaf = first.detach().requires_grad_()
    requested_score(leaf, target, request, data.ctx.label_mask).mean().backward()
    assert not leaf.grad[:, :, ~request].any()
    assert leaf.grad[:, :, request].abs().sum() > 0


def test_surface_mask_is_only_sst_and_ssh_and_zero_is_known():
    data = batch(1)
    _, known, request = task_masks("surface", data.ctx.label_mask, torch.Generator())
    assert known.any((-2, -1)).nonzero().flatten().tolist() == [38, 76, 115, 153]
    net = completion_model()
    history, forcing = data.get_initial_input()
    torch.nn.init.normal_(net.condition.weight, std=0.01)
    zero = torch.zeros_like(history)
    torch.manual_seed(4)
    available = net.complete(
        history[None], forcing, data.ctx, zero, torch.zeros_like(known), known, request
    )
    torch.manual_seed(4)
    unavailable = net.complete(
        history[None],
        forcing,
        data.ctx,
        zero,
        torch.zeros_like(known),
        torch.zeros_like(known),
        data.ctx.label_mask,
    )
    assert not torch.equal(available, unavailable)


def test_requested_loss_matches_c_and_rejects_empty_queries():
    data = batch(1)
    target = data.get_label(0)
    pred = torch.randn(4, *target.shape)
    wet = data.ctx.label_mask
    actual = requested_score(pred, target, wet, wet)
    torch.testing.assert_close(actual, (crps(pred, target) * wet).mean((0, 2, 3)))
    with pytest.raises(ValueError, match="No requested"):
        requested_score(pred, target, torch.zeros_like(wet), wet)


def test_task_rng_reset_and_checkpointed_gradients():
    a, b = completion_model(), completion_model(checkpointing=None)
    b.load_state_dict(a.state_dict(), strict=True)
    data = batch()
    for net in (a, b):
        net.reset_tasks(3, 0, "cpu")
        torch.manual_seed(22)
        loss = net.forward_task(data, "gap", True).mean()
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in net.parameters()
        )
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)
    a.reset_tasks(4, 0, "cpu")
    torch.manual_seed(39)
    first = a(data, True).detach()
    counts = a.task_counts.clone()
    a.reset_tasks(4, 0, "cpu")
    torch.manual_seed(39)
    second = a(data, True).detach()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(a.task_counts, counts, atol=0, rtol=0)


def test_forecast_validation_does_not_consume_completion_rng():
    net = completion_model()
    net.eval()
    data = batch()
    state = net.task_rng.get_state().clone()
    result = net(data)
    assert len(result) == 2
    assert torch.equal(net.task_rng.get_state(), state)
