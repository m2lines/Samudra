# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pydantic import TypeAdapter

from samudra.config import DynamicLossConfig, Loss, build_loss_fn
from samudra.utils.ctx import BatchGrid
from samudra.utils.loss import loss_fn_from_metric


def grid(mask):
    coords = (torch.arange(mask.shape[-2]), torch.arange(mask.shape[-1]))
    return BatchGrid(mask, coords, coords)


def test_mse_equal_errors_have_equal_weight_despite_wet_fraction():
    mask = torch.tensor([[[True, True], [True, True]], [[True, False], [False, False]]])
    pred = torch.full((2, 2, 2, 2), 2.0, requires_grad=True)
    target = torch.zeros_like(pred)
    actual = loss_fn_from_metric("mse")(pred, target, grid(mask))
    torch.testing.assert_close(actual, torch.tensor([4.0, 4.0]))
    actual.sum().backward()
    assert pred.grad is not None
    assert torch.equal(pred.grad[:, 1, 0, 0], torch.tensor([2.0, 2.0]))
    assert torch.count_nonzero(pred.grad[:, 1, 1]) == 0


def test_legacy_mse_exactly_preserves_full_grid_values_and_gradients():
    mask = torch.tensor([[[True, True], [True, True]], [[True, False], [False, False]]])
    pred = torch.arange(16, dtype=torch.float64).reshape(2, 2, 2, 2).requires_grad_()
    target = torch.ones_like(pred)
    actual = loss_fn_from_metric("legacy_mse")(pred, target, grid(mask))
    expected = torch.nn.functional.mse_loss(
        pred * mask, target * mask, reduction="none"
    ).mean((0, 2, 3))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (grad_actual,) = torch.autograd.grad(actual.sum(), pred, retain_graph=True)
    (grad_expected,) = torch.autograd.grad(expected.sum(), pred)
    torch.testing.assert_close(grad_actual, grad_expected, rtol=0, atol=0)


def test_mse_ignores_nonfinite_land_and_all_dry_channels():
    mask = torch.tensor([[[True, False]], [[False, False]]])
    pred = torch.tensor(
        [[[[3.0, float("nan")]], [[float("inf"), float("nan")]]]], requires_grad=True
    )
    target = torch.tensor([[[[1.0, float("inf")]], [[float("nan"), 5.0]]]])
    loss = loss_fn_from_metric("mse")(pred, target, grid(mask))
    torch.testing.assert_close(loss, torch.tensor([4.0, 0.0]))
    loss.sum().backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[[[4.0, 0.0]], [[0.0, 0.0]]]]))


def test_nonfinite_wet_data_is_not_hidden():
    pred = torch.tensor([[[[float("nan")]]]])
    ctx = grid(torch.ones((1, 1, 1), dtype=torch.bool))
    assert torch.isnan(
        loss_fn_from_metric("mse")(pred, torch.zeros_like(pred), ctx)
    ).all()


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_large_grid_and_history_channels(dtype):
    mask = torch.zeros((4, 256, 512), dtype=torch.bool)
    mask[0, :128] = True
    mask[1, :64] = True
    mask[2:] = mask[:2]  # Two time-major output states.
    pred = torch.full((2, 4, 256, 512), 2.0, dtype=dtype, requires_grad=True)
    result = loss_fn_from_metric("mse")(pred, torch.zeros_like(pred), grid(mask))
    torch.testing.assert_close(
        result,
        torch.full(
            (4,), 4.0, dtype=torch.float64 if dtype == torch.float64 else torch.float32
        ),
    )
    result.sum().backward()
    assert pred.grad is not None
    assert pred.grad.isfinite().all()
    assert torch.count_nonzero(pred.grad.masked_select(~mask[None])) == 0


@pytest.mark.parametrize("metric", ["mse", "legacy_mse"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_config_builds_plain_and_dynamic_variants(metric, dynamic):
    value = dict(type="dynamic", metric=metric, limit=20) if dynamic else metric
    config: Loss = TypeAdapter(Loss).validate_python(value)
    fn = build_loss_fn(config, torch.device("cpu"), 2, "constant")
    mask = torch.tensor([[[True, True]], [[True, False]]])
    pred = torch.full((1, 2, 1, 2), 2.0)
    actual = fn(pred, torch.zeros_like(pred), grid(mask))
    expected = torch.tensor([4.0, 4.0 if metric == "mse" else 2.0])
    torch.testing.assert_close(actual, expected)
    assert DynamicLossConfig().metric == "mse"
