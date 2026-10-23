# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timedelta

import pytest
import torch

from samudra.experiments.monthly_latent import (
    MonthlyLatentOcean,
    WindowDecoder,
    calendar_weights,
    initializer_visibility,
)


def test_calendar_partial_bins_and_missing_interval():
    first, last = datetime(2001, 2, 1), datetime(2001, 3, 1)
    starts = [first - timedelta(days=2) + timedelta(days=5 * i) for i in range(6)]
    ends = [s + timedelta(days=5) for s in starts]
    assert calendar_weights(starts, ends, first, last) == pytest.approx(
        [3 / 28] + [5 / 28] * 5
    )
    with pytest.raises(ValueError):
        calendar_weights(starts[:-1], ends[:-1], first, last)


@pytest.mark.parametrize("stride", [1, 4])
def test_initializer_never_reads_hidden_values_and_forecast_reaches_encoder(stride):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = MonthlyLatentOcean(
        8, 8, (16, 32), history=2, encoder_width=8, latent_stride=stride
    )
    fields = torch.randn(1, 2, 77, 16, 32, requires_grad=True)
    wet = torch.ones(77, 16, 32, dtype=torch.bool)
    mask = initializer_visibility(wet, "surface", torch.Generator().manual_seed(1))[
        None, None
    ]
    forcing = torch.randn(1, 3, 3, 16, 32)
    context = torch.randn(1, 3, 5, 16, 32)
    a = model.initialize(fields, mask, forcing[:, :2], context[:, 1])
    assert a.shape == (1, 2, 8, 16 // stride, 32 // stride)
    altered = torch.where(mask, fields, torch.full_like(fields, float("nan")))
    b = model.initialize(altered, mask, forcing[:, :2], context[:, 1])
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    future = model.processor(a, forcing[:, 2], context[:, 2])
    future.square().mean().backward()
    assert fields.grad is not None
    assert fields.grad[:, :, [38, 76]].abs().sum() > 0
    assert fields.grad[:, :, :38].abs().sum() == 0
    assert model.frame.weight.grad is not None
    assert model.frame.weight.grad.abs().sum() > 0
    assert model.processor.head.weight.grad is not None
    assert model.processor.head.weight.grad.abs().sum() > 0


def test_masked_denoising_is_invariant_to_missing_target_and_has_latent_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    decoder = WindowDecoder(8, 8)
    latent = torch.randn(1, 2, 8, 4, 8, requires_grad=True)
    target = torch.randn(1, 77, 16, 32)
    support = torch.zeros_like(target, dtype=torch.bool)
    support[:, 38:52] = True
    target2 = torch.where(support, target, torch.full_like(target, float("nan")))

    def loss(value):
        return decoder.loss(
            latent,
            value,
            support,
            torch.ones(16, 1),
            torch.Generator().manual_seed(23),
            duration_days=28,
        )

    a, b = loss(target), loss(target2)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.backward()
    assert latent.grad is not None
    assert latent.grad.abs().sum() > 0
    assert decoder.core.head.weight.grad is not None
    assert decoder.core.head.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="No observed"):
        decoder.loss(
            latent, target, support & False, torch.ones(16, 1), torch.Generator()
        )


def test_calendar_rejects_overlap_even_when_total_duration_matches():
    first, last = datetime(2001, 1, 1), datetime(2001, 1, 11)
    with pytest.raises(ValueError):
        calendar_weights([first, first], [first + timedelta(days=5)] * 2, first, last)


def test_exact_source_mix_and_cooldown():
    from collections import Counter

    from samudra.experiments.monthly_model_data import task_source

    assert Counter(task_source("A", i) for i in range(2000)) == {
        "obs": 1000,
        "om4": 1000,
    }
    assert Counter(task_source("D", i) for i in range(2000)) == {
        "obs": 1000,
        "om4": 500,
        "cm4": 500,
    }
    assert Counter(task_source("E", i) for i in range(2000)) == {"obs": 2000}
    assert all(
        task_source(a, i, cooldown=True) == "obs" for a in "ABCDE" for i in range(1000)
    )


def test_noleap_calendar_stays_native():
    import cftime

    from samudra.experiments.monthly_model_data import calendar_context

    first, last = cftime.DatetimeNoLeap(152, 2, 1), cftime.DatetimeNoLeap(152, 3, 1)
    starts = [first - timedelta(days=2) + timedelta(days=i * 5) for i in range(6)]
    weights = calendar_weights(
        starts, [s + timedelta(days=5) for s in starts], first, last
    )
    assert weights == pytest.approx([3 / 28] + [5 / 28] * 5)
    contexts = calendar_context([0.0], [0.0], [first, last])
    assert contexts.shape == (2, 5, 1, 1)


def test_fixed_pool_matches_adaptive_values_and_gradients():
    from torch.nn import functional as F

    value = torch.randn(1, 3, 16, 32, requires_grad=True)
    fixed = F.avg_pool2d(value, 4)
    adaptive = F.adaptive_avg_pool2d(value, (4, 8))
    torch.testing.assert_close(fixed, adaptive)
    a = torch.autograd.grad(fixed.square().sum(), value, retain_graph=True)[0]
    b = torch.autograd.grad(adaptive.square().sum(), value)[0]
    torch.testing.assert_close(a, b)
