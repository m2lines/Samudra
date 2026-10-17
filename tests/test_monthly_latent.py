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


def test_initializer_never_reads_hidden_values_and_forecast_reaches_encoder():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = MonthlyLatentOcean(8, 8, (16, 32), history=2, encoder_width=8)
    fields = torch.randn(1, 2, 77, 16, 32, requires_grad=True)
    wet = torch.ones(77, 16, 32, dtype=torch.bool)
    mask = initializer_visibility(wet, "surface", torch.Generator().manual_seed(1))[
        None, None
    ]
    forcing = torch.randn(1, 3, 3, 16, 32)
    context = torch.randn(1, 3, 5, 16, 32)
    a = model.initialize(fields, mask, forcing[:, :2], context[:, 1])
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
