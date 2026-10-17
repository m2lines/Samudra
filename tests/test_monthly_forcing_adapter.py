# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import torch

from samudra.experiments.monthly_forcing_adapter import ModelForcingAdapter


def test_identity_under_autocast_and_mask():
    adapter = ModelForcingAdapter()
    wet = torch.tensor([[True, False], [True, True]])
    forcing = torch.randn(2, 4, 3, 2, 2) * wet
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = adapter(forcing, wet)
    torch.testing.assert_close(actual, forcing, rtol=0, atol=0)
    actual.square().mean().backward()
    assert adapter.weight.grad is not None
    assert adapter.weight.grad.abs().sum() > 0
    with torch.no_grad():
        adapter.bias.fill_(2)
    assert (adapter(forcing, wet)[..., 0, 1] == 0).all()
