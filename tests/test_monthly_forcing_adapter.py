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


def test_reproduction_schedule_matches_original_pilot():
    from scripts.run_monthly_clipping_continuation import phase_settings

    main = [phase_settings(i, True) for i in range(2000)]
    cooldown = [phase_settings(i, True) for i in range(2000, 3000)]
    assert sum(source == "obs" for _, source, _ in main) == 1000
    assert sum(source == "om4" for _, source, _ in main) == 1000
    assert all(not phase and lr == 1e-4 for phase, _, lr in main)
    assert all(phase and source == "obs" for phase, source, _ in cooldown)
    assert abs(cooldown[0][2] - 9.9901e-5) < 1e-15
    assert abs(cooldown[-1][2] - 1e-6) < 1e-15
    assert phase_settings(2000, False) == (False, "obs", 1e-4)
