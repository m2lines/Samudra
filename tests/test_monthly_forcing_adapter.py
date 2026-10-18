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


def test_early_cooldowns_have_identical_relative_schedule():
    from scripts.run_monthly_clipping_continuation import phase_settings

    schedules = [
        [
            phase_settings(2000 + step - start, True)
            for step in range(start, start + 1000)
        ]
        for start in (0, 100, 1000, 2000)
    ]
    assert all(s == schedules[0] for s in schedules)
    assert all(cool and source == "obs" for cool, source, _ in schedules[0])
    assert abs(schedules[0][-1][2] - 1e-6) < 1e-15


def test_trajectory_does_not_consume_future_physical_fields():
    from types import SimpleNamespace
    from typing import cast

    from samudra.experiments.monthly_latent import MonthlyLatentOcean

    def initialize(fields, available, forcing, context):
        assert fields.shape[1] == 3
        return fields.sum() + available.sum() + forcing.sum() + context.sum()

    stub = SimpleNamespace(
        history=3,
        initialize=initialize,
        processor=SimpleNamespace(
            rollout=lambda initial, forcing, contexts: initial
            + forcing.sum()
            + contexts.sum()
        ),
    )
    fields = torch.randn(1, 7, 2, 2, 2)
    available = torch.ones_like(fields, dtype=torch.bool)
    forcing = torch.randn_like(fields)
    contexts = torch.randn(1, 7, 1, 2, 2)
    before = MonthlyLatentOcean.trajectory(
        cast(MonthlyLatentOcean, stub), fields, available, forcing, contexts
    )
    fields[:, 3:] = 999
    available[:, 3:] = False
    after = MonthlyLatentOcean.trajectory(
        cast(MonthlyLatentOcean, stub), fields, available, forcing, contexts
    )
    torch.testing.assert_close(before, after, rtol=0, atol=0)
