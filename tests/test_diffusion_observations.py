# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from samudra.experiments.diffusion_observations import (
    fair_crps,
    forecast_observation_crps,
)


def test_fair_crps_pair_correction_missingness_and_gradient():
    # Two independent draws 0,2 around truth 1: fair score 1 - 1 = 0.
    members = torch.tensor([[[[0.0, 10.0]]], [[[2.0, -20.0]]]], requires_grad=True)
    truth = torch.tensor([[[1.0, float("nan")]]])
    score, valid = fair_crps(members, truth, torch.ones_like(truth), 1, 1)
    assert valid.all()
    assert score.item() == pytest.approx(0)
    score.sum().backward()
    assert members.grad is not None
    assert torch.isfinite(members.grad).all()
    assert torch.count_nonzero(members.grad[..., 1]) == 0
    with pytest.raises(ValueError, match="two members"):
        fair_crps(members[:1], truth, torch.ones_like(truth), 1, 1)


def fixture():
    indices = list(range(38, 52)) + list(range(57, 71))
    data = SimpleNamespace(
        std=torch.ones(1, 1, 77, 1, 1),
        mean=torch.zeros(1, 1, 77, 1, 1),
        ts_indices=indices,
        ts_mask=torch.ones(28, 2, 2),
        area=torch.ones(2, 1),
        ts_scale=torch.ones(1, 28, 1, 1),
        mask=torch.ones(77, 2, 2),
        surface_scale=torch.ones(1, 1, 2, 1, 1),
    )
    data.ts_mask[0] = 0
    sample = dict(
        month_weights=torch.tensor([0.25, 0.75]),
        interior=torch.ones(1, 28, 2, 2),
        raw_surface=torch.zeros(1, 2, 2, 2, 2),
    )
    return data, sample


def test_monthly_operator_precedes_score_and_does_not_label_deep_or_velocity():
    data, sample = fixture()
    # Each member's monthly mean is 1, although neither instantaneous field is 1.
    members = torch.zeros(2, 1, 2, 77, 2, 2)
    members[:, :, 1, data.ts_indices] = 4 / 3
    members[:, :, :, [38, 76]] = 0
    assert forecast_observation_crps(data, members, sample).item() == pytest.approx(0)
    members[:, :, :, 0:38] = 100
    members[:, :, :, 52:57] = 100
    members[:, :, :, 71:76] = 100
    assert forecast_observation_crps(data, members, sample).item() == pytest.approx(0)
    members[:, :, :, 40] += 1
    members.requires_grad_()
    loss = forecast_observation_crps(data, members, sample)
    assert loss > 0
    loss.backward()
    assert members.grad is not None
    assert torch.count_nonzero(members.grad[:, :, :, :38]) == 0
    assert members.grad[:, :, :, 40].abs().sum() > 0
    # The duration-weighted gradient must follow the same monthly operator.
    torch.testing.assert_close(members.grad[:, :, 1, 40], 3 * members.grad[:, :, 0, 40])
    sample["month_weights"] *= 2
    with pytest.raises(ValueError, match="sum to one"):
        forecast_observation_crps(data, members, sample)
