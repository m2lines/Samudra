# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from samudra.experiments.diffusion_calibration import ensemble_field_statistics


def test_spread_crps_coverage_and_rank_mass_have_known_values():
    members = torch.tensor([[[[0.0, 100.0]]], [[[2.0, -100.0]]]])
    target = torch.tensor([[[1.0, float("nan")]]])
    result = ensemble_field_statistics(members, target, torch.ones_like(target), 2, 1)
    assert result["weight"].item() == 2
    assert result["cells"].item() == 1
    assert result["mean_squared_error"].item() == 0
    assert result["ensemble_variance"].item() == 4  # Unbiased variance 2 * area 2.
    assert result["empirical_crps"].item() == pytest.approx(1)
    assert result["fair_crps"].item() == pytest.approx(0)
    assert result["coverage_80"].item() == 2
    torch.testing.assert_close(
        result["rank_weights"].flatten(), torch.tensor([0.0, 2.0, 0.0])
    )
    torch.testing.assert_close(result["rank_weights"].sum(0), result["weight"])


def test_ties_distribute_rank_mass_and_missing_cells_never_contribute():
    target = torch.ones(1, 2, 3)
    members = torch.ones(4, 1, 2, 3)
    mask = torch.ones_like(target)
    mask[:, 0] = 0
    members[:, :, 0] = float("nan")
    result = ensemble_field_statistics(members, target, mask, 1, 1)
    assert result["weight"].item() == 3
    assert result["ensemble_variance"].item() == 0
    assert result["empirical_crps"].item() == 0
    torch.testing.assert_close(result["rank_weights"].flatten(), torch.full((5,), 0.6))
    members[0, 0, 1, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        ensemble_field_statistics(members, target, mask, 1, 1)


def test_monthly_calibration_aggregates_each_member_before_scoring():
    from types import SimpleNamespace

    from samudra.experiments.diffusion_calibration import (
        observation_ensemble_statistics,
    )

    members = torch.zeros(2, 1, 2, 77, 1, 1)
    members[0, 0, :, :, 0, 0] = torch.tensor([1.0, -1.0])[:, None]
    members[1] = -members[0]
    data = SimpleNamespace(
        mean=torch.zeros(1, 1, 77, 1, 1),
        std=torch.ones(1, 1, 77, 1, 1),
        ts_indices=list(range(38, 52)) + list(range(57, 71)),
        ts_mask=torch.ones(28, 1, 1),
        mask=torch.ones(77, 1, 1),
        area=torch.ones(1, 1),
        ts_scale=torch.ones(1, 28, 1, 1),
        surface_scale=torch.ones(1, 1, 2, 1, 1),
    )
    sample = dict(
        month_weights=torch.tensor([0.25, 0.75]),
        interior=torch.zeros(1, 28, 1, 1),
        raw_surface=torch.zeros(1, 2, 2, 1, 1),
    )
    statistics = observation_ensemble_statistics(data, members, sample)
    interior = statistics["interior"]
    torch.testing.assert_close(interior["mean_squared_error"], torch.zeros(1, 28))
    torch.testing.assert_close(interior["ensemble_variance"], torch.full((1, 28), 0.5))
    torch.testing.assert_close(interior["empirical_crps"], torch.full((1, 28), 0.25))
    torch.testing.assert_close(
        statistics["surface"]["ensemble_variance"], torch.full((1, 2, 2), 2.0)
    )
