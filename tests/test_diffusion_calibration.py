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
