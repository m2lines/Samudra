# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.experiments.diffusion_native_controls import velocity_statistics


def test_velocity_calibration_is_physical_and_preserves_member_structure():
    # Opposite spatial patterns have a smooth ensemble mean. Nonzero training
    # mean checks that the zero reference is physical zero, not standardized 0.
    physical = torch.tensor([[[[[0.0, 2.0]]]], [[[[2.0, 0.0]]]]])
    truth = torch.ones(1, 1, 1, 2)
    mean, std = torch.tensor([3.0]), torch.tensor([2.0])
    result = velocity_statistics(
        (physical - 3) / 2,
        (truth - 3) / 2,
        mean,
        std,
        torch.ones(1, 1, 2),
        torch.ones(1, 1, 2),
    )
    assert result["ensemble"]["mean_squared_error"].item() == 0
    assert result["ensemble"]["fair_crps"].item() == 0
    assert result["ensemble"]["ensemble_variance"].item() == 4
    assert result["zero_velocity"]["absolute_error"].item() == 2
    assert result["zero_velocity"]["mean_squared_error"].item() == 2
    assert result["mean_structure"]["variance"].item() == 0
    torch.testing.assert_close(
        result["members_structure"]["variance"], torch.ones(2, 1, 1)
    )
    torch.testing.assert_close(
        result["ensemble"]["rank_weights"].sum(0), result["ensemble"]["weight"]
    )
