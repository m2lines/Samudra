# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.aggregator.validate.seam import (
    boundary_jump_ratio,
    periodic_phase_power_ratio,
)


def test_boundary_jump_ratio_detects_fixed_window_offsets():
    height, width = 12, 16
    smooth = torch.arange(width, dtype=torch.float32).repeat(height, 1) * 0.01
    window_offsets = torch.arange(width) // 4
    error = smooth + window_offsets

    ratio = boundary_jump_ratio(error, spacing=(4, 4))

    assert ratio > 10


def test_boundary_jump_ratio_ignores_nan_land():
    error = torch.randn(2, 12, 16)
    error[:, :3, :5] = torch.nan

    ratio = boundary_jump_ratio(error, spacing=(4, 4))

    assert torch.isfinite(ratio)


def test_periodic_phase_power_detects_repeated_within_window_mode():
    height, width = 24, 32
    phase = torch.tensor([0.0, 1.0, 0.0, -1.0])
    repeated = phase.repeat(width // len(phase)).repeat(height, 1)
    random = torch.randn_like(repeated)

    repeated_ratio = periodic_phase_power_ratio(repeated, spacing=(4, 4))
    random_ratio = periodic_phase_power_ratio(random, spacing=(4, 4))

    assert repeated_ratio > 0.4
    assert repeated_ratio > 10 * random_ratio


def test_periodic_phase_power_ignores_nan_land():
    error = torch.randn(2, 12, 16)
    error[:, :3, :5] = torch.nan

    ratio = periodic_phase_power_ratio(error, spacing=(4, 4))

    assert torch.isfinite(ratio)
