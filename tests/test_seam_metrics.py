# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.aggregator.validate.seam import boundary_jump_ratio


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
