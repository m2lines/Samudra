# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

import torch

DIAGNOSTICS = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/diagnose_surface_initialization.py")
)


def test_temporal_decomposition_separates_bias_amplitude_and_pattern():
    t = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float64)[:, None, None, None]
    p = 2 * t + 3
    sums = torch.stack((p, t, p * p, t * t, p * t)).sum(1)
    parts = DIAGNOSTICS["temporal_decomposition"](sums, 4)
    torch.testing.assert_close(
        parts[:4, 0, 0, 0], torch.tensor([10.0, 9.0, 1.0, 0.0], dtype=torch.float64)
    )
    p = -t
    parts = DIAGNOSTICS["temporal_decomposition"](
        torch.stack((p, t, p * p, t * t, p * t)).sum(1), 4
    )
    torch.testing.assert_close(
        parts[:4, 0, 0, 0], torch.tensor([4.0, 0.0, 0.0, 4.0], dtype=torch.float64)
    )


def test_masked_box_preserves_wet_constants_and_wraps_longitude():
    mask = torch.ones(1, 5, 8, dtype=torch.bool)
    mask[:, 2, 3] = 0
    result = DIAGNOSTICS["masked_box"](7.0 * mask[None], mask, 3)
    torch.testing.assert_close(result, 7.0 * mask[None])
    impulse = torch.zeros(1, 1, 5, 8)
    impulse[0, 0, 2, 0] = 1
    smooth = DIAGNOSTICS["masked_box"](impulse, torch.ones_like(mask), 3)
    assert smooth[0, 0, 2, -1] > 0
    assert smooth[0, 0, 2, 4] == 0
