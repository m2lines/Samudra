# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch
from one_step_skill import shortwave_power


def test_shortwave_band_retains_four_cell_noise_and_excludes_eight_cell_wave():
    x = torch.arange(32, dtype=torch.float64)
    four_cell = 2 * torch.sin(2 * torch.pi * x / 4)
    eight_cell = 7 * torch.sin(2 * torch.pi * x / 8)
    data = torch.stack([four_cell, eight_cell])[:, None, :]
    wet = torch.ones_like(data, dtype=torch.bool)
    torch.testing.assert_close(
        shortwave_power(data, wet),
        torch.tensor([2.0, 0.0], dtype=torch.float64),
        atol=1e-12,
        rtol=1e-7,
    )


def test_shortwave_band_excludes_coasts_and_marks_unsupported_depths():
    x = torch.arange(32, dtype=torch.float64)
    wave = torch.sin(2 * torch.pi * x / 4)
    data = torch.stack([wave, 10 * wave])[None, :, :]
    wet = torch.ones_like(data, dtype=torch.bool)
    wet[:, 1, 0] = False
    torch.testing.assert_close(
        shortwave_power(data, wet), torch.tensor([0.5], dtype=torch.float64)
    )
    wet[:, 0, 0] = False
    assert torch.isnan(shortwave_power(data, wet)).all()
