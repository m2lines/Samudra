# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from samudra.aggregator.validate.spatial import (
    high_wavenumber_power_ratio,
    patch_seam_jump_ratio,
    zonal_power_spectrum,
)


def test_zonal_power_spectrum_identifies_resolved_mode():
    longitude = torch.arange(32) * (2 * torch.pi / 32)
    wave = torch.sin(4 * longitude)
    data = wave.reshape(1, 1, 1, 32).expand(2, 3, 4, 32)

    spectrum = zonal_power_spectrum(data)

    assert spectrum.shape == (3, 17)
    assert torch.equal(spectrum.argmax(dim=-1), torch.full((3,), 4))


def test_high_wavenumber_power_ratio_compares_upper_band():
    target = torch.ones(2, 9)
    generated = target.clone()
    generated[:, 5:] *= 0.25

    ratio = high_wavenumber_power_ratio(generated, target)

    torch.testing.assert_close(ratio, torch.full((2,), 0.25))


def test_patch_seam_jump_ratio_is_one_for_uniform_jumps():
    latitude = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    longitude = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    error = (latitude + longitude).reshape(1, 1, 8, 12).expand(2, 3, 8, 12)

    ratio = patch_seam_jump_ratio(error, (2, 3))

    torch.testing.assert_close(ratio, torch.ones(3))


def test_patch_seam_jump_ratio_detects_boundary_artifact():
    error = torch.zeros(1, 1, 8, 12)
    error[:, :, :, 3:6] = 4.0
    error[:, :, :, 9:] = 4.0

    ratio = patch_seam_jump_ratio(error, (2, 3))

    assert ratio.item() > 1e6


@pytest.mark.parametrize("patch_size", [(0, 2), (2, 0), (8, 2), (2, 12)])
def test_patch_seam_jump_ratio_rejects_invalid_patch_size(patch_size):
    with pytest.raises(ValueError):
        patch_seam_jump_ratio(torch.zeros(1, 1, 8, 12), patch_size)
