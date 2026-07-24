# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from scripts.audit_coarse_inverse import _coarse_patch_means, _coarse_valid_mask


def test_coarse_patch_means_preserve_channel_constants_with_masks() -> None:
    value = torch.stack(
        (
            torch.full((4, 6), 2.0),
            torch.full((4, 6), -3.0),
        )
    ).unsqueeze(0)
    mask = torch.ones(2, 4, 6, dtype=torch.bool)
    mask[0, :2, :3] = False
    latitude = torch.tensor([-67.5, -22.5, 22.5, 67.5])

    mean, valid = _coarse_patch_means(value, mask, latitude, (2, 2))

    assert mean.shape == valid.shape == (1, 2, 2, 2)
    torch.testing.assert_close(mean[valid], value[:, :, ::2, ::3][valid])
    assert not valid[0, 0, 0, 0]
    assert valid[0, 1].all()


def test_coarse_valid_mask_marks_patches_with_any_wet_channel() -> None:
    mask = torch.zeros(2, 4, 6, dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[1, 3, 5] = True

    coarse = _coarse_valid_mask(mask, (2, 2))

    expected = torch.tensor([[[[True, False], [False, True]]]])
    torch.testing.assert_close(coarse, expected)
