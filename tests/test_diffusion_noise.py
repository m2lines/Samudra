# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import torch

from samudra.experiments.diffusion_noise import diffusion_noise


def test_white_path_preserves_values_and_rng():
    mask = torch.ones(2, 9, 11)
    mask[:, 2:4, 2:4] = 0
    a, b = (torch.Generator().manual_seed(11) for _ in range(2))
    expected = torch.randn((3, 2, 9, 11), generator=a) * mask
    actual = diffusion_noise(expected.shape, mask, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(a.get_state(), b.get_state())


def test_correlated_noise_unit_variance_coasts_and_periodic_seam():
    torch.set_num_threads(1)
    mask = torch.ones(1, 9, 11)
    mask[:, 3:6, 4:7] = 0
    noise = diffusion_noise(
        (4096, 1, 9, 11), mask, torch.Generator().manual_seed(37), 0.5
    )
    assert not noise[:, :, 3:6, 4:7].any()
    var = noise.var(0, unbiased=False)
    torch.testing.assert_close(
        var[mask.bool()], torch.ones_like(var[mask.bool()]), atol=0.09, rtol=0
    )
    # Adjacent longitude cells correlate across the periodic seam, not across poles.
    seam = (noise[:, 0, 4, 0] * noise[:, 0, 4, -1]).mean()
    poles = (noise[:, 0, 0, 0] * noise[:, 0, -1, 0]).mean()
    assert 0.30 < seam < 0.48
    assert abs(poles) < 0.06


def test_paired_arms_consume_identical_random_streams():
    mask = torch.ones(1, 9, 11)
    white_rng = torch.Generator().manual_seed(31)
    correlated_rng = torch.Generator().manual_seed(31)
    for _ in range(3):
        torch.testing.assert_close(
            torch.randn(1, generator=white_rng),
            torch.randn(1, generator=correlated_rng),
            rtol=0,
            atol=0,
        )
        diffusion_noise((2, 1, 9, 11), mask, white_rng, 0, paired_draws=True)
        diffusion_noise((2, 1, 9, 11), mask, correlated_rng, 0.5, paired_draws=True)
        assert torch.equal(white_rng.get_state(), correlated_rng.get_state())
