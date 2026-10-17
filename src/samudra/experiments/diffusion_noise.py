# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit-marginal-variance Gaussian noise with optional spatial covariance."""

import math

import torch
from torch.nn import functional as F


def diffusion_noise(shape, mask, generator, correlation=0.0, paired_draws=False):
    """Mix independent white and Gaussian-filtered white fields.

    correlation is the variance fraction in the filtered component, not a pair
    correlation coefficient. A five-cell Gaussian kernel has standard deviation
    one grid cell. Longitude is periodic, latitude zero-padded. Wet-cell variance
    is normalized analytically, including coastlines. No cross-channel coupling.
    The nonzero white component makes the covariance positive definite on wet
    cells. This is a grid-space diagnostic, not a fixed-km ocean covariance model.
    """
    if not 0 <= correlation < 1:
        raise ValueError("Correlated variance fraction must be in [0, 1)")
    white = torch.randn(shape, device=mask.device, generator=generator) * mask
    if correlation == 0 and not paired_draws:
        return white
    independent = torch.randn(shape, device=mask.device, generator=generator)
    if correlation == 0:
        return white
    with torch.autocast(device_type=mask.device.type, enabled=False):
        m = mask.float().expand(shape).reshape(-1, 1, *shape[-2:])
        axis = torch.arange(-2, 3, device=mask.device, dtype=torch.float32)
        kernel = torch.exp(-axis.square() / 2)
        kernel = (kernel[:, None] * kernel[None, :])[None, None]

        def convolve(x, k):
            return F.conv2d(
                F.pad(F.pad(x, (2, 2, 0, 0), mode="circular"), (0, 0, 2, 2)), k
            )

        smooth = convolve(independent.reshape_as(m) * m, kernel)
        variance = convolve(m.square(), kernel.square())
        smooth = (smooth / variance.clamp_min(1e-12).sqrt()).reshape(shape) * mask
        return math.sqrt(1 - correlation) * white + math.sqrt(correlation) * smooth
