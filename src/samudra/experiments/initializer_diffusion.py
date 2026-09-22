# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Small conditional EDM denoiser for one-field initializer residuals."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class OceanConv(nn.Conv2d):
    """Wrap longitude, replicate latitude; never wrap across the poles."""

    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs, 3)

    def forward(self, x):
        return super().forward(
            F.pad(
                F.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="replicate"
            )
        )


class ResidualBlock(nn.Module):
    def __init__(self, inputs, width, embedding=128):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, inputs)
        self.conv1 = OceanConv(inputs, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv2 = OceanConv(width, width)
        self.affine = nn.Linear(embedding, 2 * width)
        self.skip = nn.Conv2d(inputs, width, 1) if inputs != width else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.affine(emb)[:, :, None, None].chunk(2, 1)
        h = self.conv2(F.silu(self.norm2(h) * (1 + scale) + shift))
        return (self.skip(x) + h) / math.sqrt(2)


class ResidualDenoiser(nn.Module):
    """EDM preconditioning with unit training-residual RMS, sigma_data=1."""

    def __init__(self, conditions, width=64):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.stem = OceanConv(conditions + 1, width)
        self.down = nn.ModuleList(
            [
                ResidualBlock(width, width),
                ResidualBlock(width, 2 * width),
                ResidualBlock(2 * width, 4 * width),
            ]
        )
        self.middle = ResidualBlock(4 * width, 4 * width)
        self.up = nn.ModuleList(
            [ResidualBlock(6 * width, 2 * width), ResidualBlock(3 * width, width)]
        )
        self.head = OceanConv(width, 1)
        nn.init.zeros_(self.head.weight)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.bias)

    def forward(self, noisy, sigma, condition, mask):
        sigma = sigma.reshape(-1, 1, 1, 1)
        frequencies = torch.exp(
            torch.arange(32, device=noisy.device).float() * (-math.log(10000) / 31)
        )
        phase = sigma.flatten()[:, None].log() / 4 * frequencies[None]
        emb = self.embedding(torch.cat((phase.sin(), phase.cos()), 1))
        x = self.stem(
            torch.cat((noisy * torch.rsqrt(1 + sigma.square()), condition), 1)
        )
        skips = []
        for i, block in enumerate(self.down):
            if i:
                x = F.avg_pool2d(x, 2)
            x = block(x, emb)
            skips.append(x)
        x = self.middle(x, emb)
        for block, skip in zip(self.up, reversed(skips[:-1]), strict=True):
            x = block(
                torch.cat(
                    (F.interpolate(x, size=skip.shape[-2:], mode="nearest"), skip), 1
                ),
                emb,
            )
        # Preserve FP32 arithmetic around the BF16 network.
        return (
            noisy / (1 + sigma.square())
            + sigma * torch.rsqrt(1 + sigma.square()) * self.head(F.silu(x)).float()
        ) * mask


def area_mean(value, weights):
    return (value * weights).sum((-2, -1)) / weights.sum()


def edm_loss(model, target, condition, mask, weights, generator):
    sigma = (
        torch.randn(target.shape[0], device=target.device, generator=generator) * 1.2
        - 1.2
    ).exp()
    noise = torch.randn(target.shape, device=target.device, generator=generator) * mask
    prediction = model(
        target + sigma[:, None, None, None] * noise, sigma, condition, mask
    )
    weight = (1 + sigma.square()) / sigma.square()
    return (
        area_mean((prediction - target).square(), weights).flatten() * weight
    ).mean()


@torch.no_grad()
def sample_edm(model, condition, mask, generator, steps=24):
    """Deterministic Heun integration; 2*steps-1 denoiser calls per sample."""
    sigmas = (
        80 ** (1 / 7)
        + torch.linspace(0, 1, steps, device=condition.device)
        * (0.002 ** (1 / 7) - 80 ** (1 / 7))
    ) ** 7
    sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
    x = (
        torch.randn(
            (condition.shape[0], 1, *condition.shape[-2:]),
            device=condition.device,
            generator=generator,
        )
        * sigmas[0]
        * mask
    )
    for current, following in zip(sigmas[:-1], sigmas[1:], strict=True):
        denoised = model(x, current.expand(x.shape[0]), condition, mask)
        derivative = (x - denoised) / current
        trial = x + (following - current) * derivative
        if following > 0:
            next_derivative = (
                trial - model(trial, following.expand(x.shape[0]), condition, mask)
            ) / following
            trial = x + (following - current) * (derivative + next_derivative) / 2
        x = trial * mask
    return x


def ensemble_crps(samples, truth):
    """Empirical ensemble CRPS, reduced over members only (not space)."""
    count = samples.shape[0]
    sorted_samples = samples.sort(dim=0).values
    coefficients = (2 * torch.arange(count, device=samples.device) - count + 1).reshape(
        count, *([1] * (samples.ndim - 1))
    )
    return (samples - truth).abs().mean(0) - (sorted_samples * coefficients).sum(
        0
    ) / count**2
