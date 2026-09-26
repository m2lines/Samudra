# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Joint physical-field diffusion conditioned throughout on surface-derived features."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from samudra.experiments.initializer_diffusion import OceanConv, ResidualBlock


class JointInteriorDecoder(nn.Module):
    """Shared parameters for different target grids; the conditioning grid stays fixed."""

    def __init__(self, conditions, fields, width=64):
        super().__init__()
        self.fields = fields
        self.embedding = nn.Sequential(
            nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.stem = OceanConv(fields, width)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(width, width),
                ResidualBlock(width, 2 * width),
                ResidualBlock(2 * width, 4 * width),
            ]
        )
        self.condition = nn.ModuleList(
            [nn.Conv2d(conditions, n * width, 1) for n in (1, 1, 2)]
        )
        self.middle = ResidualBlock(4 * width, 4 * width)
        self.up = nn.ModuleList(
            [ResidualBlock(6 * width, 2 * width), ResidualBlock(3 * width, width)]
        )
        self.head = OceanConv(width, fields)
        # A nonzero head allows initializer gradients on the first fitting step.
        nn.init.normal_(self.head.weight, std=0.001)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.bias)

    def forward(self, noisy, sigma, latent, mask):
        sigma = sigma.reshape(-1, 1, 1, 1)
        frequency = torch.exp(
            torch.arange(32, device=noisy.device).float() * (-math.log(10000) / 31)
        )
        phase = sigma.flatten()[:, None].log() / 4 * frequency[None]
        embedding = self.embedding(torch.cat((phase.sin(), phase.cos()), 1))
        x = self.stem(noisy * torch.rsqrt(1 + sigma.square()))
        skips = []
        for i, (block, conditioning) in enumerate(
            zip(self.blocks, self.condition, strict=True)
        ):
            if i:
                x = F.avg_pool2d(x, 2)
            # Qualification only: shape-based nearest interpolation. H production
            # must replace this with registered target-coordinate geometry.
            x = x + F.interpolate(
                conditioning(latent), size=x.shape[-2:], mode="nearest"
            )
            x = block(x, embedding)
            skips.append(x)
        x = self.middle(x, embedding)
        for block, skip in zip(self.up, reversed(skips[:-1]), strict=True):
            x = block(
                torch.cat(
                    (F.interpolate(x, size=skip.shape[-2:], mode="nearest"), skip), 1
                ),
                embedding,
            )
        return (
            noisy / (1 + sigma.square())
            + sigma * torch.rsqrt(1 + sigma.square()) * self.head(F.silu(x)).float()
        ) * mask


def channel_balanced_mse(prediction, target, weights):
    """Each supported channel has equal mass, independent of its wet area/grid size."""
    denominator = weights.sum((-2, -1))
    if not bool((denominator > 0).any()):
        raise ValueError("No supervised wet cells")
    result = ((prediction - target).square() * weights).sum(
        (-2, -1)
    ) / denominator.clamp_min(1e-12)
    supported = denominator > 0
    return (result * supported).sum(-1) / supported.sum(-1).clamp_min(1)


def denoising_loss(
    decoder, latent, target, mask, weights, generator, *, known_mask=None
):
    sigma = (
        torch.randn(target.shape[0], device=target.device, generator=generator) * 1.2
        - 1.2
    ).exp()
    noise = torch.randn(target.shape, device=target.device, generator=generator) * mask
    noisy = target + sigma[:, None, None, None] * noise
    if known_mask is not None:
        if known_mask.shape not in (mask.shape, target.shape):
            raise ValueError("Known mask must match the shared mask or batched target")
        noisy = torch.where(known_mask.bool(), target, noisy)
        weights = weights * ~known_mask.bool()
    prediction = decoder(noisy, sigma, latent, mask)
    return (
        channel_balanced_mse(prediction, target, weights)
        * (1 + sigma.square())
        / sigma.square()
    ).mean()


def sample_joint(
    decoder,
    latent,
    mask,
    generator,
    steps=16,
    *,
    known_values=None,
    known_mask=None,
    checkpoint_denoiser=False,
):
    """Heun sampling, optionally differentiable for observation-operator losses.

    Known channels are conditioning values, not stochastic targets. Their state
    and decoded values stay fixed at every integration stage. Production training
    must apply the same convention to the noisy denoising inputs.
    """
    from torch.utils.checkpoint import checkpoint

    if steps < 2:
        raise ValueError("At least two integration intervals are required")
    if (known_values is None) != (known_mask is None):
        raise ValueError("Provide both known values and their mask")
    shape = (latent.shape[0], decoder.fields, *mask.shape[-2:])
    if mask.shape[-3] != decoder.fields:
        raise ValueError("Target mask does not match decoder fields")

    def anchor(value):
        if known_mask is not None:
            value = torch.where(known_mask.bool(), known_values, value)
        return value * mask

    def denoise(value, sigma):
        sigma_batch = sigma.expand(latent.shape[0])
        if checkpoint_denoiser and torch.is_grad_enabled():
            result = checkpoint(
                decoder, value, sigma_batch, latent, mask, use_reentrant=False
            )
        else:
            result = decoder(value, sigma_batch, latent, mask)
        return anchor(result.float())

    sigmas = (
        80 ** (1 / 7)
        + torch.linspace(0, 1, steps, device=latent.device)
        * (0.002 ** (1 / 7) - 80 ** (1 / 7))
    ) ** 7
    sigmas = torch.cat((sigmas, sigmas.new_zeros(1)))
    state = anchor(
        torch.randn(shape, device=latent.device, generator=generator) * sigmas[0]
    )
    for current, following in zip(sigmas[:-1], sigmas[1:], strict=True):
        derivative = (state - denoise(state, current)) / current
        trial = anchor(state + (following - current) * derivative)
        if following > 0:
            next_derivative = (trial - denoise(trial, following)) / following
            trial = anchor(
                state + (following - current) * (derivative + next_derivative) / 2
            )
        state = trial
    return state
