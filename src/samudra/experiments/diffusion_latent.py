# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent latent recurrence shared by deterministic and diffusion readouts."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from samudra.experiments.initializer_diffusion import OceanConv


class LatentHistoryEncoder(nn.Module):
    """Project surface-derived features into two latent memory slots, without interior labels."""

    def __init__(self, encoder, feature_channels, width=128, latent_shape=(45, 90)):
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Conv2d(feature_channels, 2 * width, 1)
        self.width = width
        self.latent_shape = latent_shape

    def forward(self, conditioning):
        features = (
            checkpoint(self.encoder, conditioning, use_reentrant=False)
            if self.training and torch.is_grad_enabled()
            else self.encoder(conditioning)
        )
        latent = F.adaptive_avg_pool2d(self.projection(features), self.latent_shape)
        return latent.reshape(latent.shape[0], 2, self.width, *self.latent_shape)


class LatentResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, width),
            nn.SiLU(),
            OceanConv(width, width),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            OceanConv(width, width),
        )

    def forward(self, value):
        return value + 0.1 * self.layers(value)


class PersistentLatentProcessor(nn.Module):
    """Shift/update latent memory using only fixed-grid coarse forcing and context.

    Decoded fields, target resolution and diffusion noise never enter this module.
    Future losses backpropagate through every latent step to the initializer.
    Pooling is a conditioning-feature operation, not conservative forcing regridding.
    Width/depth are configurable; production values require the prior wave report.
    """

    def __init__(
        self, width=128, depth=4, latent_shape=(45, 90), conditioning_shape=(180, 360)
    ):
        super().__init__()
        if width < 8 or width % 8 or depth < 1:
            raise ValueError(
                "Width must be a positive multiple of eight; depth positive"
            )
        self.width, self.latent_shape, self.conditioning_shape = (
            width,
            latent_shape,
            conditioning_shape,
        )
        self.stem = OceanConv(2 * width + 3 + 5, width)
        self.blocks = nn.Sequential(*[LatentResidualBlock(width) for _ in range(depth)])
        self.head = OceanConv(width, width)
        nn.init.normal_(self.head.weight, std=0.001)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.bias)

    def forward(self, latent, forcing, context):
        if tuple(latent.shape[1:]) != (2, self.width, *self.latent_shape):
            raise ValueError("Latent memory shape changed")
        if tuple(forcing.shape[1:]) != (3, *self.conditioning_shape):
            raise ValueError("Coarse forcing shape changed")
        if tuple(context.shape[1:]) != (5, *self.conditioning_shape):
            raise ValueError("Coarse context shape changed")
        condition = F.adaptive_avg_pool2d(
            torch.cat((forcing, context), 1), self.latent_shape
        )
        update = self.head(
            self.blocks(self.stem(torch.cat((latent.flatten(1, 2), condition), 1)))
        )
        return torch.stack((latent[:, -1], latent[:, -1] + update), 1)

    def rollout(self, initial, forcing, contexts):
        if forcing.shape[:2] != contexts.shape[:2]:
            raise ValueError("Future forcing/context intervals must align")
        current = initial
        result = [initial]
        for step in range(forcing.shape[1]):
            current = (
                checkpoint(
                    self,
                    current,
                    forcing[:, step],
                    contexts[:, step],
                    use_reentrant=False,
                )
                if self.training and torch.is_grad_enabled()
                else self(current, forcing[:, step], contexts[:, step])
            )
            result.append(current)
        return result
