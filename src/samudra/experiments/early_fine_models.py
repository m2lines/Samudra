# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fine-task I/O around the unchanged coarse-grid recurrent processor."""

import torch
from torch import nn
from torch.nn import functional as F


class PeriodicConv(nn.Conv2d):
    def forward(self, value):
        value = F.pad(value, (1, 1, 0, 0), mode="circular")
        return super().forward(F.pad(value, (0, 0, 1, 1)))


class FineEncoder(nn.Module):
    """Encode 19 surface/forcing/validity frames; never consume full truth."""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            PeriodicConv(19 * 7 + 5, 32, 3, stride=2),
            nn.GELU(),
            PeriodicConv(32, 32, 3, stride=2),
            nn.GELU(),
            nn.Conv2d(32, 2 * channels, 1),
        )

    def forward(self, surface, forcing, validity, context):
        inputs = torch.cat(
            (
                surface.flatten(1, 2),
                validity.flatten(1, 2).float(),
                context,
                forcing.flatten(1, 2),
            ),
            1,
        )
        result = self.net(inputs).float()
        return result.reshape(result.shape[0], 2, self.channels, *result.shape[-2:])


class FineDecoder(nn.Module):
    """Decode each predicted coarse state, including any recurrent latent slots."""

    def __init__(self, channels, physical_channels=77):
        super().__init__()
        self.physical_channels = physical_channels
        self.net = nn.Sequential(
            PeriodicConv(channels + 5, 64, 3),
            nn.GELU(),
            nn.Conv2d(64, physical_channels * 16, 1),
            nn.PixelShuffle(4),
        )

    def forward(self, state, context, mask):
        residual = self.net(torch.cat((state, context), 1)).float()
        reference = F.interpolate(
            F.pad(
                state[:, : self.physical_channels].float(),
                (1, 1, 0, 0),
                mode="circular",
            ),
            scale_factor=4,
            mode="bilinear",
            align_corners=False,
        )[..., 4:-4]
        if residual.shape[-2:] != mask.shape[-2:]:
            raise ValueError("Fine decoder grid does not match native target")
        return (reference + residual) * mask
