# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Explicit-state baselines for the surface-initialized ocean experiment.

Inputs/outputs are normalized physical fields. Only the initializer extracts
surface observations from history; forecasting never receives future interiors.
"""

import math

import torch
from torch import nn

from samudra.config import BlockConfig, UNetBackboneConfig


def make_unet(inputs: int, outputs: int, widths: list[int]) -> nn.Module:
    backbone = UNetBackboneConfig(
        ch_width=widths,
        dilation=[2**i for i in range(len(widths))],
        n_layers=[1] * len(widths),
        core_block=BlockConfig(upscale_factor=2),
    ).build(inputs, pad="circular", checkpointing=None)
    return nn.Sequential(backbone, nn.Conv2d(widths[0], outputs, 1))


def geographic_features(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """Three spherical coordinates, with shape (3, latitude, longitude)."""
    lat, lon = torch.meshgrid(torch.deg2rad(lat), torch.deg2rad(lon), indexing="ij")
    return torch.stack((lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()))


class Initializer(nn.Module):
    def __init__(self, names: list[str], widths: list[int], history: int = 6):
        super().__init__()
        self.channels = len(names)
        self.history = history
        self.surface = [names.index("thetao_0"), names.index("zos")]
        # Surface values and masks at each time; xyz and annual sine/cosine.
        self.net = make_unet(4 * history + 5, 2 * self.channels, widths)

    def forward(self, history, context, mask):
        b, _, h, w = history.shape
        states = history.reshape(b, self.history, self.channels, h, w)
        surface = states[:, :, self.surface]
        surface_mask = mask[self.surface].expand(b, self.history, 2, h, w)
        inputs = torch.cat(
            (surface.flatten(1, 2), surface_mask.flatten(1, 2), context), dim=1
        )
        result = self.net(inputs).float().reshape(b, 2, self.channels, h, w)
        # Exact known surfaces at the two output times; no hidden-state shortcut.
        result[:, :, self.surface] = states[:, -2:, self.surface]
        return result * mask


class Evolution(nn.Module):
    def __init__(self, channels: int, widths: list[int], kind: str, leads: int = 6):
        super().__init__()
        self.kind, self.leads = kind, leads
        if kind == "direct":
            # Ordered padded forcing frames and validity bits; order is preserved
            # by channel-specific weights. This is a small temporal MLP per cell.
            self.forcing_encoder = nn.Sequential(
                nn.Conv2d(4 * leads, 32, 1), nn.GELU(), nn.Conv2d(32, 16, 1)
            )
            extra = 17  # forcing embedding plus requested lead
        elif kind == "ar":
            extra = 3
        else:
            raise ValueError(kind)
        self.net = make_unet(2 * channels + 5 + extra, channels, widths)

    def forward(self, states, forcing, context, mask, lead: int):
        b, _, _, h, w = states.shape
        if self.kind == "direct":
            # Slice BEFORE encoding: future forcing after requested lead is absent.
            visible = forcing[:, :lead]
            padded = forcing.new_zeros((b, self.leads, 3, h, w))
            padded[:, :lead] = visible
            validity = forcing.new_zeros((b, self.leads, h, w))
            validity[:, :lead] = 1
            encoded = self.forcing_encoder(
                torch.cat((padded.flatten(1, 2), validity), 1)
            )
            lead_plane = forcing.new_full((b, 1, h, w), lead / self.leads)
            extra = torch.cat((encoded, lead_plane), 1)
        else:
            extra = forcing[:, lead - 1]
        inputs = torch.cat((states.flatten(1, 2), context, extra), 1)
        return self.net(inputs) * mask


class Forecast(nn.Module):
    def __init__(self, initializer: Initializer, evolution: Evolution):
        super().__init__()
        self.initializer = initializer
        self.evolution = evolution

    def forward(self, history, forcing, context, mask, mode: str, leads: list[int]):
        if mode == "inferred":
            states = self.initializer(history, context, mask)
        elif mode == "true":
            states = history.reshape(
                history.shape[0], -1, self.initializer.channels, *history.shape[-2:]
            )[:, -2:]
        else:
            raise ValueError(mode)
        initial = states
        outputs = []
        if self.evolution.kind == "ar":
            for lead in range(1, max(leads) + 1):
                # Advance seasonal phase while leaving geography unchanged.
                step_context = advance_season(context, (lead - 1) * 5)
                prediction = self.evolution(states, forcing, step_context, mask, lead)
                if lead in leads:
                    outputs.append(prediction)
                states = torch.stack((states[:, -1], prediction), 1)
        else:
            outputs = [
                self.evolution(states, forcing, context, mask, lead) for lead in leads
            ]
        return torch.stack(outputs, 1), initial


def advance_season(context: torch.Tensor, days: int) -> torch.Tensor:
    angle = 2 * math.pi * days / 365.25
    result = context.clone()
    result[:, -2] = context[:, -2] * math.cos(angle) + context[:, -1] * math.sin(angle)
    result[:, -1] = context[:, -1] * math.cos(angle) - context[:, -2] * math.sin(angle)
    return result


def channel_mse(prediction, target, weights):
    """Area-weighted errors per sample/lead/channel, excluding dry cells."""
    error = (prediction.float() - target.float()).square()
    return (error * weights).sum((-2, -1)) / weights.sum((-2, -1)).clamp_min(1e-12)


def balanced_loss(prediction, target, weights, names, interior_only=False):
    per_channel = channel_mse(prediction, target, weights)
    groups = []
    for variable in ("thetao", "so", "uo", "vo", "zos"):
        indices = [
            i
            for i, name in enumerate(names)
            if (name == variable or name.startswith(variable + "_"))
            and not (interior_only and name in ("thetao_0", "zos"))
        ]
        if indices:
            groups.append(per_channel[..., indices].mean())
    return torch.stack(groups).mean()
