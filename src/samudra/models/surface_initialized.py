# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Surface-history initialization and shared, task-conditioned physical-state evolution."""

import math

import torch
from pydantic import Field
from torch import nn
from torch.utils.checkpoint import checkpoint

from samudra.config import BlockConfig, UNetBackboneConfig
from samudra.config_base import BaseConfig


def default_backbone():
    return UNetBackboneConfig(
        ch_width=[128, 192, 256, 384],
        dilation=[1, 2, 4, 8],
        n_layers=[1] * 4,
        core_block=BlockConfig(upscale_factor=2, norm="instance", instance_affine=True),
    )


class SurfaceInitializedConfig(BaseConfig):
    """Reusable network components; task scheduling/losses belong to the harness."""

    initializer: UNetBackboneConfig = Field(default_factory=default_backbone)
    processor: UNetBackboneConfig = Field(default_factory=default_backbone)
    activation_checkpointing: bool = True

    def build(self, names: list[str]):
        return SurfaceInitializedModel(names, self)


def make_unet(inputs, outputs, config):
    backbone = config.build(inputs, pad="circular", checkpointing=None)
    return nn.Sequential(backbone, nn.Conv2d(backbone.out_channels, outputs, 1))


def identity_adapters(channels):
    # Preserve the shared backbone's random initialization when adding adapters.
    with torch.random.fork_rng(devices=[]):
        adapters = nn.ModuleDict()
        for task in ("om4", "observation"):
            layer = nn.Conv2d(channels, channels, 1)
            nn.init.dirac_(layer.weight)
            assert layer.bias is not None
            nn.init.zeros_(layer.bias)
            adapters[task] = layer
    return adapters


def advance_season(context, days):
    angle = 2 * math.pi * days / 365.25
    result = context.clone()
    result[:, -2] = context[:, -2] * math.cos(angle) + context[:, -1] * math.sin(angle)
    result[:, -1] = context[:, -1] * math.cos(angle) - context[:, -2] * math.sin(angle)
    return result


class HistoryInitializer(nn.Module):
    def __init__(self, names, config):
        super().__init__()
        self.channels = len(names)
        self.surface = [names.index("thetao_0"), names.index("zos")]
        self.net = make_unet(19 * 7 + 5, 2 * self.channels, config)
        self.input_adapters = identity_adapters(19 * 7 + 5)

    def forward(
        self, surface, past_forcing, context, mask, input_mask=None, task="observation"
    ):
        surface = surface[:, -19:]
        b, _, _, h, w = surface.shape
        masks = (
            mask[self.surface].expand_as(surface)
            if input_mask is None
            else input_mask[:, -19:].to(surface.dtype)
        )
        if masks.shape != surface.shape:
            raise ValueError("Per-frame surface validity must match the history")
        inputs = torch.cat(
            (
                surface.flatten(1, 2),
                masks.flatten(1, 2),
                context,
                past_forcing[:, -19:].flatten(1, 2),
            ),
            1,
        )
        result = (
            self.net(self.input_adapters[task](inputs))
            .float()
            .reshape(b, 2, self.channels, h, w)
        )
        result[:, :, self.surface] = torch.where(
            masks[:, -2:].bool(), surface[:, -2:], result[:, :, self.surface]
        )
        return result * mask


class Evolution(nn.Module):
    input_adapters: nn.ModuleDict

    def __init__(self, channels, config):
        super().__init__()
        self.net = make_unet(2 * channels + 8, channels, config)
        # Attached by the parent after its forcing adapter, matching the recorded initialization.

    def forward(self, states, forcing, context, mask, lead, task="observation"):
        inputs = torch.cat((states.flatten(1, 2), context, forcing[:, lead - 1]), 1)
        return self.net(self.input_adapters[task](inputs)) * mask


class SurfaceInitializedModel(nn.Module):
    """77 physical-labelled state slots, with no extra autoregressive memory slots."""

    def __init__(self, names, config):
        super().__init__()
        self.initializer = HistoryInitializer(names, config.initializer)
        self.evolution = Evolution(len(names), config.processor)
        self.adapter = nn.Sequential(
            nn.Conv2d(8, 32, 1), nn.GELU(), nn.Conv2d(32, 3, 1)
        )
        final = self.adapter[-1]
        assert isinstance(final, nn.Conv2d) and final.bias is not None
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.evolution.input_adapters = identity_adapters(2 * len(names) + 8)
        self.activation_checkpointing = config.activation_checkpointing

    def call(self, function, *args):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def adapt(self, atmosphere):
        b, t, _, h, w = atmosphere.shape
        return self.adapter(atmosphere.reshape(b * t, 8, h, w)).reshape(b, t, 3, h, w)

    def initialize(self, surface, atmosphere, context, mask, validity):
        return self.call(
            self.initializer, surface, self.adapt(atmosphere), context, mask, validity
        )

    def forecast(self, surface, atmosphere, contexts, mask, validity):
        initial = self.initialize(
            surface[:, :19], atmosphere[:, :19], contexts[:, 18], mask, validity[:, :19]
        )
        states = initial
        forcing = self.adapt(atmosphere[:, 19:])
        predictions = []
        for step in range(forcing.shape[1]):
            prediction = self.call(
                self.evolution,
                states,
                forcing[:, step : step + 1],
                contexts[:, 19 + step],
                mask,
                1,
            )
            predictions.append(prediction)
            states = torch.stack((states[:, -1], prediction), 1)
        return torch.stack(predictions, 1), initial

    def reconstruct_month(
        self, surface, atmosphere, contexts, mask, validity, month_weights
    ):
        mean = None
        for step in range(len(month_weights)):
            end = 20 + step
            initial = self.initialize(
                surface[:, end - 19 : end],
                atmosphere[:, end - 19 : end],
                contexts[:, end - 1],
                mask,
                validity[:, end - 19 : end],
            )
            value = initial[:, -1] * month_weights[step]
            mean = value if mean is None else mean + value
        return mean
