# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""The deterministic U-global baseline built with Samudra backbone configuration.

The only recurrent state is two frames of normalized physical fields (77 in the
baseline). A 19-frame, five-day surface history initializes them. Observation
normalization is supplied by the data preparation step, never fitted here.
"""

import math
from collections.abc import Sequence
from typing import Self

import torch
from pydantic import Field, model_validator
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from samudra.config import BlockConfig, UNetBackboneConfig
from samudra.config_base import BaseConfig


def baseline_backbone(
    widths: Sequence[int] = (128, 192, 256, 384),
) -> UNetBackboneConfig:
    """U-global defaults expressed through the shared U-Net and block configs."""
    return UNetBackboneConfig(
        ch_width=list(widths),
        dilation=[2**i for i in range(len(widths))],
        n_layers=[1] * len(widths),
        core_block=BlockConfig(norm="instance", upscale_factor=2),
    )


class SurfaceInitializedConfig(BaseConfig):
    """Surface initializer and evolution networks share a backbone definition."""

    backbone: UNetBackboneConfig = Field(default_factory=baseline_backbone)
    activation_checkpointing: bool = True

    @model_validator(mode="after")
    def valid_backbone(self) -> Self:
        cfg = self.backbone
        if not cfg.ch_width or any(width <= 0 for width in cfg.ch_width):
            raise ValueError("U-Net widths must be positive")
        if not len(cfg.ch_width) == len(cfg.dilation) == len(cfg.n_layers):
            raise ValueError("Backbone widths, dilations and layer counts must align")
        if any(value <= 0 for value in (*cfg.dilation, *cfg.n_layers)):
            raise ValueError("Backbone dilations and layer counts must be positive")
        return self

    def build(self, names: Sequence[str]) -> "SurfaceInitializedModel":
        return SurfaceInitializedModel(
            names,
            backbone=self.backbone,
            activation_checkpointing=self.activation_checkpointing,
        )


def make_unet(inputs: int, outputs: int, config: UNetBackboneConfig) -> nn.Sequential:
    """Build through shared configuration, with U-global's affine InstanceNorm."""
    backbone = config.build(inputs, pad="circular", checkpointing=None)
    # BlockConfig's InstanceNorm has no affine option. Keep this local adaptation
    # until normalization configuration can be shared without changing old models.
    for module in tuple(backbone.modules()):
        for name, layer in tuple(module.named_children()):
            if isinstance(layer, nn.InstanceNorm2d) and not layer.affine:
                setattr(
                    module, name, nn.InstanceNorm2d(layer.num_features, affine=True)
                )
    return nn.Sequential(backbone, nn.Conv2d(config.ch_width[0], outputs, 1))


def identity_adapters(channels):
    # Identity adapters must not perturb the backbone initialization RNG.
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
    """Rotate sine/cosine season planes; leave supplied spherical xyz alone."""
    angle = 2 * math.pi * days / 365.25
    result = context.clone()
    result[:, -2] = context[:, -2] * math.cos(angle) + context[:, -1] * math.sin(angle)
    result[:, -1] = context[:, -1] * math.cos(angle) - context[:, -2] * math.sin(angle)
    return result


def geometry_planes(context, mask):
    """U-global spacing/extent/wetness planes from actual spherical coordinates."""
    xyz = context[:, :3].float()
    h, w = xyz.shape[-2:]
    if min(h, w) < 2:
        raise ValueError("Geometry needs at least two cells in each direction")
    dx = (xyz[..., 1:] - xyz[..., :-1]).square().sum(1, keepdim=True).sqrt()
    dy = (xyz[..., 1:, :] - xyz[..., :-1, :]).square().sum(1, keepdim=True).sqrt()
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    lon = torch.atan2(xyz[:, 1:2], xyz[:, 0:1])
    delta = lon[..., 1:] - lon[..., :-1]
    delta = torch.atan2(delta.sin(), delta.cos()).abs().mean((-2, -1), keepdim=True)
    lat = xyz[:, 2:3].clamp(-1, 1).asin()
    height = (lat[..., 1:, :] - lat[..., :-1, :]).abs().mean((-2, -1), keepdim=True) * h
    return torch.cat(
        (
            100 * dx,
            100 * dy,
            (delta * w).expand_as(dx),
            height.expand_as(dx),
            torch.zeros_like(dx),  # All examples are global; no regional tasks.
            mask.any(0)[None, None].expand_as(dx).float(),
            mask.float().mean(0)[None, None].expand_as(dx),
        ),
        1,
    )


class HistoryInitializer(nn.Module):
    history = 19

    def __init__(self, names: Sequence[str], backbone: UNetBackboneConfig):
        super().__init__()
        self.channels = len(names)
        self.surface = [names.index("thetao_0"), names.index("zos")]
        self.net = make_unet(self.history * 7 + 5, 2 * self.channels, backbone)
        self.input_adapters = identity_adapters(self.history * 7 + 5)

    def forward(
        self, surface, past_forcing, context, mask, input_mask=None, task="observation"
    ):
        surface = surface[:, -self.history :]
        if surface.ndim != 5 or surface.shape[1:3] != (self.history, 2):
            raise ValueError("Initialization requires 19 frames of SST and SSH")
        b, _, _, h, w = surface.shape
        masks = mask[self.surface].bool().expand_as(surface)
        if input_mask is not None:
            if input_mask[:, -self.history :].shape != surface.shape:
                raise ValueError("Per-frame surface validity must match the history")
            masks = masks & input_mask[:, -self.history :].bool()
        # Invalid values never enter the network, even if callers use NaN fill.
        surface = torch.where(masks, surface, 0)
        inputs = torch.cat(
            (
                surface.flatten(1, 2),
                masks.to(surface.dtype).flatten(1, 2),
                context,
                past_forcing[:, -self.history :].flatten(1, 2),
            ),
            1,
        )
        result = self.net(self.input_adapters[task](inputs)).float()
        result = result.reshape(b, 2, self.channels, h, w)
        # Copy only measured wet surfaces; predict genuinely missing surfaces.
        result[:, :, self.surface] = torch.where(
            masks[:, -2:], surface[:, -2:], result[:, :, self.surface]
        )
        return result * mask


class Evolution(nn.Module):
    input_adapters: nn.ModuleDict

    def __init__(self, channels: int, backbone: UNetBackboneConfig):
        super().__init__()
        inputs = 2 * channels + 8
        self.geometry = nn.Conv2d(7, inputs, 1, bias=False)
        nn.init.zeros_(self.geometry.weight)
        self.net = make_unet(inputs, channels, backbone)
        # Input adapters attached after the ERA5 adapter to retain construction order.

    def forward(self, states, forcing, context, mask, lead, task="observation"):
        inputs = torch.cat((states.flatten(1, 2), context, forcing[:, lead - 1]), 1)
        inputs = self.input_adapters[task](inputs)
        inputs = inputs + self.geometry(geometry_planes(context, mask))
        return self.net(inputs).float() * mask


class SurfaceInitializedModel(nn.Module):
    """U-global with configurable backbones; OM4 and CM4 share the ``om4`` adapter."""

    def __init__(
        self,
        names: Sequence[str],
        widths: Sequence[int] | None = None,
        activation_checkpointing: bool = True,
        *,
        backbone: UNetBackboneConfig | None = None,
    ):
        super().__init__()
        self.names = tuple(names)
        if widths is not None and backbone is not None:
            raise ValueError("Supply widths or backbone configuration, not both")
        config = SurfaceInitializedConfig(
            backbone=backbone
            or baseline_backbone((128, 192, 256, 384) if widths is None else widths),
            activation_checkpointing=activation_checkpointing,
        )
        self.initializer = HistoryInitializer(names, config.backbone)
        self.evolution = Evolution(len(names), config.backbone)
        self.adapter = nn.Sequential(
            nn.Conv2d(8, 32, 1), nn.GELU(), nn.Conv2d(32, 3, 1)
        )
        final = self.adapter[-1]
        assert isinstance(final, nn.Conv2d) and final.bias is not None
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.evolution.input_adapters = identity_adapters(2 * len(names) + 8)
        self.activation_checkpointing = activation_checkpointing

    def call(self, function, *args):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def adapt(self, atmosphere):
        b, t, _, h, w = atmosphere.shape
        return self.adapter(atmosphere.reshape(b * t, 8, h, w)).reshape(b, t, 3, h, w)

    def forcing(self, values, task):
        if task not in ("observation", "om4"):
            raise ValueError("Task must be observation or om4 (also used for CM4)")
        expected = 8 if task == "observation" else 3
        if values.ndim != 5 or values.shape[2] != expected:
            raise ValueError(f"{task} forcing requires {expected} channels")
        return self.adapt(values) if task == "observation" else values

    def initialize(
        self, surface, atmosphere, context, mask, validity, task="observation"
    ):
        return self.call(
            self.initializer,
            surface,
            self.forcing(atmosphere, task),
            context,
            mask,
            validity,
            task,
        )

    def continue_forecast(
        self, states, forcing, contexts, mask, start_lead=1, task="observation"
    ):
        """Continue cached physical states without reading another ocean input.

        ``start_lead`` is the absolute first output lead (one-based five-day
        steps). Forcing and context are already sliced to that lead by the data
        loader: do not rotate their calendar again. U-global has no learned lead
        embedding. Returns predictions and the last two states for replay;
        detaching those states is the caller's responsibility.
        """
        if start_lead < 1 or int(start_lead) != start_lead:
            raise ValueError("start_lead must be a positive integer")
        if states.ndim != 5 or states.shape[1:3] != (2, len(self.names)):
            raise ValueError("Replay state must contain exactly two physical frames")
        if contexts.ndim != 5 or contexts.shape[:2] != forcing.shape[:2]:
            raise ValueError("Future forcing and context frames must align")
        if forcing.shape[1] == 0:
            raise ValueError("A continuation needs at least one future step")
        forcing = self.forcing(forcing, task)
        predictions = []
        for step in range(forcing.shape[1]):
            predicted = self.call(
                self.evolution,
                states,
                forcing[:, step : step + 1],
                contexts[:, step],
                mask,
                1,
                task,
            )
            predictions.append(predicted)
            states = torch.stack((states[:, -1], predicted), 1)
        return torch.stack(predictions, 1), states

    def forecast(
        self, surface, atmosphere, contexts, mask, validity, task="observation"
    ):
        initial = self.initialize(
            surface[:, :19],
            atmosphere[:, :19],
            contexts[:, 18],
            mask,
            validity[:, :19],
            task,
        )
        predictions, _ = self.continue_forecast(
            initial, atmosphere[:, 19:], contexts[:, 19:], mask, task=task
        )
        return predictions, initial

    def reconstruct_month(
        self, surface, atmosphere, contexts, mask, validity, month_weights
    ):
        """Weighted monthly interior reconstruction from contemporaneous history."""
        if len(month_weights) == 0:
            raise ValueError("Monthly reconstruction needs at least one weight")
        mean = None
        for step, weight in enumerate(month_weights):
            end = 20 + step
            initial = self.initialize(
                surface[:, end - 19 : end],
                atmosphere[:, end - 19 : end],
                contexts[:, end - 1],
                mask,
                validity[:, end - 19 : end],
            )
            value = initial[:, -1] * weight
            mean = value if mean is None else mean + value
        return mean
