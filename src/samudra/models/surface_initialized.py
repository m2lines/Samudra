# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Initialize physical ocean states from surface history, then evolve them.

The recurrent state contains two frames of normalized physical fields. The data
preparation step supplies the normalization and history length. Tensor annotations
use batch, time, channel, latitude (height) and longitude (width) axes in that order.
"""

from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal, NamedTuple, ParamSpec, Self, cast

import torch
from jaxtyping import Bool, Float
from pydantic import model_validator
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from samudra.config import UNetBackboneConfig
from samudra.config_base import BaseConfig
from samudra.observations.constants import (
    CONTEXT_CHANNELS,
    DOMAINS,
    GEOMETRY_CHANNELS,
    HISTORY_STEPS,
    LAST_HISTORY_STEP,
    MODEL_FORCING_CHANNELS,
    OBSERVATION_FORCING_CHANNELS,
    SPHERICAL_COORDINATE_CHANNELS,
    STATE_HISTORY_STEPS,
    SURFACE_NAMES,
    Domain,
)

SurfaceHistory = Float[Tensor, "batch time surface height width"]
SurfaceValidity = Bool[Tensor, "batch time surface height width"]
ForcingHistory = Float[Tensor, "batch time forcing height width"]
Context = Float[Tensor, "batch context height width"]
ContextHistory = Float[Tensor, "batch time context height width"]
WetMask = Bool[Tensor, "channels height width"]
PhysicalStates = Float[Tensor, "batch state_history channels height width"]
Predictions = Float[Tensor, "batch time channels height width"]
Padding = Literal["circular", "constant", "reflect", "replicate"]

# Unit-sphere neighbor distances at one degree are approximately 0.017. Scaling
# by 100 gives order-one spacing inputs alongside the other geometry features.
GEOMETRY_DISTANCE_SCALE = 100.0
ERA5_ENCODER_WIDTH = 32
# Forcing channels each domain supplies, before its encoder maps them to the
# model's wind stress and surface heat-flux channels.
DOMAIN_FORCING_CHANNELS: dict[Domain, int] = {
    "observation": OBSERVATION_FORCING_CHANNELS,
    "om4-family": MODEL_FORCING_CHANNELS,
}
OperationArgs = ParamSpec("OperationArgs")


class ForecastContinuation(NamedTuple):
    predictions: Predictions
    states: PhysicalStates  # Last STATE_HISTORY_STEPS frames, for the next call.


class InitializedForecast(NamedTuple):
    predictions: Predictions
    initial_states: PhysicalStates


class SurfaceInitializedConfig(BaseConfig):
    """Surface initializer and evolution networks share a backbone definition.

    For regional grids, select a nonperiodic ``pad`` and a nonperiodic backbone
    upsampling block, such as ``bilinear_upsample``.
    """

    backbone: UNetBackboneConfig
    pad: Padding = "circular"
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
            pad=self.pad,
            activation_checkpointing=self.activation_checkpointing,
        )


def make_unet(
    inputs: int, outputs: int, config: UNetBackboneConfig, pad: Padding
) -> nn.Sequential:
    """Build a shared U-Net followed by a physical-channel projection."""
    backbone = config.build(inputs, pad=pad, checkpointing=None)
    return nn.Sequential(backbone, nn.Conv2d(config.ch_width[0], outputs, 1))


def identity_adapters(channels: int) -> nn.ModuleDict:
    """Create one identity channel projection per input domain."""
    # Identity adapters must not perturb the backbone initialization RNG.
    with torch.random.fork_rng(devices=[]):
        adapters = nn.ModuleDict()
        for domain in DOMAINS:
            layer = nn.Conv2d(channels, channels, 1)
            nn.init.dirac_(layer.weight)
            assert layer.bias is not None
            nn.init.zeros_(layer.bias)
            adapters[domain] = layer
    return adapters


def geometry_planes(
    context: Context, mask: WetMask
) -> Float[Tensor, "batch geometry height width"]:
    """Return spacing x/y, extent x/y, reserved, wet-any and wet-fraction planes.

    Context begins with unit-sphere x/y/z coordinates. Spacing is chord distance
    scaled by GEOMETRY_DISTANCE_SCALE; extent is measured in radians. Wetness is
    computed across physical channels. The reserved geometry plane is zero.
    """
    xyz = context[:, :SPHERICAL_COORDINATE_CHANNELS].float()
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
            GEOMETRY_DISTANCE_SCALE * dx,
            GEOMETRY_DISTANCE_SCALE * dy,
            (delta * w).expand_as(dx),
            height.expand_as(dx),
            torch.zeros_like(dx),
            mask.any(0)[None, None].expand_as(dx).float(),
            mask.float().mean(0)[None, None].expand_as(dx),
        ),
        1,
    )


class HistoryInitializer(nn.Module):
    # Changing HISTORY_STEPS also changes the prepared monthly history windows.
    # Update observations/constants.py and regenerate monthly samples/normalization;
    # daily shards can be reused. A different history needs newly trained weights.
    history = HISTORY_STEPS

    def __init__(
        self, names: Sequence[str], backbone: UNetBackboneConfig, pad: Padding
    ) -> None:
        super().__init__()
        self.channels = len(names)
        self.surface = [names.index(name) for name in SURFACE_NAMES]
        # Per step: measured surfaces, their validity, and physical forcing.
        history_channels = 2 * len(SURFACE_NAMES) + MODEL_FORCING_CHANNELS
        inputs = self.history * history_channels + CONTEXT_CHANNELS
        self.net = make_unet(inputs, STATE_HISTORY_STEPS * self.channels, backbone, pad)
        self.input_adapters = identity_adapters(inputs)

    def forward(
        self,
        surface: SurfaceHistory,
        past_forcing: ForcingHistory,
        context: Context,
        mask: WetMask,
        input_mask: SurfaceValidity | None,
        domain: Domain,
    ) -> PhysicalStates:
        surface = surface[:, -self.history :]
        if surface.ndim != 5 or surface.shape[1:3] != (
            self.history,
            len(SURFACE_NAMES),
        ):
            raise ValueError(
                f"Initialization requires {self.history} frames of {SURFACE_NAMES}"
            )
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
        result = self.net(self.input_adapters[domain](inputs)).float()
        result = result.reshape(b, STATE_HISTORY_STEPS, self.channels, h, w)
        # Copy measured wet surfaces; infer missing surfaces.
        result[:, :, self.surface] = torch.where(
            masks[:, -STATE_HISTORY_STEPS:],
            surface[:, -STATE_HISTORY_STEPS:],
            result[:, :, self.surface],
        )
        return result * mask


class Evolution(nn.Module):
    input_adapters: nn.ModuleDict

    def __init__(
        self, channels: int, backbone: UNetBackboneConfig, pad: Padding
    ) -> None:
        super().__init__()
        inputs = (
            STATE_HISTORY_STEPS * channels + CONTEXT_CHANNELS + MODEL_FORCING_CHANNELS
        )
        self.geometry = nn.Conv2d(GEOMETRY_CHANNELS, inputs, 1, bias=False)
        nn.init.zeros_(self.geometry.weight)
        self.net = make_unet(inputs, channels, backbone, pad)
        # Input adapters are attached after the forcing encoders are built.

    def forward(
        self,
        states: PhysicalStates,
        forcing: Float[Tensor, "batch forcing height width"],
        context: Context,
        mask: WetMask,
        domain: Domain,
    ) -> Float[Tensor, "batch channels height width"]:
        inputs = torch.cat((states.flatten(1, 2), context, forcing), 1)
        inputs = self.input_adapters[domain](inputs)
        inputs = inputs + self.geometry(geometry_planes(context, mask))
        return self.net(inputs).float() * mask


class SurfaceInitializedModel(nn.Module):
    """Surface-initialized model with per-domain encoders around a shared core.

    Each input domain has its own forcing encoder and input adapters; the
    initializer and evolution networks are shared. OM4 and CM4 share the
    ``om4-family`` domain.
    """

    def __init__(
        self,
        names: Sequence[str],
        *,
        backbone: UNetBackboneConfig,
        pad: Padding = "circular",
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.names = tuple(names)
        config = SurfaceInitializedConfig(
            backbone=backbone,
            pad=pad,
            activation_checkpointing=activation_checkpointing,
        )
        self.initializer = HistoryInitializer(names, config.backbone, config.pad)
        self.evolution = Evolution(len(names), config.backbone, config.pad)
        # ERA5 fields are mapped to model forcing; model-family forcing passes
        # through. The ERA5 encoder starts at zero forcing.
        era5 = nn.Sequential(
            nn.Conv2d(OBSERVATION_FORCING_CHANNELS, ERA5_ENCODER_WIDTH, 1),
            nn.GELU(),
            nn.Conv2d(ERA5_ENCODER_WIDTH, MODEL_FORCING_CHANNELS, 1),
        )
        self.forcing_encoders = nn.ModuleDict(
            {"om4-family": nn.Identity(), "observation": era5}
        )
        final = era5[-1]
        assert isinstance(final, nn.Conv2d) and final.bias is not None
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        evolution_inputs = (
            STATE_HISTORY_STEPS * len(names) + CONTEXT_CHANNELS + MODEL_FORCING_CHANNELS
        )
        self.evolution.input_adapters = identity_adapters(evolution_inputs)
        self.activation_checkpointing = activation_checkpointing

    def call(
        self,
        function: Callable[OperationArgs, Tensor],
        *args: OperationArgs.args,
        **kwargs: OperationArgs.kwargs,
    ) -> Tensor:
        """Call a tensor operation, retaining activations according to config."""
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return cast(
                Tensor,
                checkpoint(partial(function, **kwargs), *args, use_reentrant=False),
            )
        return function(*args, **kwargs)

    def encode_forcing(self, forcing: ForcingHistory, domain: Domain) -> ForcingHistory:
        """Map a domain's forcing to model wind stress and surface heat flux."""
        if domain not in DOMAINS:
            raise ValueError(f"Domain must be one of {DOMAINS}")
        expected = DOMAIN_FORCING_CHANNELS[domain]
        if forcing.ndim != 5 or forcing.shape[2] != expected:
            raise ValueError(f"{domain} forcing requires {expected} channels")
        b, t, _, h, w = forcing.shape
        return self.forcing_encoders[domain](
            forcing.reshape(b * t, expected, h, w)
        ).reshape(b, t, MODEL_FORCING_CHANNELS, h, w)

    def initialize(
        self,
        surface: SurfaceHistory,
        forcing: ForcingHistory,
        context: Context,
        mask: WetMask,
        validity: SurfaceValidity,
        *,
        domain: Domain,
    ) -> PhysicalStates:
        return self.call(
            self.initializer,
            surface,
            self.encode_forcing(forcing, domain),
            context,
            mask,
            validity,
            domain,
        )

    def continue_forecast(
        self,
        states: PhysicalStates,
        forcing: ForcingHistory,
        contexts: ContextHistory,
        mask: WetMask,
        *,
        domain: Domain,
    ) -> ForecastContinuation:
        """Continue cached states and return predictions plus the final state pair.

        The step is time-invariant: the caller supplies forcing and context for
        each future step and owns lead-time bookkeeping. It also owns detaching
        the final state pair before saving it for replay.
        """
        if states.ndim != 5 or states.shape[1:3] != (
            STATE_HISTORY_STEPS,
            len(self.names),
        ):
            raise ValueError("Replay state must contain exactly two physical frames")
        if contexts.ndim != 5 or contexts.shape[:2] != forcing.shape[:2]:
            raise ValueError("Future forcing and context frames must align")
        if forcing.shape[1] == 0:
            raise ValueError("A continuation needs at least one future step")
        forcing = self.encode_forcing(forcing, domain)
        predictions: list[Tensor] = []
        for step in range(forcing.shape[1]):
            predicted = self.call(
                self.evolution,
                states,
                forcing[:, step],
                contexts[:, step],
                mask,
                domain,
            )
            predictions.append(predicted)
            states = torch.stack((states[:, -1], predicted), 1)
        return ForecastContinuation(torch.stack(predictions, 1), states)

    def forecast(
        self,
        surface: SurfaceHistory,
        forcing: ForcingHistory,
        contexts: ContextHistory,
        mask: WetMask,
        validity: SurfaceValidity,
        *,
        domain: Domain,
    ) -> InitializedForecast:
        initial = self.initialize(
            surface[:, :HISTORY_STEPS],
            forcing[:, :HISTORY_STEPS],
            contexts[:, LAST_HISTORY_STEP],
            mask,
            validity[:, :HISTORY_STEPS],
            domain=domain,
        )
        predictions, _ = self.continue_forecast(
            initial,
            forcing[:, HISTORY_STEPS:],
            contexts[:, HISTORY_STEPS:],
            mask,
            domain=domain,
        )
        return InitializedForecast(predictions, initial)
