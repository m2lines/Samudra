# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.utils.checkpoint

from samudra.constants import Boundary, GridSize, Prognostic
from samudra.models.base import BaseModel
from samudra.models.modules.unet_backbone import UNetBackbone
from samudra.utils.ctx import BatchGrid
from samudra.utils.device import autocast


class Samudra(BaseModel):
    """Samudra ocean emulator using a ConvNeXt U-Net backbone.

    Implements the Samudra (and Samudra 2) model architecture for
    single-scale ocean emulation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pred_residuals: bool,
        last_kernel_size: int,
        pad: str,
        unet: UNetBackbone,
        pos_channels: int,
        add_3d_coordinates: nn.Module | None,
        input_steps: int,
        grid_size: GridSize,
        gradient_detach_interval: int,
        use_bfloat16: bool,
        persistence_gate: bool = False,
        persistence_gate_kind: str = "features",
        prognostic_channels: int | None = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            input_steps=input_steps,
            pred_residuals=pred_residuals,
            last_kernel_size=last_kernel_size,
            pad=pad,
            gradient_detach_interval=gradient_detach_interval,
        )

        if pos_channels > 0:
            self.positional_params = nn.Parameter(torch.empty(pos_channels, *grid_size))
            nn.init.normal_(self.positional_params, mean=0.0, std=1e-5)
        else:
            self.register_parameter("positional_params", None)

        self.add_3d_coordinates = add_3d_coordinates
        self.unet = unet
        self.decoder = nn.Conv2d(unet.out_channels, out_channels, last_kernel_size)

        self.persistence_gate_kind = persistence_gate_kind
        self.persistence_gate: nn.Module | None = None
        if persistence_gate:
            if pred_residuals:
                raise ValueError("persistence_gate requires pred_residuals=False")
            if persistence_gate_kind not in ("features", "direct", "nonlinear"):
                raise ValueError("Unknown persistence gate kind")
            gate_inputs = unet.out_channels
            if persistence_gate_kind != "features":
                if prognostic_channels is None or prognostic_channels % input_steps:
                    raise ValueError("Direct gates require complete prognostic frames")
                gate_inputs += prognostic_channels // input_steps + out_channels
            # Preserve common parameters and the subsequent RNG stream across arms.
            with torch.random.fork_rng(devices=[]):
                if persistence_gate_kind == "nonlinear":
                    projection = nn.Conv2d(64, out_channels, 1)
                    self.persistence_gate = nn.Sequential(
                        nn.Conv2d(gate_inputs, 64, 1), nn.GELU(), projection
                    )
                else:
                    projection = nn.Conv2d(gate_inputs, out_channels, 1)
                    self.persistence_gate = projection
            nn.init.zeros_(projection.weight)
            assert projection.bias is not None
            nn.init.constant_(projection.bias, -6.0)

        self.use_bfloat16 = use_bfloat16

    def forward_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: BatchGrid
    ) -> Prognostic:
        # Samudra is a single-scale model; fuse prognostic + boundary into
        # the single channel-stacked input its backbone expects.
        fts = torch.cat((prognostic, boundary), dim=1)

        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            if self.positional_params is not None:
                pos = self.positional_params.unsqueeze(0).expand(
                    fts.shape[0], -1, -1, -1
                )
                fts = torch.cat([fts, pos], dim=1)

            if self.add_3d_coordinates is not None:
                fts = self.add_3d_coordinates(fts, ctx.input_resolution_cpu)

            fts = self.unet(fts)
            decoder_features = fts
            fts = torch.nn.functional.pad(
                fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
            )
            fts = torch.nn.functional.pad(
                fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
            )
        # TODO(jder): would be nice to keep inputs in bfloat16 and
        # have the convolution use float32 internally & in output dtype.
        fts = fts.to(torch.float32)
        fts = self.decoder(fts)

        if self.persistence_gate is not None:
            channels = prognostic.shape[1] // self.input_steps
            if prognostic.shape[1] % self.input_steps or self.out_channels % channels:
                raise ValueError("Gate requires complete prognostic frames")
            latest = prognostic[:, -channels:].repeat(
                1, self.out_channels // channels, 1, 1
            )
            gate_inputs = decoder_features.float()
            if self.persistence_gate_kind != "features":
                gate_inputs = torch.cat(
                    (gate_inputs, prognostic[:, -channels:], fts), dim=1
                )
            gate = self.persistence_gate(gate_inputs).sigmoid()
            fts = fts + gate * (latest - fts)

        return torch.where(ctx.label_mask, fts, 0.0)
