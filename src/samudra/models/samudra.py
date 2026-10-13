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
        auxiliary_ke_outputs: int = 0,
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

        self.use_bfloat16 = use_bfloat16
        self.auxiliary_head = None
        if auxiliary_ke_outputs:
            # Preserve shared weights and the subsequent sampler RNG sequence.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(271828)
                self.auxiliary_head = nn.Conv2d(
                    unet.out_channels, auxiliary_ke_outputs, 1
                )

    def forward_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: BatchGrid
    ) -> Prognostic:
        fts = self.forward_features(prognostic, boundary, ctx)
        return torch.where(ctx.label_mask, self.decoder(fts), 0.0)

    def forward_training_step(self, prognostic, boundary, batch, step):
        if self.auxiliary_head is None:
            raise ValueError("Auxiliary targets require the KE head")
        fts = self.forward_features(prognostic, boundary, batch.ctx)
        prediction = torch.where(batch.ctx.label_mask, self.decoder(fts), 0.0)
        interior = (
            fts[:, :, self.N_pad : -self.N_pad, self.N_pad : -self.N_pad]
            if self.N_pad
            else fts
        )
        auxiliary = self.auxiliary_head(interior)
        target = batch.auxiliary_targets[step]
        if auxiliary.shape != target.shape:
            raise ValueError("KE head and target shapes differ")
        loss = (
            ((auxiliary - target).square() * batch.auxiliary_weights)
            .sum((-2, -1))
            .mean()
        )
        return prediction, loss

    def forward_features(
        self, prognostic: Prognostic, boundary: Boundary, ctx: BatchGrid
    ) -> torch.Tensor:
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
            fts = torch.nn.functional.pad(
                fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
            )
            fts = torch.nn.functional.pad(
                fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
            )
        # TODO(jder): would be nice to keep inputs in bfloat16 and
        # have the convolution use float32 internally & in output dtype.
        fts = fts.to(torch.float32)
        return fts
