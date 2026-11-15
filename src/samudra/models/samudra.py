# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.utils.checkpoint

from samudra.constants import Boundary, GridSize, Prognostic
from samudra.models.base import BaseModel
from samudra.models.modules.land import LandMode, partial_conv2d
from samudra.models.modules.unet_backbone import UNetBackbone
from samudra.utils.ctx import BatchGrid
from samudra.utils.device import autocast


class Samudra(BaseModel):
    """Samudra ocean emulator using a ConvNeXt U-Net backbone.

    Implements the Samudra (and Samudra 2) model architecture for
    single-scale ocean emulation.

    ``land`` chooses how the network treats land cells. With ``"mask"`` (the
    original behavior), every layer computes over land as if it were ocean, and
    only the output is masked. With ``"omit"``, land is absent from every layer:
    convolutions, normalization, pooling and upsampling see only ocean cells
    (see :mod:`samudra.models.modules.land`), so the forecast does not depend on
    the values on land. Both settings have the same parameters, so a checkpoint
    loads with either.
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
        land: LandMode = "mask",
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
        if land not in ("mask", "omit"):
            raise ValueError(f"land must be 'mask' or 'omit', got {land!r}")
        self.land: LandMode = land

    def _omit_land_from_inputs(
        self, prognostic: Prognostic, boundary: Boundary, ctx: BatchGrid
    ) -> tuple[Prognostic, Boundary, torch.Tensor]:
        """Zero the inputs on land and return the column (surface) ocean mask.

        Each prognostic channel uses its own depth's mask. The boundary channels,
        and every hidden layer, use the column mask: a cell is ocean if it is
        ocean at any depth (the surface mask, since masks nest with depth).
        ``torch.where`` rather than a product, so NaN on land cannot leak.
        """
        label_mask = ctx.label_mask
        n_prog_channels = prognostic.shape[1]
        if n_prog_channels % self.input_steps != 0:
            raise ValueError(
                f"{n_prog_channels} prognostic channels do not split into "
                f"{self.input_steps} input steps"
            )
        n_vars = n_prog_channels // self.input_steps
        if label_mask.shape[0] % n_vars != 0:
            raise ValueError(
                f"label mask has {label_mask.shape[0]} channels, not a multiple "
                f"of the {n_vars} prognostic variables"
            )
        # The label mask tiles the per-variable mask once per output step, and
        # the prognostic input tiles it once per input step.
        prognostic_mask = label_mask[:n_vars].repeat(self.input_steps, 1, 1)
        column = label_mask.any(dim=0)
        prognostic = torch.where(prognostic_mask, prognostic, 0.0)
        boundary = torch.where(column, boundary, 0.0)
        return prognostic, boundary, column.view(1, 1, *column.shape)

    def forward_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: BatchGrid
    ) -> Prognostic:
        ocean: torch.Tensor | None = None
        if self.land == "omit":
            prognostic, boundary, ocean = self._omit_land_from_inputs(
                prognostic, boundary, ctx
            )
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

            if ocean is not None:
                # Positional and coordinate channels are ocean-only too.
                fts = fts * ocean.to(fts.dtype)
                fts = self.unet(fts, ocean)
            else:
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
        if ocean is not None:
            fts = partial_conv2d(fts, self.decoder, ocean, self.N_pad, self.pad)
        else:
            fts = self.decoder(fts)

        return torch.where(ctx.label_mask, fts, 0.0)
