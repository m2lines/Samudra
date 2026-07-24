# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from samudra.constants import Boundary, Prognostic
from samudra.models.base import BaseModel
from samudra.utils.ctx import GridContext
from samudra.utils.device import autocast


class Otter(BaseModel):
    """Otter-inspired shifted-window Transformer ocean emulator."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pred_residuals: bool,
        last_kernel_size: int,
        pad: str,
        backbone: nn.Module,
        hist: int,
        gradient_detach_interval: int,
        use_bfloat16: bool,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hist=hist,
            pred_residuals=pred_residuals,
            last_kernel_size=last_kernel_size,
            pad=pad,
            gradient_detach_interval=gradient_detach_interval,
        )
        self.backbone = backbone
        self.use_bfloat16 = use_bfloat16

    def forward_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: GridContext
    ) -> Prognostic:
        features = torch.cat((prognostic, boundary), dim=1)
        latitude, longitude = ctx.input_resolution_cpu
        if features.shape[-2:] != (latitude.numel(), longitude.numel()):
            raise ValueError(
                "Input tensor and resolution size mismatch: "
                f"tensor has {features.shape[-2:]} but resolution has "
                f"{(latitude.numel(), longitude.numel())}."
            )

        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            output = self.backbone(features, latitude, longitude)
        output = output.to(torch.float32)
        return torch.where(ctx.label_mask, output, 0.0)
