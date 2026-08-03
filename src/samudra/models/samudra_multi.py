# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

import torch
from perceiver_pytorch import Perceiver
from perceiver_pytorch.perceiver_pytorch import Attention, FeedForward
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from samudra.constants import Boundary, Prognostic
from samudra.models.base import BaseModel
from samudra.models.modules import PerceiverDecoder, PerceiverEncoder
from samudra.models.modules.unet_backbone import UNetBackbone
from samudra.utils.ctx import GridContext
from samudra.utils.device import autocast

if TYPE_CHECKING:
    from samudra.config import Checkpointing

_checkpoint_types: tuple[type, ...] = (
    nn.LayerNorm,
    FeedForward,
    nn.Linear,
    Perceiver,
    PerceiverDecoder,
    PerceiverEncoder,
    UNetBackbone,
    Attention,
)

try:
    from flash_attn.modules.block import (
        Block as FlashBlock,  # type: ignore[import-not-found]
    )
    from flash_perceiver.perceiver import (
        PerceiverBase as FlashPerceiverBase,  # type: ignore[import-not-found]
    )

    _checkpoint_types = _checkpoint_types + (FlashPerceiverBase, FlashBlock)
except ImportError:
    pass


class SamudraMulti(BaseModel):
    """Multi-resolution encoder-processor-decoder model.

    Currently, this model is used only as a physical ocean emulator.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pred_residuals: bool,
        last_kernel_size: int,
        pad: str,
        add_3d_coordinates: nn.Module | None,
        encoder: PerceiverEncoder,
        processor: UNetBackbone,
        decoder: PerceiverDecoder,
        hist: int,
        checkpointing: "Checkpointing | None",
        gradient_detach_interval: int,
        use_bfloat16: bool,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hist=hist,
            pred_residuals=pred_residuals,
            last_kernel_size=last_kernel_size,
            pad=pad,
            gradient_detach_interval=gradient_detach_interval,
        )

        self.maybe_add_3d_coordinates = add_3d_coordinates
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.use_bfloat16 = use_bfloat16

        if checkpointing == "all":
            apply_activation_checkpointing(
                self,
                check_fn=lambda m: isinstance(m, _checkpoint_types),
            )

    def set_epoch(self, epoch: int) -> None:
        self.processor.set_epoch(epoch)

    def forward_once(
        self, prognostic: Prognostic, boundary: Boundary, ctx: GridContext
    ) -> Prognostic:
        # Prognostic and boundary are carried as separate tensors through the
        # data pipeline, but this encoder still expects a single concatenated
        # input.  The dual-perceiver encoder that fuses them at the token level
        # (enabling cross-resolution) lands in a follow-up PR.
        fts = torch.cat((prognostic, boundary), dim=1)
        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            if self.maybe_add_3d_coordinates is not None:
                fts = self.maybe_add_3d_coordinates(fts, ctx.input_resolution_cpu)
            fts = self.encoder(fts, ctx.input_resolution_cpu)
            fts = self.processor(fts)

            fts = self.decoder(fts, ctx.output_resolution_cpu)

        # Convert back to float32
        # TODO(alxmrs): We actually only support float16 when turned on; this kind of tricks us.
        fts = fts.to(torch.float32)

        return torch.where(ctx.label_mask, fts, 0.0)
