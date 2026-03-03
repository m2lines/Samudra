from typing import TYPE_CHECKING

import torch
from perceiver_pytorch import Perceiver
from perceiver_pytorch.perceiver_pytorch import Attention, FeedForward
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules import PerceiverDecoder, PerceiverEncoder
from ocean_emulators.models.modules.unet_backbone import UNetBackbone
from ocean_emulators.utils.ctx import GridContext
from ocean_emulators.utils.device import autocast

if TYPE_CHECKING:
    from ocean_emulators.config import Checkpointing


class FOMO(BaseModel):
    """FOMO: A Foundation Model for the Oceans + Observations.

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
                check_fn=lambda m: isinstance(
                    m,
                    nn.LayerNorm
                    | FeedForward
                    | nn.Linear
                    | Perceiver
                    | PerceiverDecoder
                    | PerceiverEncoder
                    | UNetBackbone
                    | Attention,
                ),
            )

    def forward_once(self, fts: torch.Tensor, ctx: GridContext) -> torch.Tensor:
        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            if self.maybe_add_3d_coordinates is not None:
                fts = self.maybe_add_3d_coordinates(fts, ctx.input_resolution_cpu)
            fts = self.encoder(fts, ctx.input_resolution_cpu)
            fts = self.processor(fts)
            fts = self.decoder(fts, ctx.input_resolution_cpu)

        # Convert back to float32
        # TODO(alxmrs): We actually only support float16 when turned on; this kind of tricks us.
        fts = fts.to(torch.float32)

        return torch.where(ctx.label_mask, fts, 0.0)
