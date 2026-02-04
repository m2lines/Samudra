from typing import TYPE_CHECKING

import torch
from einops import rearrange
from perceiver_pytorch import Perceiver
from perceiver_pytorch.perceiver_pytorch import Attention, FeedForward
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from ocean_emulators.constants import GridContext
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules import PerceiverEncoder
from ocean_emulators.models.modules.encoder import patch_from
from ocean_emulators.models.modules.unet_backbone import UNetBackbone
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
        hist: int,
        checkpointing: "Checkpointing | None",
        gradient_detach_interval: int,
        all_grids: list[tuple[int, int]],
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
        self.use_bfloat16 = use_bfloat16
        # Placeholder decoder is a non-globe aware Conv2d.
        self.decoder = nn.Conv2d(
            processor.out_channels,
            out_channels,
            last_kernel_size,
            padding=last_kernel_size // 2,
        )
        all_patches = [
            patch_from(self.encoder.patch_extent, *grid) for grid in all_grids
        ]

        self.unpatch = nn.ModuleDict(
            {
                str(patch_size): nn.Linear(
                    out_channels, out_channels * patch_size[0] * patch_size[1]
                )
                for patch_size in all_patches
            }
        )

        if checkpointing == "all":
            apply_activation_checkpointing(
                self,
                check_fn=lambda m: isinstance(
                    m,
                    nn.LayerNorm
                    | FeedForward
                    | nn.Linear
                    | Perceiver
                    | PerceiverEncoder
                    | UNetBackbone
                    | Attention,
                ),
            )

    def forward_once(self, fts: torch.Tensor, ctx: GridContext) -> torch.Tensor:
        _, _, H, W = fts.shape

        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            if self.maybe_add_3d_coordinates is not None:
                fts = self.maybe_add_3d_coordinates(fts, ctx.input_resolution)
            fts = self.encoder(fts, ctx.input_resolution)
            fts = self.processor(fts)

        # Convert back to float32 for decoder and unpatchify operations
        fts = fts.to(torch.float32)
        fts = self.decoder(fts)

        # Unpatchify: project to patch area, then reshape back to original spatial dimensions
        patch_size = patch_from(self.encoder.patch_extent, H, W)
        _, _, h, w = fts.shape
        fts = rearrange(fts, "b l h w -> b h w l")
        fts = self.unpatch[str(patch_size)](fts)  # (b, h, w, out_channels * ph * pw)
        fts = rearrange(
            fts,
            "b h w (c ph pw) -> b c (h ph) (w pw)",
            c=self.out_channels,
            ph=patch_size[0],
            pw=patch_size[1],
            h=h,
            w=w,
        )

        return torch.where(ctx.label_mask, fts, 0.0)
