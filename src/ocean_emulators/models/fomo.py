from typing import TYPE_CHECKING

import torch
import xarray as xr
from einops import rearrange
from perceiver_pytorch import Perceiver
from perceiver_pytorch.perceiver_pytorch import Attention, FeedForward
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from ocean_emulators.constants import Grid
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules import PerceiverEncoder
from ocean_emulators.models.modules.unet_backbone import UNetBackbone

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
        encoder: PerceiverEncoder,
        processor: UNetBackbone,
        add_3d_coordinates: nn.Module,
        hist: int,
        wet: Grid,
        static_data: xr.Dataset | None,
        checkpointing: "Checkpointing | None",
        gradient_detach_interval: int,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            wet=wet,
            hist=hist,
            pred_residuals=pred_residuals,
            last_kernel_size=last_kernel_size,
            pad=pad,
            static_data=static_data,
            gradient_detach_interval=gradient_detach_interval,
        )
        self.patch_size = encoder.patch_size

        # Placeholder decoder is a non-globe aware Conv2d.
        layers = [
            add_3d_coordinates,
            encoder,
            processor,
            nn.Conv2d(
                processor.out_channels,
                out_channels,
                last_kernel_size,
                padding=last_kernel_size // 2,
            ),
        ]
        self.layers = nn.ModuleList(layers)
        self.unpatch = nn.Linear(
            out_channels, out_channels * self.patch_size[0] * self.patch_size[1]
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

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            fts = layer(fts)

        # Unpatchify: project to patch area, then reshape back to original spatial dimensions
        _, _, h, w = fts.shape
        fts = rearrange(fts, "b l h w -> b h w l")
        fts = self.unpatch(fts)  # (b, h, w, out_channels * ph * pw)
        fts = rearrange(
            fts,
            "b h w (c ph pw) -> b c (h ph) (w pw)",
            c=self.out_channels,
            ph=self.patch_size[0],
            pw=self.patch_size[1],
            h=h,
            w=w,
        )

        return torch.where(self.wet, fts, 0.0)
