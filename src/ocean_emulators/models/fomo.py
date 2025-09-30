import torch
import xarray as xr
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from ocean_emulators.constants import Grid
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules import PerceiverDecoder, PerceiverEncoder
from ocean_emulators.models.modules.unet_backbone import UNetBackbone


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
        decoder: PerceiverDecoder,
        hist: int,
        wet: Grid,
        static_data: xr.Dataset | None,
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
        )
        self.layers = nn.ModuleList([encoder, processor, decoder])

        apply_activation_checkpointing(
            self,
            # check_fn=lambda m: m.__class__.__name__
            # in ["LayerNorm", "FeedForward", "Linear", "Perceiver"],
        )

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            fts = layer(fts)

        return torch.where(self.wet, fts, 0.0)
