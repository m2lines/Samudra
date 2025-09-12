import torch
import xarray as xr
from torch import nn

from ocean_emulators.constants import Grid
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules import PerceiverEncoder
from ocean_emulators.models.modules.unet_backbone import UNetBackbone


class FOMO(BaseModel):
    """A placeholder FOMO model. It currently combines an encoder and processor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pred_residuals: bool,
        last_kernel_size: int,
        pad: str,
        encoder: PerceiverEncoder,
        processor: UNetBackbone,
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
        # TODO(alxmrs): Properly wire up the encoder with the processor.
        self.layers = [encoder, processor]
        self.layers.append(
            # Placeholder decoder -- ignoring global padding for now.
            nn.Conv2d(processor.out_channels, out_channels, last_kernel_size),
        )

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            fts = layer(fts)
        return fts
