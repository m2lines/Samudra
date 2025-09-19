import torch
import xarray as xr
from einops import rearrange
from torch import nn

from ocean_emulators.constants import Grid
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules import PerceiverEncoder
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
        self.patch_size = encoder.patch_size

        # Placeholder decoder is a non-globe aware Conv2d.
        layers = [
            encoder,
            processor,
            nn.Conv2d(processor.out_channels, out_channels, last_kernel_size),
        ]
        self.layers = nn.ModuleList(layers)
        self.unpatch = nn.Linear(
            out_channels, out_channels * self.patch_size[0] * self.patch_size[1]
        )

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            fts = layer(fts)

        # Get current patch-level dimensions
        _, _out_channels, h_patches, w_patches = fts.shape

        # project latent to output channels × patch area
        fts = rearrange(fts, "b l h w -> b h w l")
        fts = self.unpatch(fts)  # (b, h, w, out_channels * ph * pw)

        # Unpatchify: reshape back to original spatial dimensions
        fts = rearrange(
            fts,
            "b h w (c ph pw) -> b c (h ph) (w pw)",
            c=self.out_channels,
            ph=self.patch_size[0],
            pw=self.patch_size[1],
            h=h_patches,
            w=w_patches,
        )

        return torch.where(self.wet, fts, 0.0)
