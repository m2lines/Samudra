import torch
import torch.nn as nn
import torch.utils.checkpoint
import xarray as xr
from aurora.model.posencoding import lat_lon_meshgrid

from ocean_emulators.constants import Grid, Lat, Lon
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules.unet_backbone import UNetBackbone


class Samudra(BaseModel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pred_residuals: bool,
        last_kernel_size: int,
        pad: str,
        unet: UNetBackbone,
        corrector: nn.Module | None,
        pos_channels: int,
        add_2d_coordinates: bool,
        lat: Lat,
        lon: Lon,
        hist: int,
        wet: Grid,
        static_data: xr.Dataset | None,
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

        if pos_channels > 0:
            self.positional_params = nn.Parameter(
                torch.empty(pos_channels, *wet.shape[-2:])
            )
            nn.init.normal_(self.positional_params, mean=0.0, std=1e-5)
        else:
            self.register_parameter("positional_params", None)

        self.lat, self.lon = lat, lon
        self.add_2d_coordinates = add_2d_coordinates

        layers = [
            # Add UNet core.
            unet,
            # Samudra "decoder".
            nn.Conv2d(unet.out_channels, out_channels, last_kernel_size),
        ]

        self.layers = nn.ModuleList(layers)
        self.corrector = corrector

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        fts_input = fts.clone().detach()

        if self.positional_params is not None:
            pos = self.positional_params.unsqueeze(0).expand(fts.shape[0], -1, -1, -1)
            fts = torch.cat([fts, pos], dim=1)

        if self.add_2d_coordinates:
            grid = lat_lon_meshgrid(self.lat, self.lon)
            # Normalize the grid
            grid = (grid - grid.mean()) / grid.std()
            grid = (
                grid.float().to(fts.device).unsqueeze(0).repeat(fts.shape[0], 1, 1, 1)
            )
            fts = torch.cat((fts, grid), dim=1)

        for layer in self.layers:
            # Circular/Globe padding
            if isinstance(layer, nn.Conv2d):
                fts = torch.nn.functional.pad(
                    fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                fts = torch.nn.functional.pad(
                    fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )

            # TODO(alxmrs): Find a clean way to checkpoint the decoder Conv block.

            # Apply layer
            fts = layer(fts)

        if self.corrector is not None:
            fts = self.corrector(fts_input, fts)
        return torch.where(self.wet, fts, 0.0)
