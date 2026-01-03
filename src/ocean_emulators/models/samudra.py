import torch
import torch.nn as nn
import torch.utils.checkpoint
import xarray as xr

from ocean_emulators.constants import Lat, Lon, PrognosticMask
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules.unet_backbone import UNetBackbone
from ocean_emulators.utils.device import autocast


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
        add_3d_coordinates: nn.Module | None,
        hist: int,
        grid: tuple[int, int],
        static_data: xr.Dataset | None,
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
            static_data=static_data,
            gradient_detach_interval=gradient_detach_interval,
        )

        if pos_channels > 0:
            self.positional_params = nn.Parameter(torch.empty(pos_channels, *grid))
            nn.init.normal_(self.positional_params, mean=0.0, std=1e-5)
        else:
            self.register_parameter("positional_params", None)

        self.add_3d_coordinates = add_3d_coordinates
        self.unet = unet
        self.decoder = nn.Conv2d(unet.out_channels, out_channels, last_kernel_size)

        self.corrector = corrector
        self.use_bfloat16 = use_bfloat16

    def forward_once(
        self, fts: torch.Tensor, wet: PrognosticMask, resolution: tuple[Lat, Lon]
    ) -> torch.Tensor:
        if self.corrector is not None:
            fts_input = fts.clone().detach()

        with autocast(enabled=self.use_bfloat16, dtype=torch.bfloat16):
            if self.positional_params is not None:
                pos = self.positional_params.unsqueeze(0).expand(
                    fts.shape[0], -1, -1, -1
                )
                fts = torch.cat([fts, pos], dim=1)

            if self.add_3d_coordinates is not None:
                fts = self.add_3d_coordinates(fts)

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
        fts = self.decoder(fts)

        if self.corrector is not None:
            fts = self.corrector(fts_input, fts)
        # Ensure mask is on the same device as fts
        wet = wet.to(device=fts.device)
        return torch.where(wet, fts, 0.0)
