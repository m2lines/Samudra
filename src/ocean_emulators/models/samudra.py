from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.utils.checkpoint
import xarray as xr

from ocean_emulators.constants import Grid
from ocean_emulators.models.base import BaseModel
from ocean_emulators.models.modules.unet_backbone import UNetBackbone
from ocean_emulators.utils.sharding import (
    ActivationLayout,
    create_device_mesh,
    shard_activations,
    shard_pad,
    to_replicated,
)

if TYPE_CHECKING:
    from ocean_emulators.config import ShardingConfig

try:
    from physicsnemo.distributed.shard_tensor import DeviceMesh
except Exception:  # pragma: no cover
    DeviceMesh = None


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
        wet: Grid,
        static_data: xr.Dataset | None,
        gradient_detach_interval: int,
        sharding: ShardingConfig,
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

        self.add_3d_coordinates = add_3d_coordinates
        self.unet = unet
        self.decoder = nn.Conv2d(unet.out_channels, out_channels, last_kernel_size)

        self.corrector = corrector
        self.sharding_cfg = sharding
        self.device_mesh: DeviceMesh | None = None
        if sharding.enable_sharding:
            self.device_mesh = create_device_mesh(sharding.mesh_shape)
            self.activation_layout: ActivationLayout = sharding.activation_layout
        else:
            self.activation_layout = "lon"

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        fts_input = fts.clone().detach()

        if self.positional_params is not None:
            pos = self.positional_params.unsqueeze(0).expand(fts.shape[0], -1, -1, -1)
            fts = torch.cat([fts, pos], dim=1)

        if self.add_3d_coordinates is not None:
            fts = self.add_3d_coordinates(fts)

        using_sharding = (
            self.sharding_cfg.enable_sharding
            and self.device_mesh is not None
            and (self.training or self.sharding_cfg.shard_inference)
        )
        if using_sharding:
            fts = shard_activations(fts, self.device_mesh, self.activation_layout)

        fts = self.unet(fts)
        if using_sharding:
            fts = shard_pad(fts, self.N_pad, lon_mode=self.pad)
        else:
            fts = torch.nn.functional.pad(
                fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
            )
            fts = torch.nn.functional.pad(
                fts, (0, 0, self.N_pad, self.N_pad), mode="constant"
            )
        fts = self.decoder(fts)

        if self.corrector is not None:
            fts = self.corrector(fts_input, fts)
        if using_sharding:
            fts = to_replicated(fts)

        return torch.where(self.wet, fts, 0.0)
