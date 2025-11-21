from typing import TYPE_CHECKING

import torch
import xarray as xr

from ocean_emulators.constants import Grid
from ocean_emulators.models.base import BaseModel

if TYPE_CHECKING:
    from ocean_emulators.config import Checkpointing


class HilT(BaseModel):
    """Hilbert Transformer Ocean Emulator.

    See https://openreview.net/forum?id=ltYXDRLDGW.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pred_residuals: bool,
        last_kernel_size: int,
        pad: str,
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

        # TODO(alxmrs): Add in activation checkpointing
        _ = checkpointing

    def forward_once(self, fts: torch.Tensor) -> torch.Tensor:
        return torch.where(self.wet, fts, 0.0)
