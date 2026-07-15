# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from aurora.model.posencoding import lat_lon_meshgrid
from einops import rearrange
from jaxtyping import Float
from torch import nn

from samudra.constants import Lat, Lon


def make_3d_coordinate_grid(lat: Lat, lon: Lon) -> Float[torch.Tensor, "3 H W"]:
    """Make 3D Cartesian coordinates on a unit sphere.

    Returns:
        ``(3, H, W)`` tensor of ``(x, y, z)`` unit-sphere coordinates.
    """
    lat_lon_grid = lat_lon_meshgrid(lat, lon)  # [2, H, W]
    lat_rad = torch.deg2rad(lat_lon_grid[0])  # [H, W]
    lon_rad = torch.deg2rad(lat_lon_grid[1])  # [H, W]

    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)

    return torch.stack([x, y, z], dim=0).float()  # [3, H, W]


class Concat3dCoordinates(nn.Module):
    """Add 3d Cartesian Coordinates on a unit sphere to the channel dimension.

    3D coordinates are structured like so:
     x  y  z
    (0, 0, 0) is earth center.
    (1, 0, 0) is at lat, lon = (0, 0)
    (0, 1, 0) is at lat, lon = (0, 90)
    (0, 0, 1) is at the North Pole

    This is known to provide better pole handling than raw lat/lon coordinates, see [1].

    > Note: This module assumes that the data at each lat/lon is located at the center of each
    > grid point! Please ensure this is the case during pre-processing.

    Args:
        lat: A vector of latitudes representing the center of the grid point.
        lon: A vector of longitudes representing the center of the grid point.

    References:
        [1]: https://ar5iv.labs.arxiv.org/html/2410.07472v1#S4.SS9
    """

    def forward(
        self,
        fts: Float[torch.Tensor, "batch channel height width"],
        resolution: tuple[Lat, Lon],
    ) -> Float[torch.Tensor, "batch channel+3 height width"]:
        grid = (
            make_3d_coordinate_grid(*resolution)
            .unsqueeze(0)
            .to(fts.device)
            .expand(fts.shape[0], -1, -1, -1)
        )
        return torch.cat((fts, grid), dim=1)


def fourier_features_2d_dim(num_freq_bands: int) -> int:
    """Channel-dim contribution of `FourierFeatures2D(num_freq_bands)`."""
    return 2 * (2 * num_freq_bands + 1)


class FourierFeatures2D(nn.Module):
    """Concatenate 2D Fourier positional features along the channel dim.

    Matches the `fourier_encode_data=True, input_axis=2` layout from
    perceiver-pytorch's Perceiver. Configured encoder Perceiver paths use this
    module explicitly so flash and naive implementations encode intra-patch
    position equivalently.

    Frequency layout matches `perceiver_pytorch.fourier_encode`: scales are
    `linspace(1., max_freq / 2, num_freq_bands)`, applied to positions
    in [-1, 1].

    Input:  (..., ph, pw, V)
    Output: (..., ph, pw, V + fourier_features_2d_dim(num_freq_bands))
    """

    def __init__(self, num_freq_bands: int, max_freq: float) -> None:
        super().__init__()
        self.num_freq_bands = num_freq_bands
        self.max_freq = max_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ph, pw = x.shape[-3], x.shape[-2]
        ph_pos = torch.linspace(-1.0, 1.0, steps=ph, device=x.device, dtype=x.dtype)
        pw_pos = torch.linspace(-1.0, 1.0, steps=pw, device=x.device, dtype=x.dtype)
        # (ph, pw, 2)
        pos = torch.stack(torch.meshgrid(ph_pos, pw_pos, indexing="ij"), dim=-1)

        scales = torch.linspace(
            1.0,
            self.max_freq / 2,
            self.num_freq_bands,
            device=x.device,
            dtype=x.dtype,
        )
        # (ph, pw, 2, 1) * (num_bands,) -> (ph, pw, 2, num_bands)
        scaled = pos.unsqueeze(-1) * scales * math.pi
        enc = torch.cat([scaled.sin(), scaled.cos(), pos.unsqueeze(-1)], dim=-1)
        # Flatten (axis, freq) into a single feature dim.
        enc = rearrange(enc, "ph pw ax feat -> ph pw (ax feat)")

        # Broadcast across leading batch dims; concat along channel dim.
        leading = x.shape[:-3]
        enc = enc.expand(*leading, *enc.shape)
        return torch.cat((x, enc), dim=-1)
