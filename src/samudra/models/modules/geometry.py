# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Static physical cell dimensions for global Gaussian longitude/latitude grids."""

import copy

import numpy as np
import torch
from torch import nn

from samudra.models.modules.blocks import ConvNeXtBlock


def gaussian_geometry(lat, lon, levels: int, reference_m: float = 100_000.0):
    """Return log(dx/L0), log(dy/L0) at the exact footprints of 2x2 pooling.

    Gaussian quadrature weights define latitude cell edges in sin(latitude).
    At coarser levels we merge edges, dropping trailing cells just as AvgPool2d
    does. dx is the zonal arc at the cell's area-midpoint latitude; dy is the
    meridional edge-to-edge arc. Coordinates are degrees, south to north.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1 or min(len(lat), len(lon)) < 2:
        raise ValueError("Geometry requires 1-D global Gaussian coordinates")
    nodes, weights = np.polynomial.legendre.leggauss(len(lat))
    if not np.allclose(lat, np.rad2deg(np.arcsin(nodes)), atol=2e-4, rtol=0):
        raise ValueError("Geometry requires ascending Gaussian latitudes")
    spacing = 360.0 / len(lon)
    if not np.allclose(np.diff(lon), spacing, atol=2e-4, rtol=0):
        raise ValueError("Geometry requires uniform global longitudes")
    if reference_m <= 0:
        raise ValueError("reference_m must be positive")
    sin_edges = np.r_[-1.0, -1.0 + np.cumsum(weights)]
    lat_edges = np.arcsin(np.clip(sin_edges, -1, 1))
    lon_edges = np.deg2rad(lon[0] - spacing / 2 + np.arange(len(lon) + 1) * spacing)
    result = []
    for depth in range(levels):
        # Area centroid in sin(latitude), consistent with Gaussian cell areas.
        mid_sin = (np.sin(lat_edges[1:]) + np.sin(lat_edges[:-1])) / 2
        dx = (
            6_371_000.0 * np.sqrt(1 - mid_sin**2)[:, None] * np.diff(lon_edges)[None, :]
        )
        dy = np.broadcast_to(6_371_000.0 * np.diff(lat_edges)[:, None], dx.shape)
        result.append(
            torch.from_numpy(
                np.log(np.stack([dx, dy]) / reference_m).astype(np.float32)
            ).unsqueeze(0)
        )
        if depth + 1 < levels:
            lat_edges = lat_edges[: 2 * ((len(lat_edges) - 1) // 2) + 1 : 2]
            lon_edges = lon_edges[: 2 * ((len(lon_edges) - 1) // 2) + 1 : 2]
            if min(len(lat_edges), len(lon_edges)) < 2:
                raise ValueError("Too many U-Net levels for this geometry")
    return result


class GeometryConditioning(nn.Module):
    """Shared pointwise geometry encoder and zero-initialized block heads."""

    def __init__(self, widths: list[int], levels: int, reference_m: float, film: bool):
        super().__init__()
        self.levels = levels
        self.reference_m = reference_m
        self._coordinates: tuple | None = None
        self._maps: list[torch.Tensor] = []
        self.encoder = (
            nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 16), nn.SiLU())
            if film
            else nn.Identity()
        )
        self.heads = nn.ModuleList(
            [nn.Linear(16, 2 * c) for c in widths] if film else []
        )
        for head in self.heads:
            assert isinstance(head, nn.Linear)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def maps(self, resolution, device):
        lat, lon = resolution
        key = (tuple(lat.tolist()), tuple(lon.tolist()), device)
        if key != self._coordinates:
            self._maps = [
                g.to(device)
                for g in gaussian_geometry(lat, lon, self.levels, self.reference_m)
            ]
            self._coordinates = key
        return self._maps

    def forward(self, h, g, block_index: int):
        encoded = self.encoder(g.permute(0, 2, 3, 1))
        gamma, beta = (
            self.heads[block_index](encoded).permute(0, 3, 1, 2).chunk(2, dim=1)
        )
        return (1 + gamma.to(h.dtype)) * h + beta.to(h.dtype)


def concatenate_geometry_inputs(block: ConvNeXtBlock):
    """Widen only input projections, preserving baseline hidden widths/weights.

    This is literal concatenation into both the first spatial convolution and
    its residual projection. Additional columns start at zero.
    """

    def widen(conv):
        if not isinstance(conv, nn.Conv2d):
            raise ValueError("Geometry concatenation requires Conv2d input projections")
        if conv.groups != 1:
            raise ValueError(
                "Geometry concatenation requires ungrouped input projections"
            )
        new = copy.deepcopy(conv)
        new.in_channels += 2
        extra = conv.weight.new_zeros(conv.out_channels, 2, *conv.kernel_size)
        new.weight = nn.Parameter(torch.cat((conv.weight.detach(), extra), dim=1))
        return new

    block.convblock[0] = widen(block.convblock[0])
    block.skip_module = widen(block.skip_module)
