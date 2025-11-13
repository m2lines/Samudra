import torch
from aurora.model.posencoding import lat_lon_meshgrid
from jaxtyping import Float
from torch import nn

from ocean_emulators.constants import Lat, Lon


class Concat3dCoordinates(nn.Module):
    """Add 3d Cartesian Coordinates on a unit sphere to the channel dimension.

    3D coordinates are structured like so:
    (0, 0, 0) is earth center.
    (1, 0, 0) is at lat, lon = (0, 0)
    (0, 1, 0) is at lat, lon = (0, 90)
    (0, 0, 1) is at the North Pole

    This provides better pole handling than raw lat/lon coordinates.
    """

    def __init__(self, lat: Lat, lon: Lon):
        super().__init__()
        lat_lon_grid = lat_lon_meshgrid(lat, lon)  # [2, H, W]
        lat_rad = torch.deg2rad(lat_lon_grid[0])  # [H, W]
        lon_rad = torch.deg2rad(lat_lon_grid[1])  # [H, W]

        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)

        grid = torch.stack([x, y, z], dim=0)  # [3, H, W]
        self.grid = grid.float().unsqueeze(0)

    def forward(
        self, fts: Float[torch.Tensor, "batch channel height width"]
    ) -> Float[torch.Tensor, "batch channel+3 height width"]:
        grid = self.grid.to(fts.device).expand(fts.shape[0], -1, -1, -1)
        fts = torch.cat((fts, grid), dim=1)

        return fts
