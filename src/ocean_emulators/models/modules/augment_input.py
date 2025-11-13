import torch
from aurora.model.posencoding import lat_lon_meshgrid
from torch import nn

from ocean_emulators.constants import Lat, Lon


class Add3dCoordinates(nn.Module):
    """Add 3d Cartesian Coordinates on a unit sphere to the channel dimension.

    This provides better pole handling than raw lat/lon coordinates.
    """

    def __init__(self, lat: Lat, lon: Lon):
        super().__init__()
        self.lat, self.lon = lat, lon

    def forward(self, fts: torch.Tensor) -> torch.Tensor:
        lat_lon_grid = lat_lon_meshgrid(self.lat, self.lon)  # [2, H, W]
        lat_rad = torch.deg2rad(lat_lon_grid[0])  # [H, W]
        lon_rad = torch.deg2rad(lat_lon_grid[1])  # [H, W]

        # Compute Cartesian coordinates (naturally bounded in [-1, 1])
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)

        grid = torch.stack([x, y, z], dim=0)  # [3, H, W]
        grid = grid.float().to(fts.device).unsqueeze(0).expand(fts.shape[0], -1, -1, -1)
        fts = torch.cat((fts, grid), dim=1)

        return fts
