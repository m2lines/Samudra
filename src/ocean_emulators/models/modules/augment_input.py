import torch
from aurora.model.fourier import FourierExpansion
from aurora.model.posencoding import lat_lon_meshgrid
from einops import rearrange, repeat
from jaxtyping import Float
from torch import nn

from ocean_emulators.constants import Lat, Lon


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


class FourierPatchPositionEncoding(nn.Module):
    """Concatenate 2-D Fourier position features to each pixel within a patch.

    Given a ``(batch, ph, pw, channels)`` tensor of patches, this module
    builds a normalised ``[-1, 1]`` spatial grid over ``(ph, pw)``, expands
    it with :class:`~aurora.model.fourier.FourierExpansion`, and concatenates
    the result along the channel dimension.

    The output has ``channels + 2 * fourier_dim`` features (one set of
    ``fourier_dim`` features per spatial axis).

    Args:
        fourier_dim: Number of Fourier features per spatial axis.  Must be
            even (required by :class:`FourierExpansion`).
    """

    def __init__(self, fourier_dim: int) -> None:
        super().__init__()
        self.fourier_dim = fourier_dim
        self.expansion = FourierExpansion(lower=0.01, upper=2.0, assert_range=False)

    def forward(
        self,
        patches: Float[torch.Tensor, "batch ph pw channels"],
    ) -> Float[torch.Tensor, "batch ph pw channels_plus_fourier"]:
        _, ph, pw, _ = patches.shape
        device, dtype = patches.device, patches.dtype

        lat_pos = torch.linspace(-1.0, 1.0, steps=ph, device=device, dtype=dtype)
        lon_pos = torch.linspace(-1.0, 1.0, steps=pw, device=device, dtype=dtype)
        grid = torch.stack(
            torch.meshgrid(lat_pos, lon_pos, indexing="ij"), dim=-1
        )  # (ph, pw, 2)

        # FourierExpansion: (ph, pw, 2) → (ph, pw, 2, fourier_dim)
        enc_pos = self.expansion(grid, self.fourier_dim)
        enc_pos = rearrange(enc_pos, "ph pw ax d -> ph pw (ax d)")
        enc_pos = repeat(enc_pos, "ph pw d -> b ph pw d", b=patches.shape[0])

        return torch.cat([patches, enc_pos], dim=-1)


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
