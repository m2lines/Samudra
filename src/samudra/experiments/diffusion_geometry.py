# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinate-registered latent-feature sampling on global rectilinear grids."""

import torch
from torch import nn
from torch.nn import functional as F


class RegisteredGridMap(nn.Module):
    """Bilinear feature interpolation in physical latitude/periodic longitude.

    This maps conditioning features, not conservative cell averages or targets.
    Source latitude/longitude must describe the actual latent locations. Target
    coordinates may change while the source representation remains untouched.
    Outside the source latitude centers use the nearest polar feature row.
    """

    grid: torch.Tensor

    def __init__(self, source_lat, source_lon, target_lat, target_lon):
        super().__init__()
        slat, slon, tlat, tlon = [
            torch.as_tensor(v, dtype=torch.float64)
            for v in (source_lat, source_lon, target_lat, target_lon)
        ]
        for value in (slat, slon, tlat, tlon):
            if (
                value.ndim != 1
                or not bool(torch.isfinite(value).all())
                or value.numel() < 1
            ):
                raise ValueError(
                    "Coordinates must be finite nonempty one-dimensional arrays"
                )
        for value in (slat, slon):
            if value.numel() < 2 or not bool((value.diff() > 0).all()):
                raise ValueError(
                    "Source coordinates must strictly increase with at least two centers"
                )
        if bool((slat.abs() > 90).any()) or bool((tlat.abs() > 90).any()):
            raise ValueError("Latitude must be in degrees within [-90,90]")
        if slon[-1] - slon[0] >= 360:
            raise ValueError(
                "Global source longitude must omit its duplicate cyclic endpoint"
            )
        self.source_shape = (len(slat), len(slon))

        def fractional_index(source, target):
            upper = torch.searchsorted(source, target).clamp(1, len(source) - 1)
            lower = upper - 1
            return lower + (target - source[lower]) / (source[upper] - source[lower])

        y = fractional_index(slat, tlat).clamp(0, len(slat) - 1)
        extended = torch.cat((slon[-1:] - 360, slon, slon[:1] + 360))
        longitude = torch.remainder(tlon - slon[0], 360) + slon[0]
        x = fractional_index(extended, longitude)
        yy, xx = torch.meshgrid(
            2 * y / (len(slat) - 1) - 1, 2 * x / (len(extended) - 1) - 1, indexing="ij"
        )
        self.register_buffer("grid", torch.stack((xx, yy), -1).float()[None])
        # Geometry is auditable checkpoint metadata, not a learned input shortcut.
        for name, value in [
            ("source_lat", slat),
            ("source_lon", slon),
            ("target_lat", tlat),
            ("target_lon", tlon),
        ]:
            self.register_buffer(name, value)

    def forward(self, latent):
        if tuple(latent.shape[-2:]) != self.source_shape:
            raise ValueError(
                "Latent grid differs from its registered source coordinates"
            )
        padded = F.pad(latent.float(), (1, 1, 0, 0), mode="circular")
        return F.grid_sample(
            padded,
            self.grid.to(device=latent.device, dtype=torch.float32).expand(
                latent.shape[0], -1, -1, -1
            ),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
