# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Area-integrated heat content on the prepared global rectilinear grid."""

import numpy as np


def cell_areas(lat, lon):
    """Spherical m², using the same midpoint cell bounds as observation remapping."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    for coordinate in (lat, lon):
        if (
            coordinate.ndim != 1
            or len(coordinate) < 2
            or not np.all(np.diff(coordinate) > 0)
        ):
            raise ValueError("Coordinates must be one-dimensional and increasing")
    latitude_bounds = np.r_[-90, (lat[:-1] + lat[1:]) / 2, 90]
    longitude_bounds = np.r_[
        lon[0] - (lon[1] - lon[0]) / 2,
        (lon[:-1] + lon[1:]) / 2,
        lon[-1] + (lon[-1] - lon[-2]) / 2,
    ]
    if not np.isclose(longitude_bounds[-1] - longitude_bounds[0], 360):
        raise ValueError("Expected a global longitude grid")
    return (
        6371000.0**2
        * np.diff(np.sin(np.deg2rad(latitude_bounds)))[:, None]
        * np.diff(np.deg2rad(longitude_bounds))[None, :]
    )


def heat_total(column_j_m2, area_m2, support):
    """Total in ZJ on explicit support; missing cells are neither filled nor rescaled."""
    column, area, support = (
        np.asarray(column_j_m2),
        np.asarray(area_m2),
        np.asarray(support, dtype=bool),
    )
    if (
        column.shape[-2:] != area.shape
        or area.shape != support.shape
        or not support.any()
    ):
        raise ValueError("Nonempty, matching grid/support required")
    if not np.isfinite(column[..., support]).all():
        raise ValueError("Nonfinite heat content within selected support")
    return np.sum(np.where(support, column, 0) * area, axis=(-2, -1)) / 1e21
