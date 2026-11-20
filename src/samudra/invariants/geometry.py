# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Grid geometry: areas, thicknesses and volumes add up to what the sphere and the bathymetry allow."""

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from samudra.invariants.core import (
    EARTH_MEAN_RADIUS_M,
    Violation,
    first_location,
    relative_difference,
)


def sphere_area(
    area: xr.DataArray, radius: float = EARTH_MEAN_RADIUS_M, rtol: float = 1e-3
) -> list[Violation]:
    """A global grid's cell areas sum to the sphere's area, 4 pi R^2."""
    total = float(area.sum())
    expected = 4 * np.pi * radius**2
    error = relative_difference(total, expected)
    if error <= rtol:
        return []
    implied_radius = np.sqrt(total / (4 * np.pi))
    return [
        Violation(
            "geometry.sphere_area",
            f"total area {total:.6e} m2 differs from 4 pi R^2 = {expected:.6e} m2 by "
            f"{error:.3%}; implied radius {implied_radius / 1e3:.1f} km",
        )
    ]


def rectilinear_cell_area(
    lat_b: ArrayLike, lon_b: ArrayLike, radius: float = EARTH_MEAN_RADIUS_M
) -> np.ndarray:
    """Exact spherical area of each cell of a regular lat-lon grid, from its (y_b, x_b) vertices."""
    lat = np.deg2rad(np.asarray(lat_b)[:, 0])
    lon = np.deg2rad(np.asarray(lon_b)[0, :])
    band = np.diff(np.sin(lat))
    width = np.diff(lon)
    return radius**2 * np.abs(np.outer(band, width))


def areas_match_vertices(
    area: xr.DataArray,
    lat_b: xr.DataArray,
    lon_b: xr.DataArray,
    radius: float = EARTH_MEAN_RADIUS_M,
    rtol: float = 1e-6,
) -> list[Violation]:
    """Each cell's area equals the spherical area its vertices enclose (rectilinear grids only)."""
    lat = np.asarray(lat_b)
    lon = np.asarray(lon_b)
    if not (np.allclose(lat, lat[:, :1]) and np.allclose(lon, lon[:1, :])):
        return [
            Violation(
                "geometry.areas_match_vertices",
                "grid is not rectilinear; this check needs lat_b constant along x and lon_b along y",
            )
        ]
    expected = rectilinear_cell_area(lat_b, lon_b, radius)
    error = np.abs(np.asarray(area) - expected) / expected
    bad = error > rtol
    if not bad.any():
        return []
    flags = xr.DataArray(bad, dims=area.dims)
    return [
        Violation(
            "geometry.areas_match_vertices",
            f"{int(bad.sum())} cells differ from their spherical area by more than "
            f"{rtol:g}; worst {float(error.max()):.3%}, median ratio "
            f"{float(np.median(np.asarray(area) / expected)):.6f}",
            first_location(flags),
        )
    ]


def layer_thickness(
    dz: ArrayLike, ilev: ArrayLike, lev: ArrayLike | None = None
) -> list[Violation]:
    """Layers are positive, tile the column from the surface, and contain their centres.

    ``dz[k] == ilev[k+1] - ilev[k]``, ``ilev[0] == 0``, and ``ilev[k] < lev[k] < ilev[k+1]``.
    """
    violations = []
    dz_values = np.asarray(dz, dtype=float)
    interfaces = np.asarray(ilev, dtype=float)
    if (dz_values <= 0).any():
        violations.append(
            Violation("geometry.dz_positive", f"non-positive thicknesses: {dz_values}")
        )
    if interfaces.size != dz_values.size + 1 or not np.allclose(
        np.diff(interfaces), dz_values
    ):
        violations.append(
            Violation(
                "geometry.dz_matches_interfaces",
                f"dz {dz_values} is not diff(ilev) {np.diff(interfaces)}",
            )
        )
    if interfaces.size and interfaces[0] != 0:
        violations.append(
            Violation("geometry.surface_at_zero", f"ilev[0] = {interfaces[0]}, not 0")
        )
    if lev is not None and interfaces.size == np.asarray(lev).size + 1:
        centres = np.asarray(lev, dtype=float)
        outside = (centres <= interfaces[:-1]) | (centres >= interfaces[1:])
        if outside.any():
            violations.append(
                Violation(
                    "geometry.centres_inside_layers",
                    f"levels {np.flatnonzero(outside).tolist()} have centres outside their interfaces",
                )
            )
    return violations


def column_within_bathymetry(
    dz: xr.DataArray,
    wet: xr.DataArray,
    deptho: xr.DataArray,
    depth_dim: str = "lev",
    atol: float = 1.0,
) -> list[Violation]:
    """A water column's thickness, summed over wet layers, doesn't exceed the sea-floor depth.

    ``dz`` may vary horizontally (partial cells) or not (nominal layers). A
    column deeper than the bathymetry means the bottom cells count water that
    isn't there, which inflates ocean volume (#973).
    """
    column = (dz * wet.astype(bool)).sum(depth_dim)
    excess = column - deptho
    too_deep = (excess > atol) & wet.astype(bool).any(depth_dim)
    if not too_deep.any():
        return []
    return [
        Violation(
            "geometry.column_within_bathymetry",
            f"{int(too_deep.sum())} columns are deeper than the bathymetry; worst by "
            f"{float(excess.where(too_deep).max()):.1f} m",
            first_location(too_deep),
        )
    ]


def ocean_volume(
    area: xr.DataArray, ocean_fraction: xr.DataArray, dz: xr.DataArray
) -> float:
    """Ocean volume by the store convention: sum of area x ocean_fraction x dz."""
    return float((area * ocean_fraction * dz).sum())


def volume_matches_bathymetry(
    area: xr.DataArray,
    ocean_fraction: xr.DataArray,
    dz: xr.DataArray,
    deptho: xr.DataArray,
    rtol: float = 1e-2,
) -> list[Violation]:
    """The store's volume formula agrees with the volume under the bathymetry, sum(area x deptho)."""
    stored = ocean_volume(area, ocean_fraction, dz)
    bathymetric = float((area * deptho.fillna(0)).sum())
    error = relative_difference(stored, bathymetric)
    if error <= rtol:
        return []
    return [
        Violation(
            "geometry.volume_matches_bathymetry",
            f"sum(area x ocean_fraction x dz) = {stored:.6e} m3 but sum(area x deptho) = "
            f"{bathymetric:.6e} m3, {error:.2%} apart",
        )
    ]
