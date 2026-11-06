# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared ESMF conservative weights and explicit observational finite coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from xesmf import Regridder


def conservative_regridder(source: xr.Dataset, target: xr.Dataset) -> Regridder:
    """Build the same float64 ESMF weights for model and observation preparation.

    Source masks are deliberately not passed into ESMF: conservative weights use
    the full destination cell area. Callers choose how missing data are treated.
    """
    import xesmf as xe  # type: ignore[import-untyped]

    coordinates = ("lon", "lat", "lon_b", "lat_b")
    source_grid = xr.Dataset(
        coords={name: source[name].astype(np.float64) for name in coordinates}
    )
    target_grid = xr.Dataset(
        coords={name: target[name].astype(np.float64) for name in coordinates}
    )
    return xe.Regridder(
        source_grid,
        target_grid,
        "conservative",
        ignore_degenerate=True,
        periodic=True,
        unmapped_to_nan=True,
    )


def rectilinear_grid(
    lat: ArrayLike, lon: ArrayLike, lat_b: ArrayLike, lon_b: ArrayLike
) -> xr.Dataset:
    """Build xESMF coordinates from explicit centers and cell boundaries.

    Centers are one-dimensional latitude/longitude axes. Observation source grids
    can supply separate edge axes of lengths ``ny + 1`` and ``nx + 1``. OM4 grid
    export preserves the model pipeline's two-dimensional ``(ny + 1, nx + 1)``
    corner arrays; we broadcast the centers to match that representation for
    xESMF. Both formats retain the caller's supplied cell boundaries.
    """
    lat, lon = np.asarray(lat), np.asarray(lon)
    lat_b, lon_b = np.asarray(lat_b), np.asarray(lon_b)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(
            "Observation preparation requires one-dimensional latitude/longitude axes"
        )
    if lat_b.ndim == lon_b.ndim == 1:
        if lat_b.size != lat.size + 1 or lon_b.size != lon.size + 1:
            raise ValueError("Cell bounds must have one more edge than centers")
        return xr.Dataset(
            coords={
                "lat": ("y", lat),
                "lon": ("x", lon),
                "lat_b": ("y_b", lat_b),
                "lon_b": ("x_b", lon_b),
            }
        )
    shape = (lat.size + 1, lon.size + 1)
    if lat_b.shape != shape or lon_b.shape != shape:
        raise ValueError("Two-dimensional cell bounds must describe every cell corner")
    return xr.Dataset(
        coords={
            "lat": (("y", "x"), np.broadcast_to(lat[:, None], (lat.size, lon.size))),
            "lon": (("y", "x"), np.broadcast_to(lon[None, :], (lat.size, lon.size))),
            "lat_b": (("y_b", "x_b"), lat_b),
            "lon_b": (("y_b", "x_b"), lon_b),
        }
    )


class FiniteCoverageRemap:
    """ESMF mean over finite source area, gated by full-cell coverage.

    Regrid finite-value numerators and a 0/1 validity field with the same weights.
    Their ratio is the supported-area mean. The denominator remains the fraction
    of the FULL destination cell, so a tiny coastal sliver cannot become 100%
    coverage through missing-value renormalization.
    """

    def __init__(
        self,
        source: xr.Dataset,
        target: xr.Dataset,
    ) -> None:
        # Bounds are supplied by the source product and model-grid metadata.
        # A center coordinate does not uniquely define its cell boundaries.
        for grid in (source, target):
            for name in ("lat", "lon", "lat_b", "lon_b"):
                if name not in grid:
                    raise ValueError(f"Grid requires explicit {name} coordinates")
                if not np.isfinite(grid[name]).all():
                    raise ValueError(f"Grid coordinate {name} must be finite")
        self.source_shape = (
            tuple(source["lat"].shape)
            if source.lat.ndim == 2
            else (source.lat.size, source.lon.size)
        )
        self.shape = (
            tuple(target["lat"].shape)
            if target.lat.ndim == 2
            else (target.lat.size, target.lon.size)
        )
        self.regridder = conservative_regridder(source, target)

    def __call__(
        self, values: ArrayLike, min_coverage: float = 0.9
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        values = np.asarray(values)
        if values.shape[-2:] != self.source_shape:
            raise ValueError("Source shape does not match coordinates")
        if not 0 <= min_coverage <= 1:
            raise ValueError("Coverage threshold must be between zero and one")
        valid = np.isfinite(values)
        numerator = np.asarray(
            self.regridder(np.where(valid, values, 0).astype(np.float64))
        )
        fraction = np.asarray(self.regridder(valid.astype(np.float64)))
        fraction = np.nan_to_num(fraction, nan=0.0)
        mean = np.divide(
            numerator, fraction, out=np.full(fraction.shape, np.nan), where=fraction > 0
        )
        mean[fraction < min_coverage - 1e-10] = np.nan
        return mean.astype(np.float32), fraction.astype(np.float32)
