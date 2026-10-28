# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared ESMF conservative weights and explicit observational finite coverage."""

from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray


def conservative_regridder(source: xr.Dataset, target: xr.Dataset) -> Any:
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


def global_cell_bounds(
    centers: ArrayLike, *, latitude: bool = False
) -> NDArray[np.float64]:
    """Midpoint cell edges in degrees, including polar caps and the cyclic seam."""
    centers = np.asarray(centers, dtype=np.float64)
    if (
        centers.ndim != 1
        or len(centers) < 2
        or not np.isfinite(centers).all()
        or not np.all(np.diff(centers) > 0)
    ):
        raise ValueError("Coordinates must be finite and strictly increasing")
    edges = np.r_[
        centers[0] - (centers[1] - centers[0]) / 2,
        (centers[:-1] + centers[1:]) / 2,
        centers[-1] + (centers[-1] - centers[-2]) / 2,
    ]
    if latitude:
        if centers[0] < -90 or centers[-1] > 90:
            raise ValueError("Latitude is outside the sphere")
        edges = np.clip(edges, -90, 90)
        edges[0], edges[-1] = -90, 90
    elif not np.isclose(edges[-1] - edges[0], 360, rtol=0, atol=1e-6):
        raise ValueError("Expected a complete global longitude axis")
    return edges


def rectilinear_global_grid(lat: ArrayLike, lon: ArrayLike) -> xr.Dataset:
    return xr.Dataset(
        coords={
            "lat": ("lat", np.asarray(lat, dtype=np.float64)),
            "lon": ("lon", np.asarray(lon, dtype=np.float64)),
            "lat_b": ("lat_b", global_cell_bounds(lat, latitude=True)),
            "lon_b": ("lon_b", global_cell_bounds(lon)),
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
        lat: ArrayLike,
        lon: ArrayLike,
        target_lat: ArrayLike,
        target_lon: ArrayLike,
    ) -> None:
        self.source_shape = (len(np.asarray(lat)), len(np.asarray(lon)))
        self.shape = (len(np.asarray(target_lat)), len(np.asarray(target_lon)))
        self.regridder = conservative_regridder(
            rectilinear_global_grid(lat, lon),
            rectilinear_global_grid(target_lat, target_lon),
        )

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
