# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Metamorphic checks between two versions of one ocean: before and after a step, or two resolutions.

Conservative regridding and the gcm-filters smoothers are weighted averages with
non-negative weights that preserve area integrals. So, between a field and its
regridded or filtered version:

* linear integrals (area, volume, heat, salt) agree,
* variance doesn't grow (law of total variance),
* no new extremes appear,
* zonal power spectra agree at wavelengths both grids resolve.

The same relations hold between the published 1, 1/2 and 1/4 degree stores,
which share one source.
"""

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from samudra.invariants.core import (
    SEAWATER_DENSITY,
    SEAWATER_HEAT_CAPACITY,
    Violation,
    weighted_mean,
    weighted_sum,
)


def area_weights(area: xr.DataArray, ocean_fraction: xr.DataArray) -> xr.DataArray:
    """Ocean area of each cell: full cell area times the fraction that is ocean."""
    return area * ocean_fraction


def volume_weights(
    area: xr.DataArray, ocean_fraction: xr.DataArray, dz: xr.DataArray
) -> xr.DataArray:
    """Ocean volume of each cell, by the store convention area x ocean_fraction x dz."""
    return area * ocean_fraction * dz


def heat_content(thetao: xr.DataArray, volume: xr.DataArray, dims) -> xr.DataArray:
    """Heat content relative to 0 C, in J: rho_0 c_p sum(theta dV)."""
    return (
        SEAWATER_DENSITY * SEAWATER_HEAT_CAPACITY * weighted_sum(thetao, volume, dims)
    )


def salt_content(so: xr.DataArray, volume: xr.DataArray, dims) -> xr.DataArray:
    """Salt mass in kg (practical salinity as g/kg): rho_0 sum(S dV) / 1000."""
    return SEAWATER_DENSITY * weighted_sum(so, volume, dims) / 1000


def integral_preserved(
    fine: xr.DataArray,
    fine_weights: xr.DataArray,
    coarse: xr.DataArray,
    coarse_weights: xr.DataArray,
    name: str,
    dims=("y", "x"),
    rtol: float = 1e-3,
    atol: float = 0.0,
) -> list[Violation]:
    """``sum(fine * w_fine) == sum(coarse * w_coarse)`` over ``dims``, for each remaining index.

    ``rtol`` is relative to ``sum(|fine| * w_fine)``, not to the total, so a field
    whose positive and negative parts cancel (a velocity) isn't judged on round-off.
    """
    a = weighted_sum(fine, fine_weights, list(dims))
    b = weighted_sum(coarse, coarse_weights, list(dims))
    scale = weighted_sum(abs(fine), fine_weights, list(dims))
    return _compare(a, b, scale, f"conservation.{name}_integral", rtol, atol)


def mean_preserved(
    fine: xr.DataArray,
    fine_weights: xr.DataArray,
    coarse: xr.DataArray,
    coarse_weights: xr.DataArray,
    name: str,
    dims=("y", "x"),
    rtol: float = 1e-3,
    atol: float = 0.0,
) -> list[Violation]:
    """Weighted means agree over ``dims``, for each remaining index.

    ``rtol`` is relative to the weighted mean of ``|fine|``, as in `integral_preserved`.
    """
    a = weighted_mean(fine, fine_weights, list(dims))
    b = weighted_mean(coarse, coarse_weights, list(dims))
    scale = weighted_mean(abs(fine), fine_weights, list(dims))
    return _compare(a, b, scale, f"conservation.{name}_mean", rtol, atol)


def _compare(
    a: xr.DataArray,
    b: xr.DataArray,
    scale: xr.DataArray,
    invariant: str,
    rtol: float,
    atol: float,
) -> list[Violation]:
    a_values, b_values = np.ravel(np.asarray(a)), np.ravel(np.asarray(b))
    scale_values = np.ravel(np.asarray(scale))
    bad = ~(np.abs(a_values - b_values) <= rtol * scale_values + atol)
    if not bad.any():
        return []
    i = int(np.argmax(bad))
    return [
        Violation(
            invariant,
            f"{int(bad.sum())} of {bad.size} totals differ; e.g. {a_values[i]:.9g} vs "
            f"{b_values[i]:.9g} (difference {abs(a_values[i] - b_values[i]) / scale_values[i]:.3e} of the "
            "integral of |field|)",
            {"index": i},
        )
    ]


def variance_not_increased(
    fine: xr.DataArray,
    fine_weights: xr.DataArray,
    coarse: xr.DataArray,
    coarse_weights: xr.DataArray,
    name: str,
    dims=("y", "x"),
    rtol: float = 1e-6,
) -> list[Violation]:
    """Averaging can't create variance: weighted variance after <= before (law of total variance)."""

    def variance(field, weights):
        mean = weighted_mean(field, weights, list(dims))
        return weighted_mean((field - mean) ** 2, weights, list(dims)), mean

    before, mean = (np.ravel(np.asarray(v)) for v in variance(fine, fine_weights))
    after = np.ravel(np.asarray(variance(coarse, coarse_weights)[0]))
    # Round-off in a constant field's variance scales with the squared mean.
    grew = after > before * (1 + rtol) + 1e-12 * (mean**2 + 1e-30)
    if not grew.any():
        return []
    i = int(np.argmax(grew))
    return [
        Violation(
            f"conservation.{name}_variance_not_increased",
            f"variance grew from {before[i]:.6g} to {after[i]:.6g}",
            {"index": i},
        )
    ]


def no_new_extremes(
    fine: xr.DataArray, coarse: xr.DataArray, name: str, atol: float = 0.0
) -> list[Violation]:
    """The coarse range lies inside the fine range: weighted averages only move extremes inward."""
    fine_min, fine_max = float(fine.min()), float(fine.max())
    coarse_min, coarse_max = float(coarse.min()), float(coarse.max())
    if coarse_min >= fine_min - atol and coarse_max <= fine_max + atol:
        return []
    return [
        Violation(
            f"conservation.{name}_no_new_extremes",
            f"range widened from {fine_min:.6g}..{fine_max:.6g} to {coarse_min:.6g}..{coarse_max:.6g}",
        )
    ]


def band_zonal_spectrum(
    field: xr.DataArray,
    lat_b: ArrayLike,
    band: tuple[float, float],
    x_dim="x",
    y_dim="y",
) -> np.ndarray:
    """Zonal power spectrum of the area-weighted mean over a latitude band.

    Each row counts with the area it shares with ``band``, from its edges
    ``lat_b`` (``ny + 1`` values, or a ``(y_b, x_b)`` array of a rectilinear
    grid). Grids whose rows don't nest therefore average exactly the same band.
    Every row that overlaps the band must be fully wet. Returns power at zonal
    wavenumbers 0..nx/2 (cycles per circle).

    A cell holds its whole row's mean, so a row only partly inside the band adds
    an error that grows with the field's meridional gradient. When comparing two
    grids, put the band on the coarse grid's edges, so only the finer grid's
    thinner rows straddle it.
    """
    edges = np.asarray(lat_b, dtype=float)
    if edges.ndim == 2:
        edges = edges[:, 0]
    if edges.size != field.sizes[y_dim] + 1:
        raise ValueError(
            f"{edges.size} latitude edges for {field.sizes[y_dim]} rows; need one more"
        )
    lower = np.maximum(np.minimum(edges[:-1], edges[1:]), band[0])
    upper = np.minimum(np.maximum(edges[:-1], edges[1:]), band[1])
    overlap = np.clip(np.sin(np.deg2rad(upper)) - np.sin(np.deg2rad(lower)), 0, None)
    rows = np.flatnonzero(overlap > 0)
    if rows.size == 0:
        raise ValueError(f"no rows overlap latitude band {band}")
    selected = field.isel({y_dim: rows})
    if bool(selected.isnull().any()):
        raise ValueError(f"latitude band {band} has land; pick fully wet rows")
    weights = xr.DataArray(overlap[rows], dims=[y_dim])
    profile = (selected * weights).sum(y_dim) / weights.sum()
    nx = profile.sizes[x_dim]
    return np.abs(np.fft.rfft(np.asarray(profile), axis=-1)) ** 2 / nx**2


def zonal_spectra_agree(
    fine: xr.DataArray,
    fine_lat_b: ArrayLike,
    coarse: xr.DataArray,
    coarse_lat_b: ArrayLike,
    band: tuple[float, float],
    name: str,
    resolved_fraction: float = 0.125,
    rtol: float = 0.05,
) -> list[Violation]:
    """Zonal spectra of a fully wet latitude band agree at wavenumbers both grids resolve.

    Block-averaging n fine columns into one multiplies wavenumber k's amplitude by
    sinc(k / N) for N coarse columns, so at k <= ``resolved_fraction`` N the
    coarse power must lie between that attenuation and the fine power, within
    ``rtol``. ``*_lat_b`` are each grid's row edges (see `band_zonal_spectrum`).
    """
    fine_power = band_zonal_spectrum(fine, fine_lat_b, band)
    coarse_power = band_zonal_spectrum(coarse, coarse_lat_b, band)
    n_coarse = 2 * (coarse_power.size - 1)
    k_max = max(1, int(resolved_fraction * n_coarse))
    k = np.arange(1, k_max + 1)
    attenuation = np.sinc(k / n_coarse) ** 2
    lower = fine_power[k] * attenuation * (1 - rtol)
    upper = fine_power[k] * (1 + rtol)
    floor = 1e-12 * max(float(fine_power[1:].max(initial=0.0)), 1e-30)
    bad = ((coarse_power[k] < lower) | (coarse_power[k] > upper)) & (
        fine_power[k] > floor
    )
    if not bad.any():
        return []
    i = int(np.argmax(bad))
    return [
        Violation(
            f"conservation.{name}_zonal_spectrum",
            f"{int(bad.sum())} of {k_max} resolved wavenumbers disagree; at k={k[i]}: fine "
            f"{fine_power[k[i]]:.4g}, coarse {coarse_power[k[i]]:.4g}",
            {"wavenumber": int(k[i]), "band": band},
        )
    ]
