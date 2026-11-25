# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Physical bounds, aimed at the artifacts processing leaves behind.

A plain range check rarely fails on real model output. Processing artifacts
cluster at coasts: averaging a field with zero-filled land, or with a fill value,
drags coastal cells towards that value. `coastal_dilution` compares each coastal
cell with its wet neighbours, which catches that even when the result still lies
in the physical range.
"""

import xarray as xr

from samudra.invariants.core import Violation, first_location, isfinite

# Linearized TEOS-10 freezing point of seawater (IOC 2010), in degrees C:
# about -1.9 C at 35 psu at the surface, falling with salinity and pressure.
FREEZING_INTERCEPT = 0.0901
FREEZING_SALINITY_SLOPE = -0.0575  # C per psu
FREEZING_PRESSURE_SLOPE = -7.61e-4  # C per dbar, with 1 dbar ~ 1 m


def freezing_point(salinity, depth_m=0.0):
    """Freezing temperature of seawater in degrees C, linearized in salinity and depth."""
    return (
        FREEZING_INTERCEPT
        + FREEZING_SALINITY_SLOPE * salinity
        + FREEZING_PRESSURE_SLOPE * depth_m
    )


def above_freezing(
    thetao: xr.DataArray,
    so: xr.DataArray,
    depth_m: xr.DataArray | float = 0.0,
    tolerance: float = 0.1,
) -> list[Violation]:
    """Temperature is at or above the freezing point for its salinity and depth."""
    margin = thetao - freezing_point(so, depth_m)
    too_cold = margin < -tolerance
    if not too_cold.any():
        return []
    return [
        Violation(
            "bounds.above_freezing",
            f"{int(too_cold.sum())} cells are colder than freezing by more than "
            f"{tolerance} C; worst {float(margin.min()):.3f} C",
            first_location(too_cold),
        )
    ]


def within(
    field: xr.DataArray, name: str, lower: float, upper: float
) -> list[Violation]:
    """All finite values lie in ``[lower, upper]``."""
    outside = (field < lower) | (field > upper)
    if not outside.any():
        return []
    return [
        Violation(
            f"bounds.{name}_range",
            f"{int(outside.sum())} values outside [{lower}, {upper}]; range "
            f"{float(field.min()):.6g}..{float(field.max()):.6g}",
            first_location(outside),
        )
    ]


def finite_where_wet(
    field: xr.DataArray, wet: xr.DataArray, name: str
) -> list[Violation]:
    """No NaN or infinity in the ocean. Fill values such as 1e20 count as out of range elsewhere."""
    bad = wet.astype(bool) & ~isfinite(field)
    if not bad.any():
        return []
    return [
        Violation(
            f"bounds.{name}_finite",
            f"{int(bad.sum())} non-finite ocean values",
            first_location(bad),
        )
    ]


def wet_neighbours(
    field: xr.DataArray, wet: xr.DataArray, y_dim: str = "y", x_dim: str = "x"
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """For each cell, the min and max of its wet 4-neighbours, and whether it borders land.

    Longitude wraps; latitude doesn't.
    """
    wet = wet.astype(bool)
    values = field.where(wet)
    shifted = [
        values.roll({x_dim: 1}, roll_coords=False),
        values.roll({x_dim: -1}, roll_coords=False),
        values.shift({y_dim: 1}),
        values.shift({y_dim: -1}),
    ]
    wet_shifted = [
        wet.roll({x_dim: 1}, roll_coords=False),
        wet.roll({x_dim: -1}, roll_coords=False),
        wet.shift({y_dim: 1}, fill_value=True),
        wet.shift({y_dim: -1}, fill_value=True),
    ]
    stacked = xr.concat(shifted, dim="neighbour")
    borders_land = ~xr.concat(wet_shifted, dim="neighbour").all("neighbour")
    return stacked.min("neighbour"), stacked.max("neighbour"), borders_land & wet


def coastal_dilution(
    field: xr.DataArray,
    wet: xr.DataArray,
    name: str,
    toward: float = 0.0,
    max_fraction: float = 0.5,
    y_dim: str = "y",
    x_dim: str = "x",
) -> list[Violation]:
    """Coastal cells aren't pulled towards a land value beyond their wet neighbours' range.

    A coastal cell's value can lie outside its wet neighbours' range: estuaries
    are fresher than the sea next to them. But if averaging with land drags it
    towards ``toward`` (0 for zero-filled land), it ends up much closer to
    ``toward`` than any neighbour. This flags coastal cells whose distance from
    ``toward`` is below ``max_fraction`` times their nearest wet neighbour's.

    Use with fields that are far from ``toward`` in the open ocean, such as
    salinity (``toward=0``) or temperature in kelvin.
    """
    low, high, coastal = wet_neighbours(field, wet, y_dim, x_dim)
    distance = abs(field - toward)
    nearest_neighbour = xr.where(
        abs(low - toward) < abs(high - toward), abs(low - toward), abs(high - toward)
    )
    diluted = coastal & (distance < max_fraction * nearest_neighbour)
    if not diluted.any():
        return []
    return [
        Violation(
            f"bounds.{name}_coastal_dilution",
            f"{int(diluted.sum())} of {int(coastal.sum())} coastal cells are closer to "
            f"{toward} than {max_fraction} x their nearest wet neighbour",
            first_location(diluted),
        )
    ]
