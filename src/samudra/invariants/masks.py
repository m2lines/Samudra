# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Masks agree with each other, with the fields, and with the grid.

A store describes where the ocean is several times over: the boolean masks, the
NaN pattern of every field, `ocean_fraction`, and the bathymetry. These checks
make every pair of descriptions agree.
"""

import re

import xarray as xr

from samudra.invariants.core import Violation, first_location, isfinite

LEVEL_SUFFIX = re.compile(r"^(?P<base>[a-z]+)_(?P<level>\d+)$")
STATE_3D = ("thetao", "so", "uo", "vo")
# Fields defined only on the surface ocean, so NaN exactly where mask_0 is land.
# Wind stress and heat flux are defined over the ocean only in OM4 output.
STATE_2D = ("zos", "hfds", "tauuo", "tauvo", "wfo")


def nest_with_depth(wet: xr.DataArray, depth_dim: str = "lev") -> list[Violation]:
    """A cell that is ocean at level k+1 is ocean at level k."""
    wet = wet.astype(bool)
    deeper = wet.isel({depth_dim: slice(1, None)})
    shallower = wet.isel({depth_dim: slice(None, -1)})
    shallower = shallower.assign_coords({depth_dim: deeper[depth_dim]})
    orphans = deeper & ~shallower
    count = int(orphans.sum())
    if not count:
        return []
    return [
        Violation(
            "masks.nest_with_depth",
            f"{count} cells are ocean below a land cell",
            first_location(orphans),
        )
    ]


def ocean_fraction_bounds(
    ocean_fraction: xr.DataArray, depth_dim: str = "lev", atol: float = 1e-6
) -> list[Violation]:
    """`ocean_fraction` is in [0, 1], finite, and never increases with depth."""
    violations = []
    nonfinite = ~isfinite(ocean_fraction)
    if nonfinite.any():
        violations.append(
            Violation(
                "masks.ocean_fraction_finite",
                f"{int(nonfinite.sum())} non-finite values",
                first_location(nonfinite),
            )
        )
    out_of_range = (ocean_fraction < -atol) | (ocean_fraction > 1 + atol)
    if out_of_range.any():
        violations.append(
            Violation(
                "masks.ocean_fraction_in_unit_interval",
                f"{int(out_of_range.sum())} values outside [0, 1]; range "
                f"{float(ocean_fraction.min()):.6g}..{float(ocean_fraction.max()):.6g}",
                first_location(out_of_range),
            )
        )
    if depth_dim in ocean_fraction.dims:
        increase = ocean_fraction.diff(depth_dim) > atol
        if increase.any():
            violations.append(
                Violation(
                    "masks.ocean_fraction_non_increasing_with_depth",
                    f"{int(increase.sum())} cells have more ocean than the level above; "
                    f"largest increase {float(ocean_fraction.diff(depth_dim).max()):.6g}",
                    first_location(increase),
                )
            )
    return violations


def nan_matches_mask(
    field: xr.DataArray, wet: xr.DataArray, name: str
) -> list[Violation]:
    """``field`` is finite exactly where ``wet`` is True, at every time step.

    ``wet`` may have fewer dimensions than ``field`` (e.g. no time); it broadcasts.
    """
    wet = wet.astype(bool)
    finite = isfinite(field)
    land_values = finite & ~wet
    ocean_holes = ~finite & wet
    violations = []
    if land_values.any():
        violations.append(
            Violation(
                "masks.nan_on_land",
                f"{name} has {int(land_values.sum())} finite values on land",
                first_location(land_values),
            )
        )
    if ocean_holes.any():
        violations.append(
            Violation(
                "masks.finite_in_ocean",
                f"{name} has {int(ocean_holes.sum())} NaN or infinite values in the ocean",
                first_location(ocean_holes),
            )
        )
    return violations


def mask_matches_fraction(
    wet: xr.DataArray, ocean_fraction: xr.DataArray, name: str, atol: float = 1e-9
) -> list[Violation]:
    """A cell is masked as ocean exactly when some of it is ocean (``ocean_fraction > atol``).

    Conservative regridding leaves round-off fractions (~1e-16) where a land
    cell's great-circle edges graze a wet neighbour; ``atol`` treats those as land.
    """
    partly_ocean = ocean_fraction > atol
    disagree = wet.astype(bool) != partly_ocean
    if not disagree.any():
        return []
    return [
        Violation(
            "masks.mask_matches_ocean_fraction",
            f"{name} disagrees with ocean_fraction > 0 in {int(disagree.sum())} cells",
            first_location(disagree),
        )
    ]


def flattened_levels(ds: xr.Dataset, base: str) -> list[int]:
    """Level indices present for a flattened variable family, e.g. ``thetao_0``..``thetao_18``."""
    levels = []
    for name in ds.data_vars:
        match = LEVEL_SUFFIX.match(str(name))
        if match and match["base"] == base:
            levels.append(int(match["level"]))
    return sorted(levels)


def flattened_store(
    ds: xr.Dataset, time_index: int | None = 0, fraction_atol: float = 1e-9
) -> list[Violation]:
    """Every mask description in a flattened store agrees with every other.

    Checks, per level k: ``mask_k`` against the NaN pattern of each ``{var}_k``,
    against ``ocean_fraction[k] > 0``, and nesting of ``mask_k`` with depth. Also
    ``mask_0`` against the surface fields, and that ``areacello`` (the full cell
    area, land included) is finite and positive everywhere.

    Args:
        ds: A flattened store, such as a published OM4 release.
        time_index: The time step whose fields are compared with the masks, or
            None to compare every step (expensive for large stores).
        fraction_atol: Fractions at or below this count as land when compared
            with the masks.
    """
    levels = flattened_levels(ds, "mask")
    if not levels:
        return [Violation("masks.present", "no mask_k variables found")]
    sample = (
        ds
        if time_index is None or "time" not in ds.dims
        else ds.isel(time=[time_index])
    )
    masks_k = [f"mask_{k}" for k in levels]
    fields_3d = {
        k: [f"{base}_{k}" for base in STATE_3D if f"{base}_{k}" in ds] for k in levels
    }
    fields_2d = [name for name in STATE_2D if name in ds]
    extras = [name for name in ("ocean_fraction", "areacello") if name in ds]
    # One compute for everything, so a remote store's chunks download in parallel
    # instead of one request at a time.
    names = masks_k + [n for k in levels for n in fields_3d[k]] + fields_2d + extras
    loaded = sample[names].load()
    violations: list[Violation] = []

    wet = xr.concat([loaded[name] for name in masks_k], dim="lev")
    violations += nest_with_depth(wet)
    if "ocean_fraction" in loaded:
        fraction = loaded["ocean_fraction"]
        violations += ocean_fraction_bounds(fraction)
        for k in levels:
            violations += mask_matches_fraction(
                loaded[f"mask_{k}"],
                fraction.isel(lev=k),
                f"mask_{k}",
                atol=fraction_atol,
            )

    for k in levels:
        for name in fields_3d[k]:
            violations += nan_matches_mask(loaded[name], loaded[f"mask_{k}"], name)
    for name in fields_2d:
        violations += nan_matches_mask(loaded[name], loaded["mask_0"], name)

    if "areacello" in loaded.variables:
        area = loaded["areacello"]
        bad = ~isfinite(area) | (area <= 0)
        if bad.any():
            violations.append(
                Violation(
                    "masks.areacello_positive_everywhere",
                    f"{int(bad.sum())} cells have non-finite or non-positive area",
                    first_location(bad),
                )
            )
    return violations
