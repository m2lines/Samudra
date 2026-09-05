# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Ocean-basin masks on a model's native horizontal grid.

The published basin masks (``s3://m2lines-pubs/Samudra/basins/``) are all on
regular lat-lon grids, so none of them can be applied to OM4's native 1080x1440
tripolar grid: the shapes do not match, and where they happen to match on some
other curvilinear grid, matching shapes would not imply matching geography.

OM4's ``ocean_static`` already carries a ``basin`` field of integer region codes
on the native grid, alongside the real 2-D cell centers and the wet mask. That
is enough to build native masks directly, with no regridding and no polygon
work: the codes partition the grid by construction, so a cell cannot land in two
basins.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Region codes on OM4's `ocean_static.basin`, from its own CF flag attributes:
#   flag_values:   0 1 2 3 4 5 6 7 8 9 10
#   flag_meanings: global_land southern_ocean atlantic_ocean pacific_ocean
#                  arctic_ocean indian_ocean mediterranean_sea black_sea
#                  hudson_bay baltic_sea red_sea
OM4_BASIN_CODES: dict[str, int] = {
    "basin_southern": 1,
    "basin_atlantic": 2,
    "basin_pacific": 3,
    "basin_arctic": 4,
    "basin_indian": 5,
}

# The marginal seas (codes 6-10). The published Gaussian masks leave every one
# of them unassigned -- checked against `basin_masks_original.zarr`, where the
# Mediterranean, Black Sea, Red Sea, Hudson Bay and Baltic are zero in all five
# basins -- so we drop them too rather than inventing an attribution the rest of
# the basin diagnostics have never made.
OM4_MARGINAL_SEA_CODES: tuple[int, ...] = (6, 7, 8, 9, 10)

# `basin_id` as the published masks number them. This is *not* OM4's code: the
# published stores use their own ordering, and we keep it so the two sets of
# masks describe themselves the same way.
PUBLISHED_BASIN_IDS: dict[str, int] = {
    "basin_arctic": 1,
    "basin_atlantic": 2,
    "basin_indian": 3,
    "basin_pacific": 4,
    "basin_southern": 5,
}

_LONG_NAMES: dict[str, str] = {
    "basin_arctic": "Arctic",
    "basin_atlantic": "Atlantic",
    "basin_indian": "Indian",
    "basin_pacific": "Pacific",
    "basin_southern": "Southern",
}


def basin_masks_from_static(
    static: xr.Dataset,
    *,
    basin_var: str = "basin",
    wet_var: str = "wet",
    lat_var: str = "geolat",
    lon_var: str = "geolon",
) -> xr.Dataset:
    """Build native-grid basin masks from an OM4 ``ocean_static`` dataset.

    Args:
        static: OM4 ``ocean_static``, carrying integer region codes, a wet mask
            and the real 2-D cell centers on the native horizontal grid.
        basin_var: Name of the integer region-code field.
        wet_var: Name of the wet (ocean fraction) mask.
        lat_var: Name of the 2-D cell-center latitude.
        lon_var: Name of the 2-D cell-center longitude.

    Returns:
        A dataset of five boolean masks stored as int8, on horizontal dims named
        ``lat``/``lon`` (the index axes, matching what ``samudra viz`` renames
        its data to), carrying the true cell centers as ``lat_2d``/``lon_2d`` so
        the masks can be matched to data by geography rather than by position.
    """
    for name in (basin_var, wet_var, lat_var, lon_var):
        if name not in static.variables:
            raise ValueError(
                f"ocean_static carries no {name!r}. Native basin masks need the "
                f"region codes ({basin_var!r}), the wet mask ({wet_var!r}) and "
                f"the real 2-D cell centers ({lat_var!r}/{lon_var!r})."
            )

    basin = static[basin_var]
    if basin.ndim != 2:
        raise ValueError(
            f"{basin_var!r} must be 2-D on the horizontal grid, got dims {basin.dims}."
        )

    dims = basin.dims
    codes = np.asarray(basin.values)
    wet = np.asarray(static[wet_var].values) > 0.5
    lat2d = np.asarray(static[lat_var].values, dtype=np.float64)
    lon2d = np.asarray(static[lon_var].values, dtype=np.float64)

    for name, array in ((wet_var, wet), (lat_var, lat2d), (lon_var, lon2d)):
        if array.shape != codes.shape:
            raise ValueError(
                f"{name!r} has shape {array.shape} but {basin_var!r} is "
                f"{codes.shape}; they must be on the same horizontal grid."
            )

    ny, nx = codes.shape
    data_vars: dict[str, xr.DataArray] = {}
    for mask_name, code in OM4_BASIN_CODES.items():
        mask = (codes == code) & wet
        long_name = _LONG_NAMES[mask_name]
        data_vars[mask_name] = xr.DataArray(
            mask.astype(np.int8),
            dims=("lat", "lon"),
            attrs={
                "basin_id": PUBLISHED_BASIN_IDS[mask_name],
                "description": (
                    f"Boolean mask for {long_name} basin "
                    "(1=in basin, 0=not in basin or land)"
                ),
                "dtype": "bool",
                "long_name": f"{long_name} basin mask",
                "note": (
                    "Zero values represent land/non-basin areas and remain unassigned"
                ),
                "source": (
                    f"ocean_static '{basin_var}' region code {code} on the native "
                    "grid, intersected with the wet mask"
                ),
                "om4_basin_code": code,
            },
        )

    masks = xr.Dataset(
        data_vars,
        coords={
            # Index axes, carried over from the source so the store is
            # self-describing; viz matches on the 2-D coords below.
            "lat": ("lat", _axis(static, dims[0], ny)),
            "lon": ("lon", _axis(static, dims[1], nx)),
            "lat_2d": (("lat", "lon"), lat2d),
            "lon_2d": (("lat", "lon"), lon2d),
        },
        attrs={
            "description": (
                "Ocean-basin masks on the model's native horizontal grid, built "
                "from OM4 ocean_static region codes."
            ),
            "marginal_seas": (
                "Codes "
                + ", ".join(str(c) for c in OM4_MARGINAL_SEA_CODES)
                + " (Mediterranean, Black Sea, Hudson Bay, Baltic, Red Sea) are "
                "left unassigned, matching the published Gaussian masks."
            ),
        },
    )

    validate_basin_masks(masks, wet=wet)
    return masks


def _axis(static: xr.Dataset, dim: str, size: int) -> np.ndarray:
    """The source's index axis for `dim`, or a plain range when it has none."""
    if dim in static.coords:
        return np.asarray(static.coords[dim].values)
    return np.arange(size)


def validate_basin_masks(masks: xr.Dataset, *, wet: np.ndarray | None = None) -> None:
    """Check the masks partition the ocean, as the basin diagnostics assume.

    Raises:
        ValueError: If any cell is claimed by more than one basin, or if a mask
            claims a cell the wet mask calls land.
    """
    names = list(OM4_BASIN_CODES)
    stacked = np.stack([np.asarray(masks[name].values) for name in names])
    overlap = stacked.sum(axis=0) > 1
    if overlap.any():
        raise ValueError(
            f"{int(overlap.sum())} cells are assigned to more than one basin; "
            "the masks must partition the ocean."
        )

    if wet is not None:
        on_land = (stacked.sum(axis=0) > 0) & ~wet
        if on_land.any():
            raise ValueError(
                f"{int(on_land.sum())} cells are assigned to a basin but are "
                "land in the wet mask."
            )
