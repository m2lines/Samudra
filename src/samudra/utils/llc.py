# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import xarray as xr

from samudra.constants import DataLayout, build_llc_layout

_LLC_CENTER_LON_CANDIDATES = ("XC", "longitude", "lon")
_LLC_CENTER_LAT_CANDIDATES = ("YC", "latitude", "lat")
_LLC_CELL_AREA_CANDIDATES = ("rA", "areacello")
_LLC_GRID_METADATA_CANDIDATES = (
    *_LLC_CENTER_LON_CANDIDATES,
    *_LLC_CENTER_LAT_CANDIDATES,
    *_LLC_CELL_AREA_CANDIDATES,
)


def _rename_llc_level_index_vars(ds: xr.Dataset) -> xr.Dataset:
    """Rename LLC statistics variables to the common level-index format.

    For example, ``Theta_lev_0`` becomes ``Theta_0``.
    """
    rename_map = {
        name: f"{var}_{lev}"
        for name in ds.variables
        if isinstance(name, str) and "_lev_" in name
        for var, lev in [name.split("_lev_", maxsplit=1)]
    }
    return ds.rename(rename_map) if rename_map else ds


def _flatten_llc_level_vars(
    data: xr.Dataset,
    *,
    num_levels: int,
) -> xr.Dataset:
    """Flatten LLC level dimensions into level-indexed variables.

    For example, ``Theta(time, lev, y, x)`` becomes ``Theta_0(time, y, x)``,
    ``Theta_50(time, y, x)``, and the remaining configured levels.
    """
    data_copy = data.copy()
    for name in list(data_copy.data_vars):
        if not isinstance(name, str):
            continue
        if "lev" not in data_copy[name].dims:
            continue

        n_levels = data_copy[name].sizes["lev"]
        if n_levels != num_levels:
            raise ValueError(
                f"Expected {num_levels} levels for LLC variable {name}, got {n_levels}"
            )

        for index in range(num_levels):
            data_copy[f"{name}_{index}"] = data_copy[name].isel(lev=index)
        data_copy = data_copy.drop_vars(name)

    return data_copy


def _var_without_level(var_name: str) -> str:
    """Return the raw LLC variable name for a level-indexed model variable.

    For example, ``Theta_50`` becomes ``Theta`` while ``oceQnet`` is unchanged.
    """
    suffix = var_name.rsplit("_", maxsplit=1)[-1]
    return var_name.rsplit("_", maxsplit=1)[0] if suffix.isdigit() else var_name


def _preferred_available_var(
    data: xr.Dataset, candidates: tuple[str, ...]
) -> str | None:
    """Select the first candidate present in an LLC dataset.

    For example, candidates ``("mask_w", "hFacW")`` select ``mask_w`` when
    both variables are available.
    """
    return next((name for name in candidates if name in data.data_vars), None)


def _available_data_vars(data: xr.Dataset, candidates: tuple[str, ...]) -> set[str]:
    """Return candidate names that are data variables in the dataset."""
    return {name for name in candidates if name in data.data_vars}


def _llc_grid_coord(
    data: xr.Dataset,
    candidates: tuple[str, ...],
    target_name: str,
    *,
    horizontal_dims: tuple[str, str],
) -> tuple[str, xr.DataArray]:
    """Return one required 2D LLC grid coordinate on horizontal dims."""
    lat_dim, lon_dim = horizontal_dims
    incompatible_dims = {}
    for source_name in candidates:
        if source_name not in data.variables:
            continue
        coord = data[source_name]
        if set(coord.dims) != {lat_dim, lon_dim}:
            incompatible_dims[source_name] = coord.dims
            continue
        return source_name, coord.transpose(lat_dim, lon_dim).rename(target_name)

    details = (
        f" Found incompatible candidate dims: {incompatible_dims}."
        if incompatible_dims
        else ""
    )
    raise ValueError(
        "LLC canonicalization requires real grid metadata. Missing one of "
        f"{candidates} for canonical coordinate {target_name!r}.{details}"
    )


def _assign_llc_grid_metadata(
    data: xr.Dataset, *, horizontal_dims: tuple[str, str]
) -> xr.Dataset:
    """Attach real LLC geographic coordinates and cell area as coordinates."""
    lon_source, lon = _llc_grid_coord(
        data, _LLC_CENTER_LON_CANDIDATES, "lon_2d", horizontal_dims=horizontal_dims
    )
    lat_source, lat = _llc_grid_coord(
        data, _LLC_CENTER_LAT_CANDIDATES, "lat_2d", horizontal_dims=horizontal_dims
    )
    area_source, area = _llc_grid_coord(
        data, _LLC_CELL_AREA_CANDIDATES, "areacello", horizontal_dims=horizontal_dims
    )
    coords = {
        "lon_2d": lon,
        "lat_2d": lat,
        "areacello": area,
    }
    out = data.assign_coords(coords)
    raw_metadata_vars = [
        source
        for source in _LLC_GRID_METADATA_CANDIDATES
        if source not in coords and source in out.variables
    ]
    return out.drop_vars(raw_metadata_vars)


def _llc_staggered_mask_vars(
    data: xr.Dataset,
    requested_data_vars: set[str],
) -> set[str]:
    """Find staggered masks needed by the requested LLC variables.

    For example, requesting ``U`` from data containing ``mask_w`` adds
    ``mask_w`` to the variables retained during canonicalization.
    """
    mask_vars = set()
    if {"U", "oceTAUX"} & requested_data_vars:
        if mask_var := _preferred_available_var(data, ("mask_w", "hFacW")):
            mask_vars.add(mask_var)
    if {"V", "oceTAUY"} & requested_data_vars:
        if mask_var := _preferred_available_var(data, ("mask_s", "hFacS")):
            mask_vars.add(mask_var)
    return mask_vars


def canonicalize_llc_datasets(
    data: xr.Dataset,
    means: xr.Dataset,
    stds: xr.Dataset,
    *,
    face: int,
    i_start: int,
    i_end: int,
    j_start: int,
    j_end: int,
    prognostic_vars_key: str,
    boundary_vars_key: str,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, DataLayout]:
    """Standardize raw LLC inputs to the common non-compact loader layout.

    For example, a selected ``Theta(time, face, k, j, i)`` crop becomes
    level-indexed ``Theta_0(time, lat, lon)``, ``Theta_50(time, lat, lon)``, and the
    remaining configured levels; statistics names undergo the same renaming.
    """
    data_layout = build_llc_layout(prognostic_vars_key, boundary_vars_key)
    data_copy = data.copy()

    requested_data_vars = {
        _var_without_level(var_name)
        for var_name in (
            data_layout.prognostic_var_names + data_layout.boundary_var_names
        )
    }
    center_mask_var = (
        None
        if "wetmask" in data_copy.data_vars
        else _preferred_available_var(data_copy, ("mask_c", "hFacC"))
    )
    requested_data_vars.add("wetmask")
    if center_mask_var is not None:
        requested_data_vars.add(center_mask_var)
    requested_data_vars.update(_llc_staggered_mask_vars(data_copy, requested_data_vars))
    requested_data_vars.update(
        _available_data_vars(data_copy, _LLC_CENTER_LON_CANDIDATES)
    )
    requested_data_vars.update(
        _available_data_vars(data_copy, _LLC_CENTER_LAT_CANDIDATES)
    )
    requested_data_vars.update(
        _available_data_vars(data_copy, _LLC_CELL_AREA_CANDIDATES)
    )
    data_copy = data_copy[
        [name for name in data_copy.data_vars if name in requested_data_vars]
    ]

    if "face" in data_copy.dims or "face" in data_copy.coords:
        data_copy = data_copy.sel(face=face, drop=True)

    spatial_indexers = {}
    if "i" in data_copy.dims:
        spatial_indexers["i"] = slice(i_start, i_end)
    if "i_g" in data_copy.dims:
        spatial_indexers["i_g"] = slice(i_start, i_end)
    if "j" in data_copy.dims:
        spatial_indexers["j"] = slice(j_start, j_end)
    if "j_g" in data_copy.dims:
        spatial_indexers["j_g"] = slice(j_start, j_end)
    if spatial_indexers:
        data_copy = data_copy.isel(spatial_indexers)

    unstagger_map = {
        "U": {"i_g": "i"},
        "V": {"j_g": "j"},
        "oceTAUX": {"i_g": "i"},
        "oceTAUY": {"j_g": "j"},
        "mask_w": {"i_g": "i"},
        "hFacW": {"i_g": "i"},
        "mask_s": {"j_g": "j"},
        "hFacS": {"j_g": "j"},
    }
    for var_name, rename_dims in unstagger_map.items():
        if var_name not in data_copy.variables:
            continue
        used_renames = {
            old: new
            for old, new in rename_dims.items()
            if old in data_copy[var_name].dims
        }
        if used_renames:
            data_copy[var_name] = data_copy[var_name].rename(used_renames)

    remaining_staggered = [
        name
        for name in data_copy.data_vars
        if "i_g" in data_copy[name].dims or "j_g" in data_copy[name].dims
    ]
    if remaining_staggered:
        raise ValueError(
            "LLC canonicalization left staggered variables unresolved: "
            f"{remaining_staggered}"
        )

    data_copy = data_copy.drop_vars(["i_g", "j_g"], errors="ignore")
    data_copy = _assign_llc_grid_metadata(data_copy, horizontal_dims=("j", "i"))
    rename_map = {
        old: new
        for old, new in {
            "k": "lev",
            "i": "lon",
            "j": "lat",
        }.items()
        if old in data_copy.dims
        or old in data_copy.variables
        or old in data_copy.coords
    }
    if center_mask_var is not None:
        rename_map[center_mask_var] = "wetmask"
    if rename_map:
        data_copy = data_copy.rename(rename_map)

    means_copy = _rename_llc_level_index_vars(means.copy())
    stds_copy = _rename_llc_level_index_vars(stds.copy())
    data_copy = _flatten_llc_level_vars(
        data_copy, num_levels=len(data_layout.depth_levels)
    )
    mask_rename_map = {
        f"wetmask_{level}": f"mask_{level}"
        for level in range(len(data_layout.depth_levels))
        if f"wetmask_{level}" in data_copy
    }
    if mask_rename_map:
        data_copy = data_copy.rename(mask_rename_map)

    return data_copy, means_copy, stds_copy, data_layout
