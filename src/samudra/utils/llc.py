# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import xarray as xr

from samudra.constants import DataLayout


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
    data_layout: DataLayout,
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
        expected_levels = len(data_layout.depth_i_levels)
        if n_levels != expected_levels:
            raise ValueError(
                f"Expected {expected_levels} levels for LLC variable {name}, got "
                f"{n_levels}"
            )

        for index, lev in enumerate(data_layout.depth_i_levels):
            data_copy[f"{name}_{lev}"] = data_copy[name].isel(lev=index)
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
    data_layout: DataLayout,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Standardize raw LLC inputs to the common non-compact loader layout.

    For example, a selected ``Theta(time, face, k, j, i)`` crop becomes
    level-indexed ``Theta_0(time, y, x)``, ``Theta_50(time, y, x)``, and the
    remaining configured levels; statistics names undergo the same renaming.
    """
    data_copy = data.copy()

    requested_data_vars = {
        _var_without_level(var_name)
        for var_name in (
            data_layout.prognostic_var_names + data_layout.boundary_var_names
        )
    }
    requested_data_vars.update(
        [data_layout.mask_all_levels_var, "mask_c", *data_layout.mask_vars]
    )
    requested_data_vars.update(_llc_staggered_mask_vars(data_copy, requested_data_vars))
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
    rename_map = {
        old: new
        for old, new in {
            "k": "lev",
            "mask_c": "wetmask",
            "i": "x",
            "j": "y",
        }.items()
        if old in data_copy.dims
        or old in data_copy.variables
        or old in data_copy.coords
    }
    if rename_map:
        data_copy = data_copy.rename(rename_map)

    means_copy = _rename_llc_level_index_vars(means.copy())
    stds_copy = _rename_llc_level_index_vars(stds.copy())
    data_copy = _flatten_llc_level_vars(data_copy, data_layout=data_layout)

    return data_copy, means_copy, stds_copy
