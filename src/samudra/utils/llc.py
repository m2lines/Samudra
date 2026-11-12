# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Adapt coarsened LLC stores on the Samudra Gaussian grid."""

import numpy as np
import xarray as xr

from samudra.constants import DataLayout, build_llc_layout
from samudra.utils.data import expand_compact_levels

_DEPTH_VARIABLES = {"U", "V", "Theta", "Salt"}


def _llc_statistics(
    dataset: xr.Dataset, variables: list[str], suffix: str, num_levels: int
) -> xr.Dataset:
    """Select embedded statistics and expose one scalar per canonical channel."""
    statistics = xr.Dataset()
    for name in variables:
        statistic = dataset[f"{name}_{suffix}"].load()
        expected_dims = ("k",) if name in _DEPTH_VARIABLES else ("one",)
        expected_size = num_levels if name in _DEPTH_VARIABLES else 1
        if statistic.dims != expected_dims or statistic.size != expected_size:
            raise ValueError(
                f"LLC {name}_{suffix} must have dimensions {expected_dims} "
                f"and size {expected_size}; got {statistic.dims}, {statistic.shape}"
            )
        values = statistic.to_numpy()
        if not np.isfinite(values).all() or (suffix == "std" and (values <= 0).any()):
            raise ValueError(
                f"LLC {name}_{suffix} must be finite"
                + (" and strictly positive" if suffix == "std" else "")
            )
        statistics[name] = (
            statistic.rename({"k": "lev"})
            if name in _DEPTH_VARIABLES
            else statistic.isel(one=0, drop=True)
        )
    return expand_compact_levels(statistics)


def canonicalize_llc_datasets(
    data: xr.Dataset,
    means: xr.Dataset,
    stds: xr.Dataset,
    *,
    prognostic_vars_key: str,
    boundary_vars_key: str,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, DataLayout]:
    """Read compact, already centered east/north fields without modifying storage.

    The producer stores fields on (time, k, y, x), geographic axes in lat(y)
    and lon(x), masks as wet_area(k, y, x), and embedded *_mean/*_std arrays.
    Native LLC faces, staggered grids, and cropped tile caches are unsupported.
    """
    unsupported = {"face", "i", "j", "i_g", "j_g"}.intersection(data.dims)
    if unsupported:
        raise ValueError(
            "LLC requires a coarsened Gaussian-grid store; "
            f"native face/tile dimensions are unsupported: {sorted(unsupported)}"
        )
    layout = build_llc_layout(prognostic_vars_key, boundary_vars_key)
    num_levels = len(layout.depth_levels)
    if data.sizes.get("k") != num_levels:
        raise ValueError(f"LLC requires {num_levels} depth levels on dimension 'k'")
    if "time" not in data.coords or not np.issubdtype(data.time.dtype, np.datetime64):
        raise ValueError("LLC requires decoded Gregorian datetime64 timestamps")
    for coordinate, dimension in (("lat", "y"), ("lon", "x")):
        if coordinate not in data or data[coordinate].dims != (dimension,):
            raise ValueError(f"LLC requires {coordinate}({dimension}) coordinates")
        values = data[coordinate].to_numpy()
        if not np.isfinite(values).all() or not (np.diff(values) > 0).all():
            raise ValueError(f"LLC {coordinate} must be finite and strictly increasing")

    channels = layout.prognostic_var_names + layout.boundary_var_names
    variables = list(
        dict.fromkeys(
            name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1].isdigit() else name
            for name in channels
        )
    )
    for name in variables:
        expected_dims = (
            ("time", "k", "y", "x") if name in _DEPTH_VARIABLES else ("time", "y", "x")
        )
        if data[name].dims != expected_dims:
            raise ValueError(
                f"LLC {name} must have dimensions {expected_dims}; got {data[name].dims}"
            )
    if data["wet_area"].dims != ("k", "y", "x"):
        raise ValueError("LLC wet_area must have dimensions (k, y, x)")
    wet_area = data["wet_area"].load().to_numpy()
    if not np.isfinite(wet_area).all() or (wet_area < 0).any():
        raise ValueError("LLC wet_area must be finite and non-negative")
    if data["rA"].dims != ("y", "x"):
        raise ValueError("LLC rA must have dimensions (y, x)")

    # Promote geographic variables before selecting fields: otherwise xarray
    # would discard them and expose positional y/x indices as latitude/longitude.
    canonical = data.set_coords(["lat", "lon", "rA"]).swap_dims(
        {"y": "lat", "x": "lon"}
    )
    canonical = canonical[variables + ["wet_area"]].rename(
        {"k": "lev", "rA": "areacello"}
    )
    canonical["mask"] = canonical["wet_area"] > 0
    canonical = expand_compact_levels(canonical.drop_vars("wet_area"))
    return (
        canonical,
        _llc_statistics(means, variables, "mean", num_levels),
        _llc_statistics(stds, variables, "std", num_levels),
        layout,
    )
