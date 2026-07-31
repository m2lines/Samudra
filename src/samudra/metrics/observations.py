# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Loaders for observation products, and the model-side adapter.

Three observation products back the metrics:

- **DUACS** — surface geostrophic velocity from satellite altimetry, used for
  the velocity and EKE metrics.
- **OISST** — sea-surface temperature, used for the SST metrics.
- **ARGO-IAP** — gridded temperature profiles, used for ocean heat content.

Each ships on its own native grid with its own naming, so the loaders resolve
coordinate and variable aliases, put longitude on `[0, 360)`, and attach a
spherical cell area where the product does not carry one.

The model adapter is deliberately thin. Our `predictions.zarr` is already
written in the OM4 analysis layout with true 2-D `lat`/`lon`, `areacello`, `dz`
and a wet mask (see `samudra/utils/writer.py`), so there is no geometry to
reconstruct — only a presentation change from `(y, x)` dims to the `(lat, lon)`
dims the observation grids use.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import xarray as xr

from samudra.constants import DatasetSpec
from samudra.metrics import kernels
from samudra.utils.location import ResolvedLocation

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = kernels.EARTH_RADIUS_M

LAT_ALIASES = ["lat", "latitude", "y", "nav_lat"]
LON_ALIASES = ["lon", "longitude", "x", "nav_lon"]
TIME_ALIASES = ["time", "valid_time"]
# `depth_std` is what the IAP/ARGO product calls its standard-level depth axis.
DEPTH_ALIASES = ["depth", "depth_std", "lev", "level", "z", "DEPTH", "depth_0", "z_t"]

# DUACS ships both absolute geostrophic velocity (from absolute dynamic
# topography) and the anomaly (from sea-level anomaly). The model derives
# absolute velocity from `zos`, so `absolute` is the like-for-like comparison.
DUACS_VELOCITY_COMPONENTS = {
    "absolute": {
        "u": ["ugos", "surface_geostrophic_eastward_sea_water_velocity"],
        "v": ["vgos", "surface_geostrophic_northward_sea_water_velocity"],
        "label": "absolute geostrophic velocity",
    },
    "anomaly": {
        "u": ["ugosa"],
        "v": ["vgosa"],
        "label": "geostrophic velocity anomaly",
    },
}

SST_ALIASES = ["sst", "analysed_sst", "thetao", "temperature"]
ARGO_TEMPERATURE_ALIASES = ["thetao", "temp", "temperature", "T"]


def find_coord_name(
    obj: xr.Dataset | xr.DataArray, candidates: Sequence[str], required: bool = True
) -> str | None:
    """First of `candidates` present as a coordinate or dimension."""
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    if required:
        raise KeyError(
            f"Could not find a coordinate among {candidates}. "
            f"Available: {list(obj.coords)} / {list(obj.dims)}"
        )
    return None


def find_var_name(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    """First of `candidates` present as a data variable."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(
        f"Could not find a variable among {candidates}. Available: {list(ds.data_vars)}"
    )


def spherical_cell_area(lat_vals: np.ndarray, lon_vals: np.ndarray) -> np.ndarray:
    """Cell area on a sphere for a rectilinear lat/lon grid, in m^2."""
    lat = np.asarray(lat_vals, dtype=float)
    lon = np.asarray(lon_vals, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("Expected 1-D lat/lon coordinates")
    dlon = np.gradient(np.deg2rad(lon))
    dlat = np.gradient(np.deg2rad(lat))
    dx = EARTH_RADIUS_M * np.cos(np.deg2rad(lat))[:, None] * dlon[None, :]
    dy = EARTH_RADIUS_M * dlat[:, None] * np.ones((lat.size, lon.size))
    return np.abs(dx) * np.abs(dy)


def standardize(ds: xr.Dataset) -> xr.Dataset:
    """Rename to `lat`/`lon`/`time`, coerce time, and normalize longitude."""
    lat_name = find_coord_name(ds, LAT_ALIASES)
    lon_name = find_coord_name(ds, LON_ALIASES)
    time_name = find_coord_name(ds, TIME_ALIASES)

    ds = ds.assign_coords(
        {time_name: kernels.coerce_datetime_values(ds[time_name].values)}
    )
    ds = ds.rename({lat_name: "lat", lon_name: "lon", time_name: "time"})
    ds = kernels.normalize_lon(ds, "lon")
    ds = ds.sortby("lat").sortby("lon")
    return _drop_singleton_dims(ds)


def _drop_singleton_dims(ds: xr.Dataset) -> xr.Dataset:
    """Squeeze away degenerate non-physical dims some products carry."""
    keep = {"time", "lat", "lon", *DEPTH_ALIASES}
    squeeze = [dim for dim, size in ds.sizes.items() if size == 1 and dim not in keep]
    return ds.squeeze(squeeze, drop=True) if squeeze else ds


def with_cell_area(ds: xr.Dataset) -> xr.Dataset:
    """Attach a spherical `area` variable unless the product already has one."""
    if "area" in ds:
        return ds
    area = spherical_cell_area(ds["lat"].values, ds["lon"].values)
    return ds.assign(area=(["lat", "lon"], area))


def open_product(location: ResolvedLocation, name: str) -> xr.Dataset:
    """Open and standardize one observation product."""
    logger.info("Opening observation product %s from %s", name, location)
    ds = location.open(chunks={})
    ds = standardize(ds)
    ds = with_cell_area(ds)
    logger.info(
        "  %s: %s, %s to %s",
        name,
        dict(ds.sizes),
        np.datetime_as_string(ds["time"].values[0], unit="D"),
        np.datetime_as_string(ds["time"].values[-1], unit="D"),
    )
    return ds


def duacs_velocity(
    ds: xr.Dataset, kind: str = "absolute"
) -> tuple[xr.DataArray, xr.DataArray]:
    """The DUACS eastward/northward geostrophic velocity pair, equator masked."""
    if kind not in DUACS_VELOCITY_COMPONENTS:
        raise ValueError(
            f"Unknown DUACS velocity kind {kind!r}; "
            f"expected one of {sorted(DUACS_VELOCITY_COMPONENTS)}"
        )
    config = DUACS_VELOCITY_COMPONENTS[kind]
    u = ds[find_var_name(ds, config["u"])]
    v = ds[find_var_name(ds, config["v"])]
    return (
        kernels.apply_equatorial_mask(u, "lat"),
        kernels.apply_equatorial_mask(v, "lat"),
    )


def model_on_latlon_grid(ds: xr.Dataset, dataset_spec: DatasetSpec) -> xr.Dataset:
    """Present a rollout on `(lat, lon)` dims for comparison with observations.

    Our rollouts carry `(y, x)` dims alongside *2-D* `lat`/`lon` coordinates, so
    the coordinates have to be dropped before `y`/`x` can take those names —
    otherwise the rename collides. On a rectilinear ("gaussian") grid the `y`
    and `x` axes are exactly the latitude and longitude in degrees, so the swap
    is lossless.

    Curvilinear grids are refused rather than mishandled: there `y`/`x` are
    index space, not degrees, and comparing against observations needs a real
    regridding step (xesmf) that is not implemented here yet.
    """
    if dataset_spec.grid_type != "gaussian":
        raise NotImplementedError(
            f"Observation metrics need a rectilinear model grid, but this dataset "
            f"has grid_type={dataset_spec.grid_type!r}. On a curvilinear grid the "
            "'y'/'x' axes are index space rather than degrees, so comparison "
            "requires conservative regridding (xesmf) onto the observation grid. "
            "See issues #801 and #809."
        )

    dims = {str(dim) for dim in ds.dims}
    missing = {"y", "x"} - dims
    if missing:
        raise ValueError(
            f"Rollout is missing expected horizontal dimensions {sorted(missing)}; "
            f"found dims {sorted(dims)}"
        )

    # Preserve the real geometry before the 2-D coords are dropped: `areacello`
    # is the model's own cell area and is more accurate than anything we could
    # re-derive from the axes.
    areacello = (
        ds["areacello"] if "areacello" in ds.coords or "areacello" in ds else None
    )

    ds = ds.drop_vars(["lat", "lon", "lat_b", "lon_b"], errors="ignore")
    ds = ds.rename({"y": "lat", "x": "lon"})
    if areacello is not None:
        renamed_area = areacello.drop_vars(
            ["lat", "lon", "lat_b", "lon_b"], errors="ignore"
        ).rename({"y": "lat", "x": "lon"})
        ds = ds.assign_coords(areacello=renamed_area)

    ds = kernels.normalize_lon(ds, "lon").sortby("lat").sortby("lon")

    # OM4 (and therefore every rollout derived from it) carries a cftime axis,
    # which cannot be compared against the pandas timestamps the metrics select
    # windows with -- `.sel(time=slice(...))` raises "different calendars".
    # Normalise here, at the single entry point into comparison form, so callers
    # never have to think about which calendar a store happened to use.
    if "time" in ds.coords:
        ds = ds.assign_coords(time=kernels.coerce_datetime_values(ds["time"].values))
    return ds


def model_cell_area(ds: xr.Dataset) -> xr.DataArray:
    """The rollout's own cell area, falling back to a spherical estimate."""
    if "areacello" in ds.coords or "areacello" in ds:
        return ds["areacello"]
    logger.warning(
        "Rollout carries no 'areacello'; falling back to a spherical area estimate"
    )
    return xr.DataArray(
        spherical_cell_area(ds["lat"].values, ds["lon"].values),
        dims=["lat", "lon"],
        coords={"lat": ds["lat"], "lon": ds["lon"]},
    )


def model_depth_thickness(ds: xr.Dataset, dataset_spec: DatasetSpec) -> xr.DataArray:
    """Exact native layer thicknesses aligned to the rollout's `lev` axis."""
    lev = ds["lev"]
    if "dz" in ds.coords or "dz" in ds:
        return ds["dz"]
    thickness = np.asarray(dataset_spec.depth_thickness[: lev.size], dtype=float)
    return xr.DataArray(thickness, dims=["lev"], coords={"lev": lev})
