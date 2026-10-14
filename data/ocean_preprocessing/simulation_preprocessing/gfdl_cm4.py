# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import datetime
from types import ModuleType
from typing import Any

import fsspec
import numpy as np
import xarray as xr
from fsspec.spec import AbstractFileSystem
from xgcm import Grid

from .gfdl_om4 import native_ocean_preprocessing, om4_preprocessing
from .interpolate import interpolate_to_cell_centers


def cm4_ocean_preprocessing(
    zarr_data_path: str,
    nc_grid_path: str,
    nc_mosaic_path: str,
    fs: ModuleType | AbstractFileSystem = fsspec,
    backend_kwargs: dict[str, Any] | None = None,
    time_bounds_path: str | None = None,
) -> xr.Dataset:
    """Prepare CM4 ocean output, preserving five-day interval-end times.

    The published raw CM4 store keeps ``average_DT`` but stores its time bounds
    in a separate file. Accept that companion file, or reconstruct the bounds
    from the verified averaging durations without shifting the timestamps.
    """
    ds = xr.open_dataset(
        zarr_data_path, engine="zarr", chunks={}, backend_kwargs=backend_kwargs
    )
    if ds.sizes.get("time", 0) == 0:
        raise ValueError("CM4 source has no timestamps")
    times = ds.time.values
    if any(getattr(t, "calendar", None) not in {"noleap", "365_day"} for t in times):
        raise ValueError("CM4 source must use the no-leap calendar")
    if any((b - a).total_seconds() != 5 * 86400 for a, b in zip(times, times[1:])):
        raise ValueError("CM4 source timestamps must be exactly five days apart")
    if "wfo" not in ds.data_vars:
        raise ValueError("CM4 source must contain freshwater flux 'wfo'")

    if time_bounds_path is not None:
        with fs.open(time_bounds_path, "rb") as handle:
            companion = xr.open_dataset(handle).load()
        if not np.array_equal(companion.time.values, times):
            raise ValueError(
                "CM4 companion time bounds must match the source timestamps"
            )
        bounds = companion.time_bnds
    elif "time_bnds" in ds:
        bounds = ds.time_bnds
    else:
        if "average_DT" not in ds or ds.average_DT.attrs.get("units") != "days":
            raise ValueError("CM4 requires time_bnds or average_DT in days")
        if not np.all(ds.average_DT.values == 5):
            raise ValueError("CM4 averaging intervals must be exactly five days")
        bounds = xr.DataArray(
            [[t - datetime.timedelta(days=5), t] for t in times],
            dims=("time", "nv"),
            coords={"time": ds.time},
        )
    if bounds.dims != ("time", "nv") or bounds.sizes["nv"] != 2:
        raise ValueError("CM4 time_bnds must have dimensions (time, nv=2)")
    if not np.array_equal(bounds.isel(nv=1).values, times) or any(
        (end - start).total_seconds() != 5 * 86400 for start, end in bounds.values
    ):
        raise ValueError(
            "CM4 timestamps must label the ends of five-day mean intervals"
        )
    bounds.encoding.update(
        {
            key: ds.time.encoding[key]
            for key in ("units", "calendar")
            if key in ds.time.encoding
        }
    )
    ds = ds.assign_coords(time_bnds=bounds)
    ds.time.attrs["bounds"] = "time_bnds"
    ds.attrs["time_sampling"] = "five-day mean, timestamp at interval end"
    return native_ocean_preprocessing(ds, nc_grid_path, nc_mosaic_path, fs=fs)


def sis2_preprocessing(zarr_data_path, backend_kwargs=None):
    """SIS2.0 specific preprocessing.

    Args:
        zarr_data_path (str): path to the sea ice model output
        backend_kwargs (dict | None): backend kwargs forwarded to ``xr.open_dataset``.
    """
    ds = xr.open_dataset(
        zarr_data_path, engine="zarr", chunks={}, backend_kwargs=backend_kwargs
    )

    # trim excess padding
    if ds["xB"].size == ds["xT"].size + 1:
        ds = ds.isel(xB=slice(1, None))
    if ds["yB"].size == ds["yT"].size + 1:
        ds = ds.isel(yB=slice(1, None))

    ds = ds.drop(["xTe", "yTe", "nv"])
    grid = Grid(
        ds,
        coords={
            "X": {"center": "xT", "right": "xB"},
            "Y": {"center": "yT", "right": "yB"},
        },
        boundary="extend",
        periodic=["xT", "xB"],
    )
    ds = interpolate_to_cell_centers(ds, ds.EXT, grid)
    ds = ds.astype(np.float32)
    return ds.rename({"xT": "x", "yT": "y"})


def cm4_preprocessing(
    om_zarr_path,
    sis_zarr_path,
    nc_grid_path,
    nc_mosaic_path,
    fs=fsspec,
    backend_kwargs=None,
):
    """CM4 specific preprocessing.

    Args:
        om_zarr_path (str): path to the ocean model output
        sis_zarr_path (str): path to the sea ice model output
        nc_grid_path (str): path to the grid file
        nc_mosaic_path (str): path to the mosaic file
        fs: filesystem object used to open remote files (defaults to ``fsspec``).
        backend_kwargs (dict | None): backend kwargs forwarded to ``xr.open_dataset``.
    """
    ds_om = om4_preprocessing(
        zarr_data_path=om_zarr_path,
        nc_grid_path=nc_grid_path,
        nc_mosaic_path=nc_mosaic_path,
        fs=fs,
        backend_kwargs=backend_kwargs,
    )
    ds_sis = sis2_preprocessing(sis_zarr_path, backend_kwargs=backend_kwargs)
    return xr.merge([ds_om, ds_sis])
