import numpy as np
import xarray as xr
from xgcm import Grid
import fsspec

from .gfdl_om4 import om4_preprocessing
from .interpolate import interpolate_to_cell_centers


def sis2_preprocessing(zarr_data_path, backend_kwargs=None):
    """SIS2.0 specific preprocessing

    Args:
        zarr_data_path (str): path to the sea ice model output
    """
    ds = xr.open_dataset(zarr_data_path, engine="zarr", chunks={}, backend_kwargs=backend_kwargs)

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


def cm4_preprocessing(om_zarr_path, sis_zarr_path, nc_grid_path, nc_mosaic_path, fs=fsspec, backend_kwargs=None):
    """CM4 specific preprocessing

    Args:
        om_zarr_path (str): path to the ocean model output
        sis_zarr_path (str): path to the sea ice model output
        nc_grid_path (str): path to the grid file
        nc_mosaic_path (str): path to the mosaic file
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
