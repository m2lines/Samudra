import fsspec
import numpy as np
import xarray as xr
from ocean_emulators.dataset_validation import ds_processed_validate
from ocean_emulators.utils import apply_mask
from xarray_schema import SchemaError
from xgcm import Grid

from .interpolate import interpolate_to_cell_centers


# load supergrid and extract the angles
# Some awesome material to understand the 'supergrid' (is that the same as the mosaic?) https://gist.github.com/adcroft/c1e207024fe1189b43dddc5f1fe7dd6c
def convert_super_grid(ds_super_grid: xr.Dataset):
    h_rename = {"nyp": "yh", "nxp": "xh"}
    b_rename = {"nyp": "yh_b", "nxp": "xh_b"}

    h_indicies = dict(nyp=slice(1, None, 2), nxp=slice(1, None, 2))
    b_indicies = dict(
        nyp=slice(0, None, 2), nxp=slice(0, None, 2)
    )  # locations of 'bound variables required by xesmf

    angle_h = ds_super_grid.angle_dx.isel(**h_indicies).rename(h_rename)
    lon_h = ds_super_grid.x.isel(**h_indicies).rename(h_rename)
    lat_h = ds_super_grid.y.isel(**h_indicies).rename(h_rename)

    lon_b = ds_super_grid.x.isel(**b_indicies).rename(b_rename)
    lat_b = ds_super_grid.y.isel(**b_indicies).rename(b_rename)
    return angle_h, lon_h, lat_h, lon_b, lat_b


def om4_preprocessing(zarr_data_path, nc_grid_path, nc_mosaic_path):
    """OM4 specific preprocessing"""
    ds = xr.open_dataset(zarr_data_path, engine="zarr", chunks={})

    if "z_l" in ds.coords:
        ds = ds.rename({"z_l": "lev"})

    # add vertical info
    dz = xr.DataArray(
        [
            5,
            10,
            15,
            20,
            30,
            50,
            70,
            100,
            150,
            200,
            250,
            300,
            400,
            500,
            600,
            800,
            1000,
            1000,
            1000,
        ],
        dims=["lev"],
    )
    ds = ds.assign_coords(dz=dz)

    ds_interpolated = interpolate_to_cell_centers(ds, ds.thetao)

    # remove the same areas as for the tracers again
    tracer_wetmask = ~np.isnan(ds_interpolated.thetao.isel(time=0)).drop_vars("time")
    ds = apply_mask(ds_interpolated, tracer_wetmask)
    ds = ds.assign_coords(wetmask=tracer_wetmask)

    with fsspec.open(nc_grid_path) as f:
        ds_grid = xr.open_dataset(f)
    ds_grid = ds_grid.drop_vars("time")
    ds_grid = ds_grid.set_coords([v for v in ds_grid.data_vars])
    # ds_grid
    # ds = xr.merge([ds, ds_grid])
    ds = ds.assign_coords(
        lon=ds_grid.geolon, lat=ds_grid.geolat, areacello=ds_grid.areacello
    )

    # drop (for now) all the coords on non-tracer position
    required_coords = [
        "lon",
        "time",
        "xh",
        "lat",
        "lev",
        "yh",
        "areacello",
        "wetmask",
        "dz",
    ]
    drop_coords = [co for co in ds.coords.keys() if co not in required_coords]
    ds = ds.drop(drop_coords)

    with fsspec.open(nc_mosaic_path) as f:
        ds_super_grid = xr.open_dataset(f).load()

    a, lon, lat, lon_b, lat_b = convert_super_grid(ds_super_grid)
    lon_expected = ds_grid.load().geolon.reset_coords(drop=True).drop(["xh", "yh"])
    lat_expected = ds_grid.load().geolat.reset_coords(drop=True).drop(["xh", "yh"])

    # asser that the grid positions extracted are correct (this should maybe live in a test for an upstream function?)
    xr.testing.assert_allclose(lon, lon_expected)
    xr.testing.assert_allclose(lat, lat_expected)

    ds = ds.assign_coords(lon_b=lon_b, lat_b=lat_b, angle=a, lon=lon, lat=lat)
    ds = ds.rename({"xh": "x", "yh": "y", "xh_b": "x_b", "yh_b": "y_b"})
    if "time_bnds" in ds.data_vars:
        ds = ds.drop_vars(["time_bnds"])
    ds = ds.astype(np.float32)
    # higher precision for the area
    ds = ds.assign_coords(areacello=ds.areacello.astype("float64"))
    try:
        ds_processed_validate(ds)
    except SchemaError:
        print("Failed validation with {e}")
    return ds
