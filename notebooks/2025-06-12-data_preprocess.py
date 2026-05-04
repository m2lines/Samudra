#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8

# # Data Processing
#
# This notebook was run on leap2i2c with the Pangeo-notebook image.

# In[1]:


get_ipython().system("pip install xpartition coiled")


# In[2]:


import os

import fsspec
import gcm_filters
import numpy as np
import xarray as xr
import xesmf as xe  # type: ignore
from xgcm import Grid
import xpartition


# ### Setup

# In[3]:


# Dataset choice
dataset = "OM4"  # 'OM4' or 'CM4'
is_cm4 = dataset == "CM4"


# In[4]:


# Dimension names
longitude_dim: str = "x"
latitude_dim: str = "y"
time_dim: str = "time"
vertical_dim: str = "lev"
vertical_idim: str = "ilev"
rotation_angle: str = "angle"
sea_water_x_velocity: str = "uo"
sea_water_y_velocity: str = "vo"
sea_water_salinity: str = "so"
sea_water_potential_temperature: str = "thetao"
surface_temperature: str = "tos"
surface_downward_x_stress: str = "tauuo"
surface_downward_y_stress: str = "tauvo"
sea_ice_x_velocity: str = "UI"
sea_ice_y_velocity: str = "VI"
sea_ice_modeled: str = "EXT"
sea_ice_fraction: str = "sea_ice_fraction"
wetmask: str = "wetmask"
ocean_layer_thickness: str = "layer_thickness"

# Full field dimensions
full_field_dims = [longitude_dim, latitude_dim, time_dim]

# Rotated variables
if is_cm4:
    rotated_vars = (
        (sea_water_x_velocity, sea_water_y_velocity),
        (sea_ice_x_velocity, sea_ice_y_velocity),
        (surface_downward_x_stress, surface_downward_y_stress),
    )
else:
    rotated_vars = (
        (sea_water_x_velocity, sea_water_y_velocity),
        (surface_downward_x_stress, surface_downward_y_stress),
    )

# 3D variables
vars_3d = (
    sea_water_x_velocity,
    sea_water_y_velocity,
    sea_water_salinity,
    sea_water_potential_temperature,
)


# In[5]:


# Output directory and files
data_output_directory = (
    "s3://fomo-data-eng/2025-06-11-data"  # "/pscratch/sd/s/suryad/Ocean_Emulator/data"
)
# RUN_KEY = "2025-03-25-cm4-piControl-sample-10yr-hal"

# run directory
if is_cm4:
    run_directory = (
        "s3://emulators/ai2_colab/2024-11-01-CM4-pre-industrial-control-simulation/"
    )
else:
    run_directory = "s3://emulators/jbusecke/ocean_emulators/OM4/"
run_directory_old = run_directory
run_directory = "s3://fomo-data-eng/2025-06-11-run/"

# Relevant data paths
ocean_static_path = "s3://emulators/ai2_colab/2024-11-01-CM4-pre-industrial-control-simulation/ocean_static.zarr"
if is_cm4:
    ocean_zarr = "ocean_5daily.zarr"
else:
    ocean_zarr = "OM4_raw_test.zarr"
ice_zarr = "ice_5daily.zarr"

# Static data paths
nc_grid_path = (
    "s3://emulators/ai2_colab/2024-11-11-static-data/ocean_static_no_mask_table.nc"
)
nc_mosaic_path = "s3://emulators/ai2_colab/2024-11-11-static-data/ocean_hgrid.nc"
nc_target_grid_path = "s3://emulators/sd5313/grids/gaussian_grid_360_by_720.nc"

# Static data names and renaming
ocean_static_names = ["wet", "hfgeou"]
ocean_static_renaming = {"xh": "x", "yh": "y", "wet": "sea_surface_fraction"}

# Chunking and dimension renaming
chunking = {time_dim: 10, latitude_dim: 360, longitude_dim: 720}
dim_renaming = {"x": "lon", "y": "lat"}

# n split for saving
END_TIME_SLICE = 730
n_split = 500


# In[ ]:


# In[6]:


# OSN Pod
import os

fs_osn = fsspec.filesystem(
    "s3",
    key=os.environ["OSN_BUCKET_ACCESS_KEY"],
    secret=os.environ["OSN_BUCKET_SECRET_KEY"],
    endpoint_url="https://nyu1.osn.mghpcc.org/",
)
afs_osn = fsspec.filesystem(
    "s3",
    key=os.environ["OSN_BUCKET_ACCESS_KEY"],
    secret=os.environ["OSN_BUCKET_SECRET_KEY"],
    endpoint_url="https://nyu1.osn.mghpcc.org/",
    asynchronous=True,
)


# In[ ]:


# In[7]:


xr.set_options(keep_attrs=True)

time_dim = time_dim
lat_dim = latitude_dim
lon_dim = longitude_dim
vdim = vertical_dim
vidim = vertical_idim

om_zarr_path = os.path.join(run_directory, ocean_zarr)
om_zarr_path_old = os.path.join(run_directory_old, ocean_zarr)
sis_zarr_path = os.path.join(run_directory, ice_zarr)


# In[ ]:


fsspec_caching = {
    "cache_type": "blockcache",  # block cache stores blocks of fixed size and uses eviction using a LRU strategy.
    "block_size": 8
    * 1024
    * 1024,  # size in bytes per block, adjust depends on the file size but the recommended size is in the MB
}


# In[9]:


# import zarr


# src = zarr.storage.FsspecStore(fs_osn, path=om_zarr_path_old.removeprefix("s3://"))
# ds = xr.open_zarr(src)
# ds.to_zarr(om_zarr_path, zarr_format=2, mode='w')


# In[10]:


# debug flag
debug = True


# In[11]:


# Data Read Checks
try:
    ds_grid = xr.open_dataset(
        fs_osn.open(nc_grid_path, "rb", **fsspec_caching), engine="h5netcdf"
    ).load()
    ds_super_grid = xr.open_dataset(
        fs_osn.open(nc_mosaic_path, "rb", **fsspec_caching), engine="h5netcdf"
    ).load()
    print("read with h5netcdf")
except:
    ds_grid = xr.open_dataset(fs_osn.open(nc_grid_path, "rb"), engine="scipy").load()
    ds_super_grid = xr.open_dataset(
        fs_osn.open(nc_mosaic_path, "rb"), engine="scipy"
    ).load()
    print("read with scipy")

try:
    ds_target_grid = xr.open_dataset(
        fs_osn.open(nc_target_grid_path, "rb"),
        engine="h5netcdf",
    ).load()
except:
    ds_target_grid = xr.open_dataset(
        fs_osn.open(nc_target_grid_path, "rb"),
        engine="scipy",
    ).load()


# ### Pre-processing

# In[12]:


def _pick_first_element_of_missing_dims(mask: xr.DataArray, data: xr.DataArray):
    missing_dims = [di for di in mask.dims if di not in data.dims]
    if len(missing_dims) == 0:
        return mask
    else:
        return mask.isel({di: 0 for di in missing_dims})


def apply_mask(ds: xr.Dataset, mask: xr.DataArray):
    """Applies mask to same and lower dimensional data"""
    ds_out = xr.Dataset(attrs=ds.attrs)
    for var in ds.data_vars:
        data = ds[var]
        mask_pruned = _pick_first_element_of_missing_dims(mask, data)
        ds_out[var] = data.where(mask_pruned)
    return ds_out


def split_2d_3d(ds: xr.Dataset, depth_dim="lev"):
    ds_2d = xr.Dataset({v: ds[v] for v in ds.data_vars if depth_dim not in ds[v].dims})
    ds_3d = xr.Dataset({v: ds[v] for v in ds.data_vars if depth_dim in ds[v].dims})
    return ds_2d, ds_3d


# In[13]:


def interpolate_to_cell_centers(
    ds: xr.Dataset,
    like: xr.DataArray,
    grid: Grid,
):
    """Interplate variables defined on cell boundaries to cell centers.

    Args:
        ds: Input dataset.
        like: The data array to use as a template for the interpolation.
        grid: The grid object which will perform the interplation.

    """
    xh, yh = grid.axes["X"].coords["center"], grid.axes["Y"].coords["center"]
    xq, yq = grid.axes["X"].coords["right"], grid.axes["Y"].coords["right"]
    ds_interpolated = xr.Dataset()
    for var in ds.data_vars:
        da = ds[var]
        if set([xh, yh]).issubset(da.dims):
            ds_interpolated[var] = da
        elif xq in da.dims or yq in da.dims:
            # fill the velocities with 0 before interpolation to avoid mismatches in nans
            ds_interpolated[var] = grid.interp_like(da.fillna(0), like)
        if var in ds_interpolated:
            ds_interpolated[var].attrs = da.attrs

    return ds_interpolated


# In[14]:


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


import zarr


def om4_preprocessing(zarr_data_path, nc_grid_path, nc_mosaic_path):
    """OM4 specific preprocessing"""
    # zstore = zarr.storage.FsspecStore(afs_osn, path=zarr_data_path.removeprefix("s3://"))
    ds = xr.open_zarr(
        zarr_data_path,
        consolidated=True,
        # storage_options={"profile": "ocean_emulator_write"},
    )

    if "z_i" in ds.coords:
        ds = ds.rename({"z_i": "ilev", "z_l": "lev"})
        dz = xr.DataArray(
            ds.ilev.diff("ilev").values,
            dims=["lev"],
        ).astype("int64")
        ilev = ds["ilev"]
    else:
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
        ilev = xr.DataArray(
            [
                0,
                5,
                15,
                30,
                50,
                80,
                130,
                200,
                300,
                450,
                650,
                900,
                1200,
                1600,
                2100,
                2700,
                3500,
                4500,
                5500,
                6500,
            ],
            dims=["ilev"],
        )

    ds = ds.assign_coords(dz=dz)

    # trim excess padding
    if ds["xq"].size == ds["xh"].size + 1:
        ds = ds.isel(xq=slice(1, None))
    if ds["yq"].size == ds["yh"].size + 1:
        ds = ds.isel(yq=slice(1, None))

    grid = Grid(
        ds,
        coords={
            "X": {"center": "xh", "right": "xq"},
            "Y": {"center": "yh", "right": "yq"},
        },
        boundary="extend",
        periodic=["xh", "xq"],
    )
    ds_interpolated = interpolate_to_cell_centers(ds, ds.thetao, grid)

    # remove the same areas as for the tracers again
    tracer_wetmask = ~np.isnan(ds_interpolated.thetao.isel(time=0)).drop_vars("time")
    ds = apply_mask(ds_interpolated, tracer_wetmask)
    ds = ds.assign_coords(ilev=ilev, wetmask=tracer_wetmask)

    try:
        ds_grid = xr.open_dataset(
            fs_osn.open(nc_grid_path, "rb", **fsspec_caching), engine="h5netcdf"
        )
        print("opened grid with h5netcdf")
    except:
        ds_grid = xr.open_dataset(fs_osn.open(nc_grid_path, "rb"), engine="scipy")
        print("opened grid with scipy")

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
        "ilev",
        "lev",
        "yh",
        "areacello",
        "wetmask",
        "dz",
    ]
    drop_coords = [co for co in ds.coords.keys() if co not in required_coords]
    ds = ds.drop(drop_coords)

    try:
        ds_super_grid = xr.open_dataset(
            fs_osn.open(nc_mosaic_path, "rb", **fsspec_caching), engine="h5netcdf"
        )
        print("opened super grid with h5netcdf")
    except:
        ds_super_grid = xr.open_dataset(
            fs_osn.open(nc_mosaic_path, "rb"), engine="scipy"
        )
        print("opened super grid with scipy")

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

    return ds


# In[ ]:


# In[15]:


# import dataclasses
# from typing import Mapping, Any

# @dataclasses.dataclass
# class DaskConfig:
#     """Configuration for Dask, either LocalCluster or dask-gateway. See
#     https://docs.2i2c.org/user/howto/launch-dask-gateway-cluster for
#     dask-gateway usage on the LEAP-Pangeo 2i2c hub, where the relevant option is
#     "worker_resource_allocation" which can be one of "1CPU, 7.2Gi", "2CPU, 14.5Gi",
#     "4CPU, 28.9Gi", "8CPU, 57.9Gi", and "16CPU, 115.8Gi". Prepend
#     https://leap.2i2c.cloud to dask dashboard urls when using the LEAP-Pangeo
#     2i2c hub.

#     Attributes:
#         n_workers: number of Dask workers.
#         use_gateway: whether to use dask-gateway
#         cluster_options: additional options for configuring the LocalCluster or
#             Gateway.

#     """

#     n_workers: int = 64
#     use_gateway: bool = True
#     cluster_options: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {"worker_resource_allocation": "8CPU, 57.9Gi", "idle_timeout_minutes": 10})
#     _cluster = None

#     def get_client(self):
#         if self._cluster is not None:
#             return self._cluster.get_client()
#         if self.use_gateway:
#             from dask_gateway import Gateway

#             # use default gateway settings
#             gateway = Gateway()
#             options = gateway.cluster_options()
#             options.update(self.cluster_options)
#             self._cluster = gateway.new_cluster(options)
#             self._cluster.scale(self.n_workers)
#         else:
#             self._cluster = LocalCluster(
#                 n_workers=self.n_workers, **self.cluster_options
#             )
#         return self._cluster.get_client()

#     def close_cluster(self):
#         if self._cluster is not None:
#             self._cluster.close()
#             self._cluster = None


# In[16]:


# dask_config = DaskConfig()
# client = dask_config.get_client()
# client


# In[17]:


# from distributed import Client, LocalCluster
# # cluster = LocalCluster(threads_per_worker=4, n_workers=16)
# cluster = LocalCluster(threads_per_worker=2, n_workers=30)
# # cluster = LocalCluster()
# client = Client(cluster)
# client


# In[18]:


ds = om4_preprocessing(
    zarr_data_path=om_zarr_path,
    nc_grid_path=nc_grid_path,
    nc_mosaic_path=nc_mosaic_path,
)


# In[19]:


if debug:
    ds = ds.isel(time=slice(0, END_TIME_SLICE))


# In[20]:


ds = ds.chunk(
    {
        vdim: 1,
        lat_dim: -1,
        lon_dim: -1,
    }
)

urls = [
    om_zarr_path,
    nc_grid_path,
    nc_mosaic_path,
    nc_target_grid_path,
]

if is_cm4:
    urls = urls + [sis_zarr_path]


# In[21]:


import zarr

if ocean_static_names:
    zarr_path = ocean_static_path
    urls.append(zarr_path)
    zstore = zarr.storage.FsspecStore(afs_osn, path=zarr_path.removeprefix("s3://"))
    # with fs.open(zarr_path) as f:
    #     ds_static = xr.open_dataset(
    #         zarr_path, engine="zarr", backend_kwargs=backend_kwargs
    #     )[ocean_static_names]

    ds_static = xr.open_zarr(
        zstore,
        consolidated=True,
        # storage_options={"profile": "ocean_emulator_write"},
    )[ocean_static_names]
    ds_static = ds_static.rename(ocean_static_renaming)
    ds = xr.merge([ds, ds_static])

idepth_data = {}

for i, depth in enumerate(ds[vidim].values):
    idepth_data[f"idepth_{i}"] = xr.DataArray(depth)
    idepth_data[f"idepth_{i}"].attrs["units"] = "meters"
    idepth_data[f"idepth_{i}"].attrs["long_name"] = f"Depth at interface level-{i}"

idepth_ds = xr.Dataset(idepth_data)

if is_cm4:
    assert sea_ice_fraction in ds, (
        f"Sea ice fraction variable {sea_ice_fraction} is missing."
    )

print(f"Preprocessed size: {ds.nbytes / 1e9:.1f} GB")

# save attributes to add back after processing
attrs = {}
for var in ds.data_vars:
    attrs[var] = ds[var].attrs


# In[22]:


def rotate_vectors(u, v, angle):
    """Rotates vector components u and v using `angle`
    (assumed to be defined in deg, and in the CCW direction)
    Currently only works when all components are on the same grid position
    """
    # angle should be a 2d array
    if not len(angle.dims) == 2:
        raise ValueError(f"Expected only two dimensions on `angle`. Got {angle.dims}")
    # assert that all components are on the same position
    if not (
        set(angle.dims).issubset(set(u.dims)) and set(angle.dims).issubset(set(v.dims))
    ):
        raise ValueError("`u` and `v` need to be on the same grid position as `angle`.")

    # rotate velocities
    theta = np.deg2rad(angle)
    vec = xr.concat([u, v], dim="dim_in")
    # construct rotation matrix
    rot = xr.concat(
        [
            xr.concat([np.cos(theta), np.sin(theta)], dim="dim_out"),
            xr.concat([-np.sin(theta), np.cos(theta)], dim="dim_out"),
        ],
        dim="dim_in",
    )
    rotated_vector = xr.dot(vec, rot, dim="dim_in")
    u_rotated = rotated_vector.isel(dim_out=0)
    v_rotated = rotated_vector.isel(dim_out=1)
    return u_rotated, v_rotated


def spatially_filter(
    ds: xr.Dataset, w_mask, filter_scale=18, depth_dim="lev", y_dim="y", x_dim="x"
):
    """Applies a spatial filter with 3d/2d wetmask depending on the variable dimensions"""
    wmask_3d = (w_mask == 1).astype(int).reset_coords(drop=True)
    depth_indexer = {depth_dim: 0}
    wmask_2d = wmask_3d.isel(**depth_indexer)

    ds_2d, ds_3d = split_2d_3d(ds, depth_dim=depth_dim)
    ds_2d = ds_2d.reset_coords(drop=True)
    ds_3d = ds_3d.reset_coords(drop=True)

    filt_2d = gcm_filters.Filter(
        filter_scale=filter_scale,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
        grid_vars={"wet_mask": wmask_2d},  # why can gcm filters not accept bool masks?
    )
    filt_3d = gcm_filters.Filter(
        filter_scale=filter_scale,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
        grid_vars={"wet_mask": wmask_3d},  # why can gcm filters not accept bool masks?
    )
    datasets = [
        filt_2d.apply(ds_2d, dims=[y_dim, x_dim]),
        filt_3d.apply(ds_3d, dims=[y_dim, x_dim]),
    ]
    ds_filtered = xr.merge(datasets)
    # get attrs and coords back
    ds_filtered = ds_filtered.assign_coords({co: ds[co] for co in ds.coords})
    ds_filtered.attrs = ds.attrs
    return ds_filtered


def horizontal_regrid(ds, ds_target):
    """Regrid `ds` horizontally, and conserve the integral in space"""
    regridder_kwargs = dict(ignore_degenerate=True, periodic=True, unmapped_to_nan=True)

    # try to run this with higher precision (TODO: Test if this actually makes a difference).
    s = xr.Dataset(
        coords={
            co: ds[co].astype("float128") for co in ["lon", "lat", "lon_b", "lat_b"]
        }
    )
    t = xr.Dataset(
        coords={
            co: ds_target[co].astype("float128")
            for co in ["lon", "lat", "lon_b", "lat_b"]
        }
    )

    regridder = xe.Regridder(s, t, "conservative", **regridder_kwargs)
    ds_regridded = regridder(ds, skipna=True, na_thres=1)

    # get lon/lats from the target grid
    lon = ds_target.lon
    lat = ds_target.lat

    lon_b = ds_target.lon_b
    lat_b = ds_target.lat_b

    # get x and y values
    x = lon.isel(y=0)
    y = lat.isel(x=0)

    # calculate new area
    r_earth = 6356  # in km
    new_area = xe.util.cell_area(ds_target, r_earth) * 1e6

    ## calculate the wetmask afterwards...
    wetmask = ~np.isnan(ds_regridded.thetao.isel(time=0).drop_vars("time")).load()
    ocean_frac = regridder(ds.wetmask.astype("float64")).fillna(0.0)

    ds_regridded = ds_regridded.drop_vars(["lon_b", "lat_b"])
    ds_regridded = ds_regridded.assign_coords(
        lon=lon,
        lat=lat,
        lon_b=lon_b,
        lat_b=lat_b,
        areacello=new_area,
        x=x,
        y=y,
        wetmask=wetmask,
        ocean_fraction=ocean_frac,
    )
    ds_regridded.attrs = ds.attrs | ds_regridded.attrs

    return ds_regridded


# In[23]:


get_ipython().run_cell_magic(
    "time",
    "",
    "# rotation\nangle = ds[rotation_angle]\nfor varname_x, varname_y in rotated_vars:\n    x_rotated, y_rotated = rotate_vectors(ds[varname_x], ds[varname_y], angle)\n    ds[varname_x] = x_rotated.astype(np.float32)\n    ds[varname_y] = y_rotated.astype(np.float32)\n",
)


# In[24]:


# spatial filtering
ds = ds.chunk({lat_dim: -1, lon_dim: -1})
ds = spatially_filter(ds, ds[wetmask], depth_dim=vdim, y_dim=lat_dim, x_dim=lon_dim)
ds


# In[25]:


# regrid
try:
    ds_target_grid = xr.open_dataset(
        fs_osn.open(nc_target_grid_path, "rb"), engine="h5netcdf"
    )
except:
    ds_target_grid = xr.open_dataset(
        fs_osn.open(nc_target_grid_path, "rb"), engine="scipy"
    )

# TODO: remove target grid dimension assumptions
ds_target_grid = ds_target_grid.rename(
    {
        "grid_x": "x_b",
        "grid_y": "y_b",
        "grid_xt": lon_dim,
        "grid_yt": lat_dim,
        "grid_lon": "lon_b",
        "grid_lat": "lat_b",
        "grid_lont": "lon",
        "grid_latt": "lat",
    }
)
# fill nans in sea_ice_fraction to be
# consistent with ocean fraction in ocean_emulators
if sea_ice_fraction in ds.data_vars:
    ds[sea_ice_fraction] = ds[sea_ice_fraction].fillna(0.0)

ds_regridded = horizontal_regrid(ds, ds_target_grid).astype("float32")


# In[26]:


if is_cm4 and sea_ice_fraction in ds_regridded.data_vars:
    ds_regridded[sea_ice_fraction] = ds_regridded[sea_ice_fraction].fillna(0.0)
print(f"Regridded size: {ds_regridded.nbytes / 1e9:.1f} GB")

ds = ds_regridded
for var, attrs in attrs.items():
    ds[var].attrs = attrs

# fill ice velocity with NaN where sea ice is 0
if is_cm4:
    cond = ds[sea_ice_modeled] > 0.0

    for var in [sea_ice_x_velocity, sea_ice_y_velocity]:
        ds[var] = ds[var].where(cond, np.nan)

wetmask = ds[wetmask].astype(np.float32)
if len(wetmask.attrs) == 0:
    wetmask.attrs["long_name"] = "ocean mask"
    wetmask.attrs["units"] = "0 if land, 1 if ocean"

ds["mask"] = wetmask
vars_3d = list(vars_3d) + ["mask"]

for i, _ in enumerate(ds[vdim].values):
    for var in vars_3d:
        long_name = ds[var].long_name
        ds[f"{var}_{i}"] = ds[var].isel({vdim: i})
        ds[f"{var}_{i}"].attrs["long_name"] = long_name + f" level-{i}"

ds = ds.drop_vars(vars_3d)
ds = ds.drop_dims(vdim)
ds = ds.reset_coords(drop=True)
ds = xr.merge([ds, idepth_ds])

# add 'sst' variable in degrees Kelvin
if is_cm4:
    ds["sst"] = ds[surface_temperature].copy() + 273.15
    ds["sst"].attrs["long_name"] = "Sea surface temperature"
    ds["sst"].attrs["units"] = "K"

ds = ds.chunk(chunking)
ds.attrs["history"] = (
    "Dataset computed on the suryadheeshjith/Ocean_Emulator repository"
    f" using following input sources: {urls}."
)

drop_dims = [x for x in list(ds.dims) if x not in full_field_dims]
ds = ds.drop_dims(drop_dims)

# rename renaming:
if is_cm4:
    ds = ds.rename(dim_renaming)
ds


# In[27]:


# n_partitions = n_split

# # set RESUME_PARTITION to the last successfully completed segment number
# # use this in case computation hangs and you have to restart the kernel
# RESUME_PARTITION = 0

# output_store = os.path.join(data_output_directory, f"{RUN_KEY}.zarr")

# for i in range(RESUME_PARTITION, n_partitions):
#     if i == 0:
#         ds.partition.initialize_store(output_store)

#     print(f"Writing segment {i + 1} / {n_partitions}")
#     ds.partition.write(
#         output_store,
#         n_partitions,
#         ["time"],
#         i,
#         collect_variable_writes=True,
#     )


# In[29]:


import coiled

cluster = coiled.Cluster(
    n_workers=32, worker_memory="64 GiB", scheduler_memory="128 GiB"
)
client = cluster.get_client()


# In[30]:


# test things run on the dask cluster
import dask.array as dsa

arr = dsa.ones((100,), chunks=(10,))
arr.compute()


# In[31]:


# dest_name =  f"{data_output_directory}/{dataset}_sample_halfdeg_10yr.zarr"
# dest_name
# {var: {"compressor": None} for var in ds.data_vars}
ds.to_zarr(
    f"{data_output_directory}/{dataset}_sample_halfdeg_10yr.zarr",
    encoding={var: {"compressor": None} for var in ds.data_vars},
    mode="w",
    zarr_format=2,
)  # write data


# Upload locally saved files to OSN Pod with:
#
# 1. Install rclone in terminal
#
#     ```conda install conda-forge::rclone -y```
# 2. Setup rclone file if not already (https://leap-stc.github.io/guides/data_guides/external_workflow.html#authentication)
# 3. Copy to OSN (https://leap-stc.github.io/guides/data_guides/external_workflow.html#moving-data)
#
#    ```rclone copy gcs:leap-scratch/suryadheeshjith/OM4_sample_halfdeg_10yr.zarr ocean_emulator_write:emulators/sd5313/test/om4_sample_halfdeg_10yr.zarr```
#

# In[ ]:
