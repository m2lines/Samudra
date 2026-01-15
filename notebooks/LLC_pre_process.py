# -------------------------------
# Ocean Emulator OM4 data pre_process (Runs locally)
# -------------------------------

#%%
# dependencies
import os
import numpy as np
import xarray as xr
from xgcm import Grid
import gcm_filters
import zarr
import xpartition
from dask.distributed import Client, LocalCluster
import dask.array as da
import sys


#%%
# choose dataset
dataset = "LLC4320"

#%%
# assign dimensions
longitude_dim = "XC"
latitude_dim = "YC"
time_dim = "time"
vertical_dim = "Z"
#vertical_idim = "ilev"
#rotation_angle = "angle"
sea_water_x_velocity = "U"
sea_water_y_velocity = "V"
sea_water_salinity = "Salt"
sea_water_potential_temperature = "Theta"
#surface_temperature = "tos"
#surface_downward_x_stress = "tauuo"
#surface_downward_y_stress = "tauvo"
#sea_ice_x_velocity = "UI"
#sea_ice_y_velocity = "VI"
#sea_ice_modeled = "EXT"
#sea_ice_fraction = "sea_ice_fraction"
#wetmask = "wetmask"
#ocean_layer_thickness = "layer_thickness"

full_field_dims = [longitude_dim, latitude_dim, time_dim]

#rotated_vars = (
#    (sea_water_x_velocity, sea_water_y_velocity),
#    (surface_downward_x_stress, surface_downward_y_stress),
#)
#if is_cm4:
#    rotated_vars += ((sea_ice_x_velocity, sea_ice_y_velocity),)

vars_3d = (
    sea_water_x_velocity,
    sea_water_y_velocity,
    sea_water_salinity,
    sea_water_potential_temperature,
)

#%%
# define local paths
data_dir = "/orcd/data/abodner/003/LLC4320/LLC4320"  # directory containing local zarr/nc files

data_output_directory = "/orcd/data/abodner/002/cody/LLC_fragment_processed_out"
os.makedirs(data_output_directory, exist_ok=True)

chunking = {time_dim: 10, latitude_dim: 360, longitude_dim: 720}
dim_renaming = {"x": "lon", "y": "lat"}

xr.set_options(keep_attrs=True)

#%%
# helper functions
def _pick_first_element_of_missing_dims(mask: xr.DataArray, data: xr.DataArray):
    missing_dims = [di for di in mask.dims if di not in data.dims]
    if not missing_dims:
        return mask
    return mask.isel({di: 0 for di in missing_dims})

def apply_mask(ds: xr.Dataset, mask: xr.DataArray):
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

def interpolate_to_cell_centers(ds: xr.Dataset, like: xr.DataArray, grid: Grid):
    ds_interpolated = xr.Dataset()
    xh, yh = grid.axes["X"].coords["center"], grid.axes["Y"].coords["center"]
    xq, yq = grid.axes["X"].coords["outer"], grid.axes["Y"].coords["outer"]
    for var in ds.data_vars:
        da = ds[var]
        if set([xh, yh]).issubset(da.dims):
            ds_interpolated[var] = da
        elif xq in da.dims or yq in da.dims:
            ds_interpolated[var] = grid.interp_like(da.fillna(0), like)
        if var in ds_interpolated:
            ds_interpolated[var].attrs = da.attrs
    return ds_interpolated

def rotate_vectors(u, v, angle):
    theta = np.deg2rad(angle)
    vec = xr.concat([u, v], dim="dim_in")
    rot = xr.concat(
        [xr.concat([np.cos(theta), np.sin(theta)], dim="dim_out"),
         xr.concat([-np.sin(theta), np.cos(theta)], dim="dim_out")],
        dim="dim_in"
    )
    rotated_vector = xr.dot(vec, rot, dim="dim_in")
    u_rotated = rotated_vector.isel(dim_out=0)
    v_rotated = rotated_vector.isel(dim_out=1)
    return u_rotated, v_rotated

def spatially_filter(ds: xr.Dataset, w_mask, filter_scale=18, depth_dim="lev", y_dim="y", x_dim="x"):
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
        grid_vars={"wet_mask": wmask_2d}
    )
    filt_3d = gcm_filters.Filter(
        filter_scale=filter_scale,
        dx_min=1,
        filter_shape=gcm_filters.FilterShape.GAUSSIAN,
        grid_type=gcm_filters.GridType.REGULAR_WITH_LAND,
        grid_vars={"wet_mask": wmask_3d}
    )
    
    datasets = [
        filt_2d.apply(ds_2d, dims=[y_dim, x_dim]),
        filt_3d.apply(ds_3d, dims=[y_dim, x_dim])
    ]
    
    ds_filtered = xr.merge(datasets)
    ds_filtered = ds_filtered.assign_coords({co: ds[co] for co in ds.coords})
    ds_filtered.attrs = ds.attrs
    return ds_filtered

    # debugging
def find_tuples(ds):
    problems = []

    # 1. Dataset-level attrs
    for k, v in ds.attrs.items():
        if isinstance(v, tuple):
            problems.append(("dataset_attr", k, v))

    # 2. Variable-level attrs
    for var in ds.variables:
        for k, v in ds[var].attrs.items():
            if isinstance(v, tuple):
                problems.append(("var_attr", var, k, v))

    # 3. Variable values
    for var in ds.variables:
        data = ds[var].values
        if isinstance(data, tuple):
            problems.append(("var_data", var, "values", data))

    # 4. Dask graph nodes (hidden tuple chunks)
        data = ds[var].data
        if isinstance(data, da.Array):
            for k, v in data.dask.items():
                if isinstance(v, tuple):
                    problems.append(("dask_chunk", var, k, v))

    # Report and abort if found
    if problems:
        print("❌ Tuples detected in dataset:")
        for item in problems:
            print("   ", item)
        sys.exit(1)  # abort
    else:
        print("✅ no tuples detected")

    return []

#%%
# define main and initialize dask
def main():
    requested_memory=256
    n_workers = 16
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        memory_limit=f'{requested_memory/n_workers}GB'
    )
    client = Client(cluster)
    client.dashboard_link
    print('requested '+ f'{requested_memory}GB' + ' memory')

    #%%
    # load datasets
    #ds_grid = xr.open_dataset(nc_grid_path, engine="h5netcdf")
    #ds_super_grid = xr.open_dataset(nc_mosaic_path, engine="h5netcdf")
    #ds_target_grid = xr.open_dataset(nc_target_grid_path, engine="h5netcdf")
    ds = xr.open_zarr(data_dir, consolidated=False)[['U','V','Salt','Theta']]
    ds = ds.isel(i=slice(2000,4000), j= slice(1000,3000), k = slice(0,20), face=7)
    ds = ds.chunk({'time': 1, 'k': 10, 'j': 500, 'i': 500})

    # check vertical coords
 #   if "z_i" in ds.coords:
 #       ds = ds.rename({"z_i": "ilev", "z_l": "lev"})
 #   else:
 #       dz = xr.DataArray([5,10,15,20,30,50,70,100,150,200,250,300,400,500,600,800,1000,1000,1000], dims=["lev"])
 #       ilev = xr.DataArray([0,5,15,30,50,80,130,200,300,450,650,900,1200,1600,2100,2700,3500,4500,5500,6500], dims=["ilev"])
 #       ds = ds.assign_coords(dz=dz, ilev=ilev)
  #  find_tuples(ds)

    #%%
    # create XGCM Grid
   # try: 
   #     print(ds)
   # except:
   #     print('no printable ds')

    coords = {
        "X": {"center": "i", "outer": "i_g"},
        "Y": {"center": "j", "outer": "j_g"},
        "Z": {"center": "k", "left": "k_l", "right": "k_u"},
        "T": {"center": "time"},
    }

    grid = Grid(
        ds,
        coords=coords,
        boundary="extend",
        periodic=["X"],
     #   periodic=["xh", "xq"]#,
        autoparse_metadata=False
    )
  #  find_tuples(ds)
    # interpolate to cell centers
    ds = interpolate_to_cell_centers(ds, ds.Theta, grid)
    ds = ds.persist()
  #  ds = ds.persist()
  #  find_tuples(ds)

    #%%
    # rotate vectors  #no need as uo is already eastward and vo is laready northward (lat-lon)
    #angle = ds[rotation_angle]
    #for var_x, var_y in rotated_vars:
    #    ds[var_x], ds[var_y] = rotate_vectors(ds[var_x], ds[var_y], angle)
    #ds = ds.persist()

    #%% spatial filter
 #   ds = spatially_filter(ds, ds[wetmask], depth_dim=vertical_dim, y_dim=latitude_dim, x_dim=longitude_dim)
 #   ds = ds.persist()
 #   find_tuples(ds)

    #%%
    # save processed data! :D
    output_path = os.path.join(data_output_directory, f"{dataset}_processed_LLC_test.zarr")
    ds.to_zarr(output_path, mode="w", encoding={var: {"compressor": None} for var in ds.data_vars})
    print(f"Processed dataset saved at: {output_path}")

#%% run main
if __name__ == "__main__":
    main()
