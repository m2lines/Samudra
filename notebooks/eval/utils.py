import os
from datetime import datetime

import numpy as np
import xarray as xr


def combine_all_variables_by_lev(ds, old_ds_gt, combine_vars):
    """
    Combine variables in the dataset along a new 'lev' dimension based on their suffix.

    Parameters:
    ds (xarray.Dataset): The input dataset containing variables with suffixes
                        (e.g., thetao_0, so_1).
    old_ds_gt (xarray.Dataset): The ground truth dataset containing lev values
    ls_3D (list): List of variable prefixes to combine.

    Returns:
    xarray.Dataset: The dataset with combined variables and a new 'lev' dimension.
    """
    for v in combine_vars:
        # print(f"Processing {v}")
        levels = old_ds_gt.lev.values
        level_numbers = [i for i in range(19)]
        sorted_vars = [v + "_" + str(lev) for lev in level_numbers]
        # print(sorted_vars)
        combined = xr.concat([ds[var] for var in sorted_vars], dim="lev")
        combined = combined.assign_coords(lev=levels)
        ds[v] = combined
        ds = ds.drop_vars(sorted_vars)
    return ds


def postprocess_ds(ds, old_ds_gt, areacello, dz, times, wetmask, coords=None):
    """
    Postprocess the dataset to make it compatible with plotting functions.
    """
    ds = ds.transpose("time", "lev", ...)
    if coords is not None:
        ds = ds.assign_coords(coords)
    ds["thetao"] = ds["thetao"].assign_attrs(
        long_name=r"${\theta_O}$", units=r"$\degree C$"
    )
    ds["lev"] = ds["lev"].assign_attrs(long_name="depth", units="m")
    ds["so"] = ds["so"].assign_attrs(long_name=r"${s}$", units=r"psu")
    ds["zos"] = ds["zos"].assign_attrs(long_name=r"SSH", units=r"m")
    ds["vo"] = ds["vo"].assign_attrs(long_name=r"${v}$", units=r"m/s")
    ds["uo"] = ds["uo"].assign_attrs(long_name=r"${u}$", units=r"m/s")

    if "init_time" in ds.coords:
        ds = ds.drop(["init_time", "valid_time"])

    for var in ds.data_vars:
        if "lev" in ds[var].dims:
            ds[var] = ds[var].where(wetmask)
        else:
            ds[var] = ds[var].where(wetmask.isel(lev=0))

    ds["time"] = times
    ds["areacello"] = (["lat", "lon"], areacello)
    ds["dz"] = ("lev", dz)

    return ds


def get_basin_masks(lat, lon):
    def process_mask(mask):
        mask = mask.where(mask != 0, np.nan)
        mask = mask.transpose("lat", "lon")
        mask = mask.assign_coords(lat=lat, lon=lon)
        mask = mask.rename({"lat": "y", "lon": "x"})
        return mask

    atlantic_mask0 = xr.open_dataset("/pscratch/sd/s/suryad/data/basin_At_noArctic.nc")[
        "basin"
    ]
    atlantic_mask = atlantic_mask0.where(atlantic_mask0["lat"] >= -32)
    atlantic_mask = process_mask(atlantic_mask)
    pacific_mask0 = xr.open_dataset("/pscratch/sd/s/suryad/data/basin_Pa.nc")["basin"]
    pacific_mask = pacific_mask0.where(pacific_mask0["lat"] >= -32)
    pacific_mask = process_mask(pacific_mask)
    indian_ocean_mask0 = xr.open_dataset("/pscratch/sd/s/suryad/data/basin_In.nc")[
        "basin"
    ]
    indian_ocean_mask = indian_ocean_mask0.where(indian_ocean_mask0["lat"] >= -32)
    indian_ocean_mask = process_mask(indian_ocean_mask)
    southern_ocean_mask0 = xr.open_dataset(
        "/pscratch/sd/s/suryad/data/basin_SO_32S.nc"
    )["basin"]
    southern_ocean_mask = process_mask(southern_ocean_mask0)
    arctic_mask0 = xr.open_dataset("/pscratch/sd/s/suryad/data/basin_Arctic.nc")[
        "basin"
    ]
    arctic_ocean_mask = process_mask(arctic_mask0)

    basin_masks = xr.Dataset(
        {
            "Atlantic": atlantic_mask,
            "Pacific": pacific_mask,
            "Southern": southern_ocean_mask,
            "Indian": indian_ocean_mask,
            "Arctic": arctic_ocean_mask,
        }
    )
    return basin_masks


def convert_nc_to_zarr(nc_file, zarr_file):
    """Convert a NetCDF file to Zarr format.

    Args:
        nc_file (str): Path to input NetCDF file
        zarr_file (str): Path to output Zarr store
    """
    # Open the NetCDF dataset
    ds = xr.open_dataset(nc_file, engine="netcdf4")

    # if sample dim exists, use first sample
    if "sample" in ds.dims:
        ds = ds.isel(sample=0)

    # Chunk the data appropriately
    ds = ds.chunk({"time": 10, "lat": 180, "lon": 360})

    # Save to Zarr format with no compression
    ds.to_zarr(
        zarr_file,
        encoding={var: {"compressor": None} for var in ds.data_vars},
        mode="w",
    )


def get_output_path(pred_dict):
    return (
        "../outputs/"
        + str(datetime.now())[:10]
        + "_"
        + "_".join([pred_dict[k]["run_name"] for k in pred_dict.keys()])
    )


def create_output_dir(output_path):
    # Create folder paths
    timeseries_path = os.path.join(output_path, f"Timeseries")
    if not os.path.isdir(timeseries_path):
        os.makedirs(timeseries_path)

    ohc_path = os.path.join(output_path, f"OHC")
    if not os.path.isdir(ohc_path):
        os.makedirs(ohc_path)

    temp_path = os.path.join(output_path, f"Temperature")
    if not os.path.isdir(temp_path):
        os.makedirs(temp_path)

    salinity_path = os.path.join(output_path, f"Salinity")
    if not os.path.isdir(salinity_path):
        os.makedirs(salinity_path)

    pdfs_path = os.path.join(output_path, f"PDFs")
    if not os.path.isdir(pdfs_path):
        os.makedirs(pdfs_path)

    enso_path = os.path.join(output_path, f"ENSO")
    if not os.path.isdir(enso_path):
        os.makedirs(enso_path)

    metrics_path = os.path.join(output_path, f"Metrics")
    if not os.path.isdir(metrics_path):
        os.makedirs(metrics_path)

    movie_path = os.path.join(output_path, f"Movies")
    if not os.path.isdir(movie_path):
        os.makedirs(movie_path)

    return (
        timeseries_path,
        ohc_path,
        temp_path,
        salinity_path,
        pdfs_path,
        enso_path,
        metrics_path,
        movie_path,
    )


def remove_climatology(ds):
    # Compute the climatology on the detrended data
    climatology = ds.groupby("time.dayofyear").mean("time").compute()

    # Remove the seasonal cycle (climatology) from the detrended data
    day_of_year = ds["time"].dt.dayofyear
    res = (ds - climatology.sel(dayofyear=day_of_year)).compute()

    return res
