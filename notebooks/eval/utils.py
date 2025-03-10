import json
import os
from copy import deepcopy
from datetime import datetime

import cftime
import numpy as np
import pandas as pd
import xarray as xr


def _combine_variables_by_level(ds, lev, combine_vars):
    """
    Combine variables in the dataset along a new 'lev' dimension based on their suffix.

    Parameters:
    ds (xarray.Dataset): The input dataset containing variables with suffixes
                        (e.g., thetao_0, so_1).
    lev (xarray.DataArray): lev dataarray containing lev values
    combine_vars (list): List of variable prefixes to combine.

    Returns:
    xarray.Dataset: The dataset with combined variables and a new 'lev' dimension.
    """
    for v in combine_vars:
        levels = lev.values
        level_numbers = [i for i in range(19)]
        sorted_vars = [v + "_" + str(lev) for lev in level_numbers]
        if sorted_vars[0] not in ds.data_vars:
            print(f"Variable {v} is 2D, skipping combine...")
            continue
        combined = xr.concat([ds[var] for var in sorted_vars], dim="lev")
        combined = combined.assign_coords(lev=levels)
        ds[v] = combined
        ds = ds.drop_vars(sorted_vars)
    return ds


def combine_variables_by_level(ds_groundtruth, lev, pred_dict, combine_ground=True):
    """
    Combine variables by level for ground truth and predictions.

    Parameters:
    ds_groundtruth (xarray.Dataset): The ground truth dataset.
    lev (xarray.DataArray): lev dataarray containing lev values
    pred_dict (dict): Dictionary containing prediction datasets.

    Returns:
    xarray.Dataset, dict: Updated ground truth and prediction datasets.
    """
    if combine_ground:
        ds_groundtruth = _combine_variables_by_level(
            ds_groundtruth, lev, ["thetao", "so", "uo", "vo", "mask"]
        )
    for key in pred_dict.keys():
        pred_dict[key]["ds_prediction"] = _combine_variables_by_level(
            pred_dict[key]["ds_prediction"], lev, pred_dict[key]["ls"]
        )
    return ds_groundtruth, pred_dict


def _postprocess_for_plot(ds, areacello, dz, times, wetmask, coords=None):
    """
    Postprocess the dataset to make it compatible with plotting functions.
    """
    ds = ds.transpose("time", "lev", ...)
    ds["time"] = times
    if coords is not None:
        ds = ds.assign_coords(coords)
    if "thetao" in ds.data_vars:
        ds["thetao"] = ds["thetao"].assign_attrs(
            long_name=r"${\theta_O}$", units=r"$\degree C$"
        )
    if "so" in ds.data_vars:
        ds["so"] = ds["so"].assign_attrs(long_name=r"${s}$", units=r"psu")
    if "zos" in ds.data_vars:
        ds["zos"] = ds["zos"].assign_attrs(long_name=r"SSH", units=r"m")
    if "vo" in ds.data_vars:
        ds["vo"] = ds["vo"].assign_attrs(long_name=r"${v}$", units=r"m/s")
    if "uo" in ds.data_vars:
        ds["uo"] = ds["uo"].assign_attrs(long_name=r"${u}$", units=r"m/s")

    ds["lev"] = ds["lev"].assign_attrs(long_name="depth", units="m")
    if "init_time" in ds.coords:
        ds = ds.drop(["init_time", "valid_time"])

    for var in ds.data_vars:
        if "lev" in ds[var].dims:
            ds[var] = ds[var].where(wetmask)
        else:
            ds[var] = ds[var].where(wetmask.isel(lev=0))

    ds["areacello"] = (["lat", "lon"], areacello)
    ds["dz"] = ("lev", dz)
    return ds


def postprocess_for_plot(ds_groundtruth, areacello, dz, pred_dict):
    """
    Postprocess for plotting.

    Parameters:
    ds_groundtruth (xarray.Dataset): The ground truth dataset.
    areacello (xarray.DataArray): areacello dataarray.
    dz (xarray.DataArray): dz dataarray.
    pred_dict (dict): Dictionary containing prediction datasets.

    Returns:
    xarray.Dataset, dict: Postprocessed ground truth and prediction datasets.
    """
    areacello = areacello.values
    dz = dz.data
    times = ds_groundtruth.time

    # Masking land with NaNs
    if "mask" in ds_groundtruth.data_vars:
        wetmask = ds_groundtruth["mask"].isel(time=0)
    else:
        wetmask = ds_groundtruth.wetmask

    ds_groundtruth = _postprocess_for_plot(
        ds_groundtruth, areacello, dz, times, wetmask
    )
    coords = ds_groundtruth.coords

    for key in pred_dict.keys():
        pred_dict[key]["ds_prediction"] = _postprocess_for_plot(
            pred_dict[key]["ds_prediction"],
            areacello,
            dz,
            times,
            wetmask,
            coords=coords,
        )
        # Rename lat and lon to y and x
        pred_dict[key]["ds_prediction"] = pred_dict[key]["ds_prediction"].rename(
            {"lat": "y", "lon": "x"}
        )

    # Rename lat and lon to y and x (This needs to be done in the end!)
    ds_groundtruth = ds_groundtruth.rename({"lat": "y", "lon": "x"})

    return ds_groundtruth, pred_dict


def get_plot_ready_cm4_data(pred_dict, output_path, long_rollout):
    """
    Get plot ready CM4 data.
    """
    print(f"Getting plot ready CM4 data with long rollout = {long_rollout}")
    # Read CM4 files
    old_ds_gt = xr.open_zarr(
        "/pscratch/sd/s/suryad/data/CM4_5daily_v0.4.0_preprocessed.zarr"
    )

    ds_groundtruth = xr.open_zarr(
        "/pscratch/sd/s/suryad/data/cm4_piControl_ocean_200yr_full_chunked.zarr"
    )

    # Create output directory
    if not os.path.isdir(os.path.join(output_path)):
        os.makedirs(os.path.join(output_path))

    ### Process data with common time slice and variable set
    if long_rollout:
        time_slice = slice("0251-01-01", "0350-12-27")

    else:
        time_slice = slice("0311-01-06", "0351-01-01")

    var_set = set()
    for key in pred_dict.keys():
        ds_prediction = xr.open_zarr(
            pred_dict[key]["path"], chunks={"time": 10, "lat": 180, "lon": 360}
        )
        # Fixing limits for hist = 1 and hist = 0 compatability
        if not long_rollout:  # Probably need a version of this for long rollout as well
            if ds_prediction.time.size == 2918:
                time_slice = slice("0311-01-11", "0350-12-27")
            elif ds_prediction.time.size != 2920:
                raise Exception(
                    "Are you sure your run is complete? Current prediction size: ",
                    ds_prediction.time.size,
                )

        var_ls = list(ds_prediction.data_vars.keys())
        if "tos" in var_ls:  # Current Data store does not have tos
            var_ls.remove("tos")
        var_set.update(set(var_ls))

    mask_vars = [v for v in ds_groundtruth if "mask_" in v]
    groundtruth_ls = list(var_set) + mask_vars
    ds_groundtruth = ds_groundtruth[groundtruth_ls].sel(time=time_slice)

    # Store ds_prediction
    copy_dict = deepcopy(pred_dict)

    for key in pred_dict.keys():
        ds_prediction = xr.open_zarr(
            pred_dict[key]["path"], chunks={"time": 10, "lat": 180, "lon": 360}
        )

        if long_rollout:
            ds_prediction = ds_prediction.isel(
                time=slice(-ds_groundtruth.time.size, None)
            )
        else:
            if (
                ds_prediction.time.size != ds_groundtruth.time.size
                and ds_groundtruth.time.size == 2918
            ):
                print(
                    f"Sizes different: {ds_prediction.time.size}!={ds_groundtruth.time.size}"
                )
                print(
                    "Updating prediction size considering (0311-01-06, 0351-01-01) -> (0311-01-11, 0350-12-27)"
                )
                # "0311-01-06", "0351-01-01" -> "0311-01-11", "0350-12-27"
                ds_prediction = ds_prediction.isel(
                    time=slice(1, 1 + ds_groundtruth.time.size)
                )

        assert ds_prediction.time.size == ds_groundtruth.time.size
        if "model_path" in ds_prediction.attrs:
            copy_dict[key]["model_path"] = ds_prediction.attrs["model_path"]

        pred_dict[key]["ds_prediction"] = ds_prediction

    with open(os.path.join(output_path, "compare_info.txt"), "w") as f:
        f.write(json.dumps(copy_dict, sort_keys=True, indent=4))

    ### Combine Variables by level
    ds_groundtruth, pred_dict = combine_variables_by_level(
        ds_groundtruth, old_ds_gt.lev, pred_dict
    )

    ### Postprocess predictions for plotting
    ds_groundtruth, pred_dict = postprocess_for_plot(
        ds_groundtruth, old_ds_gt.areacello, old_ds_gt.dz, pred_dict
    )

    return ds_groundtruth, pred_dict


def convert_datetime64_to_cftime(t):
    ts = pd.Timestamp(t)
    return cftime.DatetimeProlepticGregorian(
        ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, ts.microsecond
    )


def get_plot_ready_om4_data(pred_dict, output_path, long_rollout):
    """
    Get plot ready OM4 data.
    """
    print(f"Getting plot ready OM4 data with long rollout = {long_rollout}")
    # Read OM4 files
    if long_rollout:
        ds_groundtruth = xr.open_zarr(
            "/pscratch/sd/s/suryad/data/OM4_5daily_v0.2.1_with_hfds_anom_104_years_2014_netzerohfds_detrended"
        )
    else:
        ds_groundtruth = xr.open_zarr(
            "/pscratch/sd/s/suryad/data/OM4_5daily_v0.2.1.zarr"
        )
    # Renames so further processing is easier
    ds_groundtruth = ds_groundtruth.rename({"lat": "lat_t", "lon": "lon_t"})
    ds_groundtruth = ds_groundtruth.rename({"y": "lat", "x": "lon"})

    # Create output directory
    if not os.path.isdir(os.path.join(output_path)):
        os.makedirs(os.path.join(output_path))

    ### Process data with common time slice and variable set
    if long_rollout:
        num_timesteps = None
        for key in pred_dict.keys():
            ds_prediction = xr.open_zarr(
                pred_dict[key]["path"], chunks={"time": 10, "lat": 180, "lon": 360}
            )
            if num_timesteps is None:
                num_timesteps = ds_prediction.time.size
            else:
                assert (
                    num_timesteps == ds_prediction.time.size
                ), f"Sizes different for {key}: {num_timesteps}!={pred_dict[key]['ds_prediction'].time.size}"
        time_slice = slice("2014-01-13", None)
    else:
        time_slice = slice("2014-10-10", "2022-12-24")

    ds_groundtruth = ds_groundtruth.sel(time=time_slice)  # OM4

    # Store ds_prediction
    copy_dict = deepcopy(pred_dict)

    for key in pred_dict.keys():
        ds_prediction = xr.open_zarr(
            pred_dict[key]["path"], chunks={"time": 10, "lat": 180, "lon": 360}
        )

        if long_rollout:
            if ds_prediction.time.size < 7200:
                raise Exception(
                    "Are you sure your run is complete? Current prediction size: ",
                    ds_prediction.time.size,
                )
            ds_groundtruth = ds_groundtruth.isel(
                time=slice(None, ds_prediction.time.size)
            )
        else:  # Probably need a version of this for long rollout as well
            if ds_prediction.time.size != 600:
                raise Exception(
                    "Are you sure your run is complete? Current prediction size: ",
                    ds_prediction.time.size,
                )

        assert (
            ds_prediction.time.size == ds_groundtruth.time.size
        ), f"Sizes different for {key}: {ds_prediction.time.size}!={ds_groundtruth.time.size}"
        if "model_path" in ds_prediction.attrs:
            copy_dict[key]["model_path"] = ds_prediction.attrs["model_path"]

        pred_dict[key]["ds_prediction"] = ds_prediction

    with open(os.path.join(output_path, "compare_info.txt"), "w") as f:
        f.write(json.dumps(copy_dict, sort_keys=True, indent=4))

    ### Combine Variables by level
    ds_groundtruth, pred_dict = combine_variables_by_level(
        ds_groundtruth, ds_groundtruth.lev, pred_dict, combine_ground=False
    )

    ### Postprocess predictions for plotting
    ds_groundtruth, pred_dict = postprocess_for_plot(
        ds_groundtruth, ds_groundtruth.areacello, ds_groundtruth.dz, pred_dict
    )

    ### Convert time from np.datetime64 to DatetimeProlepticGregorian
    times = ds_groundtruth.time.values
    times_updated = [convert_datetime64_to_cftime(t) for t in times]
    ds_groundtruth["time"] = times_updated
    for key in pred_dict.keys():
        pred_dict[key]["ds_prediction"]["time"] = times_updated

    return ds_groundtruth, pred_dict


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
