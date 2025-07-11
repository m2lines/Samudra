"""Data processing module for ocean visualization."""

from copy import deepcopy
import numpy as np
import xarray as xr
from ocean_emulators.constants import DEPTH_LEVELS, DEPTH_THICKNESS
from ocean_emulators.utils.data import spherical_area_weights


def rename_vars(data: xr.Dataset) -> xr.Dataset:
    """
    Rename variables if required.
    
    Converts OM4 data format variables from: var_lev_depthlevel (e.g., so_lev_1040_0)
    to: var_depthlevelidx (e.g., so_0)
    
    Args:
        data: Input dataset with OM4 format variable names
        
    Returns:
        Dataset with renamed variables
    """
    for var_str in data.variables:
        # OM4 data format has variables in the form: var_lev_depthlevel
        # ex. so_lev_1040_0. We need to convert into var_depthlevelidx
        if "_lev_" in var_str:
            var_split = var_str.split("_lev_")
            var = var_split[0]
            lev_in_depth = float(var_split[1].replace("_", "."))
            lev_in_depth_idx = DEPTH_LEVELS.index(lev_in_depth)
            data = data.rename({var_str: var + "_" + str(lev_in_depth_idx)})
    return data


def _combine_variables_by_level(ds: xr.Dataset, combine_vars: list) -> xr.Dataset:
    """
    Combine variables in the dataset along a new 'lev' dimension based on their suffix.

    Args:
        ds: Input dataset containing variables with suffixes (e.g., thetao_0, so_1)
        combine_vars: List of variable prefixes to combine

    Returns:
        Dataset with combined variables and a new 'lev' dimension
    """
    for v in combine_vars:
        level_numbers = [i for i in range(19)]
        sorted_vars = [v + "_" + str(lev) for lev in level_numbers]
        if sorted_vars[0] not in ds.data_vars:
            continue
        combined = xr.concat([ds[var] for var in sorted_vars], dim="lev")
        combined = combined.assign_coords(lev=DEPTH_LEVELS)
        ds[v] = combined
        ds = ds.drop_vars(sorted_vars)
    return ds


def combine_variables_by_level(ds_groundtruth: xr.Dataset, pred_dict: dict, combine_ground: bool = True) -> tuple[xr.Dataset, dict]:
    """
    Combine variables by level for ground truth and predictions.

    Args:
        ds_groundtruth: The ground truth dataset
        pred_dict: Dictionary containing prediction datasets
        combine_ground: Whether to combine ground truth variables

    Returns:
        Tuple of (updated ground truth dataset, updated prediction dictionary)
    """
    if combine_ground:
        ds_groundtruth = _combine_variables_by_level(
            ds_groundtruth, ["thetao", "so", "uo", "vo", "mask"]
        )
    for key in pred_dict.keys():
        pred_dict[key]["ds_prediction"] = _combine_variables_by_level(
            pred_dict[key]["ds_prediction"], pred_dict[key]["ls"]
        )
    return ds_groundtruth, pred_dict


def _postprocess_for_plot(ds: xr.Dataset, areacello: np.ndarray, dz: np.ndarray, 
                         times: xr.DataArray, wetmask: xr.DataArray, coords=None) -> xr.Dataset:
    """
    Postprocess the dataset to make it compatible with plotting functions.
    
    Args:
        ds: Input dataset
        areacello: Area weights array
        dz: Depth thickness array
        times: Time coordinates
        wetmask: Wet mask for land/ocean
        coords: Additional coordinates to assign
        
    Returns:
        Postprocessed dataset
    """
    ds = ds.transpose("time", "lev", ...)
    ds["time"] = times
    if coords is not None:
        ds = ds.assign_coords(coords)
    
    # Add units and labels for plotting
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

    # Apply wet mask
    for var in ds.data_vars:
        if "lev" in ds[var].dims:
            ds[var] = ds[var].where(wetmask)
        else:
            ds[var] = ds[var].where(wetmask.isel(lev=0))

    ds["areacello"] = (["lat", "lon"], areacello)
    ds["dz"] = ("lev", dz)
    return ds


def postprocess_for_plot(ds_groundtruth: xr.Dataset, areacello: xr.DataArray, 
                        dz: np.ndarray, pred_dict: dict) -> tuple[xr.Dataset, dict]:
    """
    Postprocess for plotting.

    Args:
        ds_groundtruth: The ground truth dataset
        areacello: Area weights dataarray
        dz: Depth thickness array
        pred_dict: Dictionary containing prediction datasets

    Returns:
        Tuple of (postprocessed ground truth dataset, postprocessed prediction dictionary)
    """
    areacello_values = areacello.values
    times = ds_groundtruth.time

    # Masking land with NaNs
    if "mask" in ds_groundtruth.data_vars:
        wetmask = ds_groundtruth["mask"].isel(
            time=0, missing_dims="ignore"
        )  # our data does not always have time for a mask
    else:
        wetmask = ds_groundtruth.wetmask

    ds_groundtruth = _postprocess_for_plot(
        ds_groundtruth, areacello_values, dz, times, wetmask
    )
    coords = ds_groundtruth.coords

    for key in pred_dict.keys():
        pred_dict[key]["ds_prediction"] = _postprocess_for_plot(
            pred_dict[key]["ds_prediction"],
            areacello_values,
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


def load_prediction_data(pred_dict: dict) -> dict:
    """
    Load prediction data from the configured paths.
    
    Args:
        pred_dict: Dictionary containing prediction configuration
        
    Returns:
        Updated prediction dictionary with loaded datasets
    """
    copy_dict = deepcopy(pred_dict)
    
    for key in pred_dict.keys():
        ds_prediction = xr.open_zarr(
            pred_dict[key]["path"], chunks={"time": 10, "lat": 180, "lon": 360}
        )
        
        if "model_path" in ds_prediction.attrs:
            copy_dict[key]["model_path"] = ds_prediction.attrs["model_path"]

        pred_dict[key]["ds_prediction"] = ds_prediction
    
    return pred_dict


def process_data(data: xr.Dataset, pred_dict: dict) -> tuple[xr.Dataset, dict]:
    """
    Get plot ready OM4 data.
    
    Main data processing pipeline that:
    1. Renames variables from OM4 format
    2. Loads prediction data
    3. Combines variables by level
    4. Postprocesses for plotting
    
    Args:
        data: Ground truth dataset
        pred_dict: Dictionary containing prediction configuration
        
    Returns:
        Tuple of (processed ground truth dataset, processed prediction dictionary)
    """
    # Rename variables from OM4 format
    ds_groundtruth = rename_vars(data)
    
    # Load prediction data
    pred_dict = load_prediction_data(pred_dict)
    
    # Validate time dimensions match
    for key in pred_dict.keys():
        ds_prediction = pred_dict[key]["ds_prediction"]
        assert ds_prediction.time.size == ds_groundtruth.time.size, (
            f"Sizes different for {key}: {ds_prediction.time.size}!="
            f"{ds_groundtruth.time.size}; prediction range is {ds_prediction.time.values[0]} to {ds_prediction.time.values[-1]}"
            f"groundtruth range is {ds_groundtruth.time.values[0]} to {ds_groundtruth.time.values[-1]}"
        )
    
    # Combine variables by level
    ds_groundtruth, pred_dict = combine_variables_by_level(ds_groundtruth, pred_dict)

    # Postprocess predictions for plotting
    ds_groundtruth, pred_dict = postprocess_for_plot(
        ds_groundtruth, ds_groundtruth.areacello, np.array(DEPTH_THICKNESS), pred_dict
    )

    return ds_groundtruth, pred_dict


def load_groundtruth_data(data_path: str) -> xr.Dataset:
    """
    Load and prepare ground truth data.
    
    Args:
        data_path: Path to the ground truth data
        
    Returns:
        Loaded ground truth dataset with area weights
    """
    groundtruth_rollout = xr.open_dataset(
        data_path,
        engine="zarr",
        chunks={},
    )
    groundtruth_rollout = groundtruth_rollout.sel(
        time=slice("2014-10-20", "2022-12-24")
    )  # These dates are not the eval dates, they are the dates from the rollout
    
    if "y" in groundtruth_rollout.coords:
        groundtruth_rollout = groundtruth_rollout.drop_vars(["lat", "lon"], errors="ignore")
        groundtruth_rollout = groundtruth_rollout.rename({"y": "lat", "x": "lon"})

    groundtruth_rollout = groundtruth_rollout.assign(
        areacello=(["lat", "lon"], spherical_area_weights(groundtruth_rollout))
    )
    
    return groundtruth_rollout


def load_basin_data(basin_path: str) -> xr.Dataset:
    """
    Load basin mask data.
    
    Args:
        basin_path: Path to the basin mask data
        
    Returns:
        Loaded basin dataset
    """
    return xr.open_dataset(basin_path)


def remove_climatology(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove seasonal climatology from the dataset.
    
    Args:
        ds: Input dataset
        
    Returns:
        Dataset with climatology removed
    """
    # Compute the climatology on the detrended data
    climatology = ds.groupby("time.dayofyear").mean("time").compute()

    # Remove the seasonal cycle (climatology) from the detrended data
    day_of_year = ds["time"].dt.dayofyear
    res = (ds - climatology.sel(dayofyear=day_of_year)).compute()

    return res