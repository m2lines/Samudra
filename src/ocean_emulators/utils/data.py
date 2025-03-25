import logging
from typing import Dict

import cftime
import numpy as np
import torch
import xarray as xr
from einops import rearrange

from ocean_emulators.config import TimeConfig
from ocean_emulators.constants import (
    DEPTH_I_LEVELS,
    DEPTH_LEVELS,
    MASK_VARS,
    BoundaryVarNames,
    Grid,
    GridMask,
    PrognosticMask,
    PrognosticVarNames,
    TensorMap,
)
from ocean_emulators.utils.multiton import Multiton


def extract_wet_mask(
    data: xr.Dataset, prognostic_var_names: PrognosticVarNames, hist: int
) -> tuple[PrognosticMask, GridMask]:
    """A mask for where the oceans are. Water is wet."""
    data_ = flatten_masks(data)
    wet_mask = data_[MASK_VARS]
    if "time" in wet_mask.dims:
        wet_mask_np = wet_mask.isel(time=0).to_array().to_numpy()
        wet_surface_mask_np = wet_mask[MASK_VARS[0]].isel(time=0).to_numpy()
    else:
        wet_mask_np = wet_mask.to_array().to_numpy()
        wet_surface_mask_np = wet_mask[MASK_VARS[0]].to_numpy()

    depth_ind = _parse_lev_from_output_var(prognostic_var_names)

    wet_inp = torch.from_numpy(wet_mask_np[depth_ind])
    wet_surface = torch.from_numpy(wet_surface_mask_np)
    wet_inp = torch.concat([wet_inp] * (hist + 1), dim=0)
    return wet_inp.bool(), wet_surface.bool()


def _parse_lev_from_output_var(prognostic_var_names: PrognosticVarNames) -> list[int]:
    """Parse the `lev` dimension from the output var names. Default: 0 for surface."""
    depth_inds = []
    for var_depth_i in prognostic_var_names:
        # Examples: "so_18", "zos"
        var_split = var_depth_i.split("_")
        if len(var_split) == 1:
            depth_inds.append(0)
        else:
            depth_inds.append(int(var_split[-1]))

    return depth_inds


def flatten_masks(data: xr.Dataset) -> xr.Dataset:
    """Adds data_vars "mask_0"..."mask_18" with dimensions (y, x)."""
    data_ = data.copy()
    if MASK_VARS[0] not in data_.variables:
        assert "wetmask" in data_.variables, (
            "Wet mask cannot be constructed without "
            "either the wetmask variable or the level-wise masks"
        )

        wet_mask = data_["wetmask"]
        for i, lev in enumerate(DEPTH_I_LEVELS):
            assert int(lev) == i, "Level indices must match the order of DEPTH_I_LEVELS"
            data_[f"mask_{lev}"] = wet_mask.isel(lev=i)

        data_ = data_.drop_vars("wetmask")

    return data_


def unflatten_masks(data: xr.Dataset) -> xr.Dataset:
    """Adds a "wetmask" `xarray.DataArray` with dimensions (lev, y, x)."""
    data_ = data.copy()
    if "wetmask" not in data_.variables:
        assert MASK_VARS[0] in data_.variables, "Wet mask must have masks as data vars!"

        wetmask = data_[MASK_VARS].to_array(dim="lev", name="wetmask")

        data_["wetmask"] = wetmask.assign_coords(lev=data_.lev)
        data_ = data_.drop_vars(MASK_VARS)

    return data_


def mask(data: xr.Dataset, wetmask: xr.DataArray) -> xr.Dataset:
    """Apply a wetmask (areas of the ocean) to all variables in the dataset."""
    # I revised on this via Project Pythia's tutorial:
    #  https://foundations.projectpythia.org/core/xarray/computation-masking.html#masking-data
    data_ = data.copy()

    wetmask = wetmask.astype(bool)
    surface_mask = wetmask.isel(lev=0)

    for name, da in data_.items():
        # Parse the level index info from the variable name.
        is_surface = False
        tokens = str(name).split("_")
        # If the name has four tokens, then it definitely is at some depth level (i.e.,
        # not at the surface).
        if len(tokens) >= 4:  # OM4 data format (e.g., {variable}_lev_{level}_{decimal})
            # TODO(alxmrs): Is the OM4 data wrong? Is preprocessing done somewhere?
            # In this format, "level" is a member of DEPTH_LEVELS, _not_ DEPTH_I_LEVELS.
            # Thus, we need to convert it to the corresponding DEPTH_I_LEVELS index.
            _, _, level, *_ = tokens
            closest_level = min(DEPTH_LEVELS, key=lambda x: abs(x - float(level)))
            level = DEPTH_I_LEVELS[DEPTH_LEVELS.index(closest_level)]
        # If it has two tokens, then it _maybe_ at the surface.
        elif len(tokens) == 2:  # output_vars format (e.g., {variable}_{level})
            _, level = tokens
            if level not in DEPTH_I_LEVELS:
                is_surface = True
        # With any other tokens, it _definitely_ is at the surface.
        else:
            is_surface = True

        if is_surface:
            # Assume this variable is at the surface
            # Apply the boundary layer mask and continue
            data_[name] = da.where(surface_mask, 0.0)
            continue

        assert level in DEPTH_I_LEVELS, f"Found unknown Depth Level! {level!r}."
        lev = DEPTH_LEVELS[int(level)]

        data_[name] = da.where(wetmask.sel(lev=lev), 0.0)

    return data_


def spherical_area_weights(data: xr.Dataset) -> Grid:
    num_lon = data.lon.size
    lats = torch.from_numpy(data.lat.to_numpy())
    weights = torch.cos(torch.deg2rad(lats)).repeat(num_lon, 1).t()
    weights /= weights.sum()
    return weights


def get_inference_steps(time_config: TimeConfig, time_delta: int = 5, hist: int = 1):
    """
    Get the number of inference/rollout steps for the given time configuration.

    Args:
        time_config: Time configuration
        time_delta: Time delta in days
        hist: Number of rollout steps

    Returns:
        num_steps: Number of rollout steps
    """
    start_time_str = time_config.start_time
    start_year, start_month, start_day = start_time_str.split("-")
    start_time = cftime.DatetimeNoLeap(
        int(start_year), int(start_month), int(start_day), 0, 0, 0
    )

    end_time_str = time_config.end_time
    end_year, end_month, end_day = end_time_str.split("-")
    end_time = cftime.DatetimeNoLeap(
        int(end_year), int(end_month), int(end_day), 0, 0, 0
    )
    num_steps = (end_time - start_time).days // time_delta + 1
    # Might have extra remaining days, so we remove them
    mod = num_steps % (hist + 1)
    num_steps = num_steps - mod
    return num_steps


def convert_tensor_out_to_dict(tensor_out: torch.Tensor) -> Dict[str, torch.Tensor]:
    tensor_map = TensorMap.get_instance()
    assert tensor_out.ndim == 5
    assert tensor_out.shape[2] == len(tensor_map.prognostic_var_names)
    out_dict = {}
    for i, var in enumerate(tensor_map.prognostic_var_names):
        out_dict[var] = tensor_out[:, :, i]
    return out_dict


def get_norm_unnorm_dicts(
    data: torch.Tensor,
    input_type: str = "target",
    num_prognostic_channels: int = 0,
    hist: int = 1,
):
    normalize = Normalize.get_instance()
    # Remove boundary data if input
    if input_type == "input":
        data = data[:, :num_prognostic_channels]

    # Separate history from channels
    data_reshaped = rearrange(data, "n (hi c) h w -> n hi c h w", hi=hist + 1)
    # Get normalized dict
    data_dict = convert_tensor_out_to_dict(data_reshaped)
    # Unnormalize
    data_unnorm = normalize.unnormalize_tensor_prognostic(data_reshaped)
    # Get unnormalized dict
    data_unnorm_dict = convert_tensor_out_to_dict(data_unnorm)
    return data_dict, data_unnorm_dict


def get_anomalies_vars(var_names: BoundaryVarNames) -> tuple[str, ...]:
    """Get the variables that need to be computed for anomalies."""
    return tuple([var for var in var_names if var.endswith("_anomalies")])


def compute_anomalies(data: xr.Dataset, anomalies_vars: tuple[str, ...]) -> xr.Dataset:
    """
    Compute anomalies for the given variables.
    """
    data_copy = data.copy()
    for var in anomalies_vars:
        base_var = var.replace("_anomalies", "")
        if var not in data_copy.variables and base_var in data_copy.variables:
            logging.info(f"Computing anomalies for {base_var}")
            climatology = (
                data_copy[base_var].groupby("time.dayofyear").mean("time").compute()
            )
            # Remove the seasonal cycle (climatology) from the detrended data
            day_of_year = data[base_var]["time"].dt.dayofyear
            data_copy[var] = (
                data_copy[base_var] - climatology.sel(dayofyear=day_of_year)
            ).compute()
    return data_copy


def rename_vars(data: xr.Dataset) -> xr.Dataset:
    """
    Rename variables if required.
    """
    data_copy = data.copy()
    for var in data.variables:
        # OM4 data format has variables in the form: var_lev_depthlevel
        # ex. so_lev_1040_0. We need to convert into var_depthlevelidx
        var_str = str(var)
        if "_lev_" in var_str:
            var_split = var_str.split("_lev_")
            var = var_split[0]
            lev_in_depth = float(var_split[1].replace("_", "."))
            lev_in_depth_idx = DEPTH_LEVELS.index(lev_in_depth)
            data_copy = data_copy.rename({var_str: var + "_" + str(lev_in_depth_idx)})
    return data_copy


def validate_data(
    data: xr.Dataset,
    data_mean: xr.Dataset,
    data_std: xr.Dataset,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Validate the data such that we have the correct format for training.
    """
    data_copy = data.copy()
    data_mean_copy = data_mean.copy()
    data_std_copy = data_std.copy()

    # Check if Mask variables exist
    if MASK_VARS[0] not in data_copy.variables:
        assert "wetmask" in data_copy.variables, (
            "Wet mask cannot be constructed without "
        )
        "either the wetmask variable or the level-wise masks"

        # Construct the mask variables
        wet_mask = data_copy["wetmask"]
        for i, lev in enumerate(DEPTH_I_LEVELS):
            assert int(lev) == i, "Level indices must match the order of DEPTH_I_LEVELS"
            data_copy[f"mask_{lev}"] = wet_mask.isel(lev=i)

        data_copy = data_copy.drop_vars("wetmask")

    # Check if data variables are in the right format
    # This check is to ensure we convert data to the correct format
    data_copy = rename_vars(data_copy)
    data_mean_copy = rename_vars(data_mean_copy)
    data_std_copy = rename_vars(data_std_copy)

    # OM4 data has coordinates we don't need
    # We drop them and rename x, y dimensions to lon, lat
    if "lat" not in data_copy.dims:
        # Drop unnecessary coordinates and rename dimensions
        data_copy = data_copy.drop_vars(
            ["lat", "lon", "lat_b", "lon_b", "dayofyear"], errors="ignore"
        ).rename({"x": "lon", "y": "lat"})

    # Check if any anomalies are needed to be computed
    tensor_map = TensorMap.get_instance()
    anomalies_vars = get_anomalies_vars(tensor_map.boundary_var_names)
    if anomalies_vars:
        data_copy = compute_anomalies(data_copy, anomalies_vars)

    return data_copy, data_mean_copy, data_std_copy


# TODO: Repetitive code. Refactor
class Normalize(Multiton):
    def _initialize(
        self,
        data_mean: xr.Dataset,
        data_std: xr.Dataset,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        wet_mask: torch.Tensor,
    ) -> None:
        """Store normalization parameters and pre-compute numpy arrays."""
        self.prognostic_mean = data_mean[prognostic_var_names]
        self.prognostic_std = data_std[prognostic_var_names]
        self.boundary_mean = data_mean[boundary_var_names]
        self.boundary_std = data_std[boundary_var_names]
        self.wet_mask = wet_mask

        # Pre-compute numpy arrays for faster access
        self._prognostic_mean_np = (
            self.prognostic_mean.to_array().to_numpy().reshape(-1)
        )
        self._prognostic_std_np = self.prognostic_std.to_array().to_numpy().reshape(-1)
        self._boundary_mean_np = self.boundary_mean.to_array().to_numpy().reshape(-1)
        self._boundary_std_np = self.boundary_std.to_array().to_numpy().reshape(-1)
        self._wet_mask_np = self.wet_mask.numpy()

    def _to_tensor(self, array: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert numpy array to tensor on specified device."""
        return torch.from_numpy(array).to(device)

    def normalize_prognostic(
        self, data: xr.Dataset, fill_nan=True, fill_value=0.0
    ) -> xr.Dataset:
        """Normalize input dataset."""
        norm = (data - self.prognostic_mean) / self.prognostic_std
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def normalize_boundary(
        self, data: xr.Dataset, fill_nan=True, fill_value=0.0
    ) -> xr.Dataset:
        """Normalize boundary conditions."""
        norm = (data - self.boundary_mean) / self.boundary_std
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def unnormalize_prognostic(self, data: xr.Dataset) -> xr.Dataset:
        """Unnormalize prognostic dataset."""
        data_unnorm = data * self.prognostic_std + self.prognostic_mean
        data_unnorm = data_unnorm * xr.DataArray(self._wet_mask_np)
        return data_unnorm

    def normalize_tensor_prognostic(
        self, data: torch.Tensor, fill_nan=True, fill_value=0.0
    ) -> torch.Tensor:
        """Normalize prognostic tensor."""
        tensor_mean = self._to_tensor(self._prognostic_mean_np, data.device)
        tensor_std = self._to_tensor(self._prognostic_std_np, data.device)
        if data.ndim == 4:
            tensor_mean = tensor_mean.reshape([1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, -1, 1, 1])
        elif data.ndim == 5:
            tensor_mean = tensor_mean.reshape([1, 1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, 1, -1, 1, 1])

        norm = (data - tensor_mean) / tensor_std
        if fill_nan:
            norm = norm.nan_to_num(nan=fill_value)
        return norm

    def unnormalize_tensor_prognostic(self, data: torch.Tensor) -> torch.Tensor:
        """Unnormalize prognostic tensor."""
        tensor_mean = self._to_tensor(self._prognostic_mean_np, data.device)
        tensor_std = self._to_tensor(self._prognostic_std_np, data.device)

        if data.ndim == 4:
            assert data.shape[1] == self._prognostic_mean_np.shape[0]
            tensor_mean = tensor_mean.reshape([1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, -1, 1, 1])
        elif data.ndim == 5:
            assert data.shape[2] == self._prognostic_mean_np.shape[0]
            tensor_mean = tensor_mean.reshape([1, 1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, 1, -1, 1, 1])
        else:
            raise ValueError(f"Invalid data shape: {data.shape}")

        unnorm = data * tensor_std + tensor_mean
        unnorm = unnorm * self.wet_mask.to(data.device)
        return unnorm

    def normalize_numpy_prognostic(
        self, data: np.ndarray, fill_nan=True, fill_value=0.0
    ) -> np.ndarray:
        """Normalize prognostic numpy array."""
        if data.ndim == 3:
            norm = (data - self._prognostic_mean_np) / self._prognostic_std_np
        elif data.ndim == 4:
            norm = (
                data - self._prognostic_mean_np.reshape(1, -1, 1, 1)
            ) / self._prognostic_std_np.reshape(1, -1, 1, 1)
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def unnormalize_numpy_prognostic(self, data: np.ndarray) -> np.ndarray:
        """Unnormalize prognostic numpy array."""
        data_unnorm = data * self._prognostic_std_np + self._prognostic_mean_np
        data_unnorm = data_unnorm * self._wet_mask_np
        return data_unnorm
