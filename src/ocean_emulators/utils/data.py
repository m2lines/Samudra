from typing import Dict

import cftime
import numpy as np
import torch
import xarray as xr
from einops import rearrange

from ocean_emulators import config
from ocean_emulators.constants import (
    MASK_VARS,
    BoundaryVarsStr,
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
    wet_mask = data[MASK_VARS]
    if "time" in wet_mask.dims:
        wet_mask_np = wet_mask.isel(time=0).to_array().to_numpy()
        wet_surface_mask_np = wet_mask[MASK_VARS[0]].isel(time=0).to_numpy()
    else:
        wet_mask_np = wet_mask.to_array().to_numpy()
        wet_surface_mask_np = wet_mask[MASK_VARS[0]].to_numpy()

    depth_ind = []
    for var_depth_i in prognostic_var_names:
        var_split = var_depth_i.split("_")
        if len(var_split) == 1:
            depth_ind.append(0)
        else:
            depth_ind.append(int(var_split[-1]))

    wet_inp = torch.from_numpy(wet_mask_np[depth_ind])
    wet_surface = torch.from_numpy(wet_surface_mask_np)
    wet_inp = torch.concat([wet_inp] * (hist + 1), dim=0)
    return wet_inp.bool(), wet_surface.bool()


def spherical_area_weights(data: xr.Dataset) -> Grid:
    num_lon = data.lon.size
    lats = torch.from_numpy(data.lat.to_numpy())
    weights = torch.cos(torch.deg2rad(lats)).repeat(num_lon, 1).t()
    weights /= weights.sum()
    return weights


def get_time_slice(
    time_config: config.TimeConfig, time_delta: int = 5, hist: int = 1
) -> tuple[slice, int]:
    """
    Get the time slice and number of rollout steps for the given time configuration.

    The slice includes both the start and end times. There is option to include
    the initial condition but num_steps does not include the initial condition.

    Args:
        time_config: Time configuration
        initial_cond: Whether to include the initial condition
        time_delta: Time delta in days
        hist: Number of rollout steps

    Returns:
        slice: Time slice
        num_steps: Number of rollout steps (not including initial condition)
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
    return slice(start_time, end_time), num_steps


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


# TODO: Repetitive code. Refactor
class Normalize(Multiton):
    def _initialize(
        self,
        data_mean: xr.Dataset,
        data_std: xr.Dataset,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarsStr,
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
