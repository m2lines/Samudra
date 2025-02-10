from typing import Dict, Optional

import cftime
import numpy as np
import torch
import xarray as xr
from einops import rearrange

from constants import DEPTH_LEVELS, TensorMap


def extract_wet_mask(wet_zarr, outputs, hist):
    depth_ind = []
    for var_depth_i in outputs:
        ind = var_depth_i.split("_")[-1]
        if ind == "zos":
            depth_ind.append("0")
        else:
            depth_ind.append(ind)
    depths = [DEPTH_LEVELS[int(depth_i)] for depth_i in depth_ind]
    wet = wet_zarr.sel(lev=depths)
    wet = torch.from_numpy(wet.to_array().to_numpy().squeeze())
    wet = torch.concat([wet] * (hist + 1), dim=0)
    return wet


def get_time_slice(time_config, time_delta=5, hist=1):
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
    assert tensor_out.shape[2] == len(tensor_map.outputs)
    out_dict = {}
    for i, var in enumerate(tensor_map.outputs):
        out_dict[var] = tensor_out[:, :, i]
    return out_dict


def get_norm_unnorm_dicts(
    data: torch.Tensor,
    input_type: str = "target",
    output_channels: int = 0,
    hist: int = 1,
):
    normalize = Normalize.get_instance()
    # Remove boundary data if input
    if input_type == "input":
        data = data[:, :output_channels]

    # Separate history from channels
    data_reshaped = rearrange(data, "n (hi c) h w -> n hi c h w", hi=hist + 1)
    # Get normalized dict
    data_dict = convert_tensor_out_to_dict(data_reshaped)
    # Unnormalize
    data_unnorm = normalize.unnormalize_tensor_outputs(data_reshaped)
    # Get unnormalized dict
    data_unnorm_dict = convert_tensor_out_to_dict(data_unnorm)
    return data_dict, data_unnorm_dict


# TODO: Repetitive code. Refactor
class Normalize:
    _instance: Optional["Normalize"] = None

    def __new__(cls, *args, **kwargs) -> "Normalize":
        # Prevent direct instantiation
        raise TypeError(
            "Normalize cannot be instantiated directly. Use init_instance() instead."
        )

    @classmethod
    def get_instance(cls) -> "Normalize":
        if cls._instance is None:
            raise ValueError("Normalize not initialized")
        return cls._instance

    @classmethod
    def init_instance(
        cls,
        data_mean: xr.Dataset,
        data_std: xr.Dataset,
        inputs_str: str,
        extra_in_str: str,
        outputs_str: str,
    ) -> "Normalize":
        """Initialize the singleton instance with normalization parameters."""
        if cls._instance is not None:
            raise ValueError("Normalize already initialized")

        instance = super().__new__(cls)
        instance._initialize(data_mean, data_std, inputs_str, extra_in_str, outputs_str)
        cls._instance = instance
        return cls._instance

    def _initialize(
        self,
        data_mean: xr.Dataset,
        data_std: xr.Dataset,
        inputs_str: str,
        extra_in_str: str,
        outputs_str: str,
    ) -> None:
        """Store normalization parameters and pre-compute numpy arrays."""
        self.inputs_mean = data_mean[inputs_str]
        self.inputs_std = data_std[inputs_str]
        self.extras_mean = data_mean[extra_in_str]
        self.extras_std = data_std[extra_in_str]
        self.outputs_mean = data_mean[outputs_str]
        self.outputs_std = data_std[outputs_str]

        # Pre-compute numpy arrays for faster access
        self._inputs_mean_np = self.inputs_mean.to_array().to_numpy().reshape(-1)
        self._inputs_std_np = self.inputs_std.to_array().to_numpy().reshape(-1)
        self._outputs_mean_np = self.outputs_mean.to_array().to_numpy().reshape(-1)
        self._outputs_std_np = self.outputs_std.to_array().to_numpy().reshape(-1)

    def _to_tensor(self, array: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert numpy array to tensor on specified device."""
        return torch.from_numpy(array).to(device)

    def normalize_inputs(
        self, data: xr.Dataset, fill_nan=True, fill_value=0.0
    ) -> xr.Dataset:
        """Normalize input dataset."""
        norm = ((data - self.inputs_mean) / self.inputs_std).fillna(0)
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def normalize_boundary(
        self, data: xr.Dataset, fill_nan=True, fill_value=0.0
    ) -> xr.Dataset:
        """Normalize boundary conditions."""
        norm = ((data - self.extras_mean) / self.extras_std).fillna(0)
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def normalize_outputs(
        self, data: xr.Dataset, fill_nan=True, fill_value=0.0
    ) -> xr.Dataset:
        """Normalize output dataset."""
        norm = ((data - self.outputs_mean) / self.outputs_std).fillna(0)
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def unnormalize_outputs(self, data: xr.Dataset) -> xr.Dataset:
        """Unnormalize output dataset."""
        return data * self.outputs_std + self.outputs_mean

    def normalize_tensor_inputs(
        self, data: torch.Tensor, fill_nan=True, fill_value=0.0
    ) -> torch.Tensor:
        """Normalize input tensor."""
        tensor_mean = self._to_tensor(self._inputs_mean_np, data.device)
        tensor_std = self._to_tensor(self._inputs_std_np, data.device)
        norm = (data - tensor_mean) / tensor_std
        if fill_nan:
            norm = norm.nan_to_num(nan=fill_value)
        return norm

    def normalize_tensor_outputs(
        self, data: torch.Tensor, fill_nan=True, fill_value=0.0
    ) -> torch.Tensor:
        """Normalize output tensor."""
        tensor_mean = self._to_tensor(self._outputs_mean_np, data.device)
        tensor_std = self._to_tensor(self._outputs_std_np, data.device)
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

    def unnormalize_tensor_outputs(self, data: torch.Tensor) -> torch.Tensor:
        """Unnormalize output tensor."""
        tensor_mean = self._to_tensor(self._outputs_mean_np, data.device)
        tensor_std = self._to_tensor(self._outputs_std_np, data.device)

        if data.ndim == 4:
            assert (
                data.shape[1] == self._outputs_mean_np.shape[0]
            ), f"{data.shape[1]} != {self._outputs_mean_np.shape[0]}"
            tensor_mean = tensor_mean.reshape([1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, -1, 1, 1])
        elif data.ndim == 5:
            assert data.shape[2] == self._outputs_mean_np.shape[0]
            tensor_mean = tensor_mean.reshape([1, 1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, 1, -1, 1, 1])
        else:
            raise ValueError(f"Invalid data shape: {data.shape}")

        unnorm = data * tensor_std + tensor_mean
        return unnorm

    def normalize_numpy_inputs(
        self, data: np.ndarray, fill_nan=True, fill_value=0.0
    ) -> np.ndarray:
        """Normalize input numpy array."""
        norm = (data - self._inputs_mean_np) / self._inputs_std_np
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def normalize_numpy_outputs(
        self, data: np.ndarray, fill_nan=True, fill_value=0.0
    ) -> np.ndarray:
        """Normalize output numpy array."""
        if data.ndim == 3:
            norm = (data - self._outputs_mean_np) / self._outputs_std_np
        elif data.ndim == 4:
            norm = (
                data - self._outputs_mean_np.reshape(1, -1, 1, 1)
            ) / self._outputs_std_np.reshape(1, -1, 1, 1)
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def unnormalize_numpy_outputs(self, data: np.ndarray) -> np.ndarray:
        """Unnormalize output numpy array."""
        return data * self._outputs_std_np + self._outputs_mean_np
