from typing import Dict, Optional

import cftime
import numpy as np
import torch
import xarray as xr
from einops import rearrange

import config
from constants import (
    DEPTH_I_LEVELS,
    DEPTH_LEVELS,
    MASK_VARS,
    ExtraVars,
    Grid,
    GridMask,
    InputMask,
    InputVars,
    OutputVars,
    TensorMap,
)


def extract_wet_mask(
    data: xr.Dataset, outputs: OutputVars, hist: int
) -> tuple[InputMask, GridMask]:
    """A mask for where the oceans are. Water is wet."""
    wet_mask = data[MASK_VARS]
    if "time" in wet_mask.dims:
        wet_mask_np = wet_mask.isel(time=0).to_array().to_numpy()
        wet_surface_mask_np = wet_mask[MASK_VARS[0]].isel(time=0).to_numpy()
    else:
        wet_mask_np = wet_mask.to_array().to_numpy()
        wet_surface_mask_np = wet_mask[MASK_VARS[0]].to_numpy()

    depth_ind = _parse_lev_from_output_var(outputs)

    wet_inp = torch.from_numpy(wet_mask_np[depth_ind])
    wet_surface = torch.from_numpy(wet_surface_mask_np)
    wet_inp = torch.concat([wet_inp] * (hist + 1), dim=0)
    return wet_inp.bool(), wet_surface.bool()


def _parse_lev_from_output_var(outputs: OutputVars) -> list[int]:
    """Parse the `lev` dimension from the output var names. Default: 0 for surface."""
    depth_inds = []
    for var_depth_i in outputs:
        # Examples: "so_18", "zos"
        var_split = var_depth_i.split("_")
        if len(var_split) == 1:
            depth_inds.append(0)
        else:
            depth_inds.append(int(var_split[-1]))

    return depth_inds


def flatten_masks(data: xr.Dataset) -> xr.Dataset:
    """Adds data_vars "mask_0"..."mask_18" with dimensions (y, x)."""
    if MASK_VARS[0] not in data.variables:
        assert "wetmask" in data.variables, "Wet mask cannot be constructed without "
        "either the wetmask variable or the level-wise masks"

        wet_mask = data["wetmask"]
        for i, lev in enumerate(DEPTH_I_LEVELS):
            assert int(lev) == i, "Level indices must match the order of DEPTH_I_LEVELS"
            data[f"mask_{lev}"] = wet_mask.isel(lev=i)

        data = data.drop_vars("wetmask")

    return data


def unflatten_masks(data: xr.Dataset) -> xr.Dataset:
    """Adds a "wetmask" `xarray.DataArray` with dimensions (lev, y, x)."""
    if "wetmask" not in data.variables:
        assert MASK_VARS[0] in data.variables, "Wet mask must have masks as data vars!"

        wetmask = data[MASK_VARS].to_array(dim="lev", name="wetmask")
        wetmask.assign_coords(lev=data.lev)

        data["wetmask"] = wetmask
        data = data.drop_vars(MASK_VARS)

    return data


def mask(data: xr.Dataset) -> xr.Dataset:
    """Applies the wetmask to the data up-front."""
    # I revised on this via Project Pythia's tutorial:
    #  https://foundations.projectpythia.org/core/xarray/computation-masking.html#masking-data
    data_ = data.copy()
    wetmask = data_.wetmask.astype(bool)
    surface_mask = wetmask.isel(lev=0)

    for name, da in data_.items():
        try:
            variable, _, level, _ = name.split("_")
        except ValueError:
            # Assume this variable is at the surface.
            # Apply the boundary layer mask and continue on.
            data_[name] = da.where(surface_mask)
            continue

        # The string encoding is... not the best. For example, it doesn't
        # include decimal numbers. So, we set `lev` to be the closet value
        # to the whole list of DEPTH_LEVELS.
        lev = float(level)
        lev = min(DEPTH_LEVELS, key=lambda m: abs(m - lev))
        assert lev in DEPTH_LEVELS, f"Found unknown Depth Level! {lev}."

        data_[name] = da.where(wetmask.sel(lev=lev))

    return data_


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
        inputs_str: InputVars,
        extra_in_str: ExtraVars,
        outputs_str: OutputVars,
        wet_mask: torch.Tensor,
    ) -> "Normalize":
        """Initialize the singleton instance with normalization parameters."""
        if cls._instance is not None:
            raise ValueError("Normalize already initialized")

        instance = super().__new__(cls)
        instance._initialize(
            data_mean, data_std, inputs_str, extra_in_str, outputs_str, wet_mask
        )
        cls._instance = instance
        return cls._instance

    def _initialize(
        self,
        data_mean: xr.Dataset,
        data_std: xr.Dataset,
        inputs_str: InputVars,
        extra_in_str: ExtraVars,
        outputs_str: OutputVars,
        wet_mask: torch.Tensor,
    ) -> None:
        """Store normalization parameters and pre-compute numpy arrays."""
        self.inputs_mean = data_mean[inputs_str]
        self.inputs_std = data_std[inputs_str]
        self.extras_mean = data_mean[extra_in_str]
        self.extras_std = data_std[extra_in_str]
        self.outputs_mean = data_mean[outputs_str]
        self.outputs_std = data_std[outputs_str]
        self.wet_mask = wet_mask

        # Pre-compute numpy arrays for faster access
        self._inputs_mean_np = self.inputs_mean.to_array().to_numpy().reshape(-1)
        self._inputs_std_np = self.inputs_std.to_array().to_numpy().reshape(-1)
        self._outputs_mean_np = self.outputs_mean.to_array().to_numpy().reshape(-1)
        self._outputs_std_np = self.outputs_std.to_array().to_numpy().reshape(-1)
        self._wet_mask_np = self.wet_mask.numpy()

    def _to_tensor(self, array: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert numpy array to tensor on specified device."""
        return torch.from_numpy(array).to(device)

    def normalize_inputs(
        self, data: xr.Dataset, fill_nan=True, fill_value=0.0
    ) -> xr.Dataset:
        """Normalize input dataset."""
        norm = (data - self.inputs_mean) / self.inputs_std
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def normalize_boundary(
        self, data: xr.Dataset, fill_nan=True, fill_value=0.0
    ) -> xr.Dataset:
        """Normalize boundary conditions."""
        norm = (data - self.extras_mean) / self.extras_std
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def normalize_outputs(
        self, data: xr.Dataset, fill_nan=True, fill_value=0.0
    ) -> xr.Dataset:
        """Normalize output dataset."""
        norm = (data - self.outputs_mean) / self.outputs_std
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def unnormalize_outputs(self, data: xr.Dataset) -> xr.Dataset:
        """Unnormalize output dataset."""
        data_unnorm = data * self.outputs_std + self.outputs_mean
        data_unnorm = data_unnorm * xr.DataArray(self._wet_mask_np)
        return data_unnorm

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
            assert data.shape[1] == self._outputs_mean_np.shape[0]
            tensor_mean = tensor_mean.reshape([1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, -1, 1, 1])
        elif data.ndim == 5:
            assert data.shape[2] == self._outputs_mean_np.shape[0]
            tensor_mean = tensor_mean.reshape([1, 1, -1, 1, 1])
            tensor_std = tensor_std.reshape([1, 1, -1, 1, 1])
        else:
            raise ValueError(f"Invalid data shape: {data.shape}")

        unnorm = data * tensor_std + tensor_mean
        unnorm = unnorm * self.wet_mask.to(data.device)
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
        data_unnorm = data * self._outputs_std_np + self._outputs_mean_np
        data_unnorm = data_unnorm * self._wet_mask_np
        return data_unnorm
