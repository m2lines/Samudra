import dataclasses
import logging
from typing import Any, Callable, Literal, Self

import cftime
import numpy as np
import torch
import xarray as xr
from einops import rearrange

from ocean_emulators.config import EvalConfig, TimeConfig, TrainConfig
from ocean_emulators.constants import (
    DEPTH_I_LEVELS,
    DEPTH_LEVELS,
    MASK_VARS,
    BatchTimeSeriesOutput,
    BoundaryVarNames,
    DictSingleChannelVar,
    Grid,
    GridMask,
    Input,
    LoaderVersion,
    Prognostic,
    PrognosticMask,
    PrognosticVarNames,
    SingleTimeSeriesOutput,
    TensorMap,
)
from ocean_emulators.utils.multiton import Multiton


@dataclasses.dataclass
class DataSource:
    """Data source for the model."""

    name: str
    data: xr.Dataset
    means: xr.Dataset
    stds: xr.Dataset

    def filter(
        self,
        var_names: PrognosticVarNames | BoundaryVarNames,
        *,
        name: str | None = None,
    ) -> Self:
        """Filter the data source to only include the specified variables."""
        data = self.data[var_names]
        means = self.means[var_names]
        stds = self.stds[var_names]

        return dataclasses.replace(
            self, name=name or self.name, data=data, means=means, stds=stds
        )

    def map(
        self,
        func: Callable[
            [xr.Dataset, xr.Dataset, xr.Dataset],
            tuple[xr.Dataset, xr.Dataset, xr.Dataset],
        ],
        *,
        name: str | None = None,
    ) -> Self:
        """Map the function over the data source."""
        data, means, stds = func(self.data.copy(), self.means.copy(), self.stds.copy())

        return dataclasses.replace(
            self, name=name or self.name, data=data, means=means, stds=stds
        )

    def map_data(
        self, func: Callable[[xr.Dataset], xr.Dataset], *, name: str | None = None
    ) -> Self:
        """Map the function over just data in DataSource."""
        data = func(self.data.copy())
        return dataclasses.replace(self, name=name or self.name, data=data)

    def slice(self, time_slice: slice, *, name: str | None = None) -> Self:
        """Slice the data source to only include the specified time slice."""
        data = self.data.sel(time=time_slice)

        return dataclasses.replace(self, name=name or self.name, data=data)

    def normalize(
        self, data: xr.Dataset | None = None, fill_nan=True, fill_value=0.0
    ) -> xr.Dataset:
        """Normalize input data."""
        norm = ((data or self.data) - self.means) / self.stds
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def normalize_tensor(
        self,
        data: torch.Tensor | None = None,
        variable_axis: int = 0,
        fill_nan=True,
        fill_value=0.0,
    ) -> torch.Tensor:
        """Normalize input data treated as torch Tensors."""
        device = (data or self.data).device
        reshape_vars = [1] * (data or self.data).ndim
        reshape_vars[variable_axis] = -1

        if data is None:
            data_np = self.data.to_array().to_numpy().reshape(-1)
            data = torch.from_numpy(data_np).to(device).reshape(reshape_vars)

        # TODO(alxmrs): Do we have to reshape twice?
        means_np = self.means.to_array().to_numpy().reshape(-1)
        stds_np = self.stds.to_array().to_numpy().reshape(-1)
        means = torch.from_numpy(means_np).to(device).reshape(reshape_vars)
        stds = torch.from_numpy(stds_np).to(device).reshape(reshape_vars)

        norm = (data - means) / stds
        if fill_nan:
            norm = norm.nan_to_num(nan=fill_value)
        norm = norm.to(data.dtype)
        return norm

    @classmethod
    def from_config(
        cls, cfg: TrainConfig | EvalConfig, *, use_dask: bool | None = None
    ) -> Self:
        if use_dask is None:
            use_dask = cfg.data.loader_version != LoaderVersion.OM4_TORCH.value
        chunks: dict[str, int] | None = {} if use_dask else None

        root = cfg.experiment.data_dir

        if "*" in cfg.data.data_path:
            kwargs: dict[str, Any] = dict(
                engine="netcdf4", chunks={"time": 1, "lat": 180, "lon": 360}
            )
        else:
            kwargs = dict(chunks=chunks, consolidated=True)
        data = xr.open_dataset(root / cfg.data.data_path, **kwargs)

        means = xr.open_dataset(
            root / cfg.data.data_means_path,
            engine="netcdf4" if cfg.data.data_means_path.endswith(".nc") else "zarr",
            chunks=chunks,
        )
        stds = xr.open_dataset(
            root / cfg.data.data_stds_path,
            engine="netcdf4" if cfg.data.data_stds_path.endswith(".nc") else "zarr",
            chunks=chunks,
        )

        dask = "with_dask" if use_dask else "without_dask"

        return cls(
            name=f"raw-{cfg.experiment.name}-{cfg.experiment.data_dir.name}-{dask}",
            data=data,
            means=means,
            stds=stds,
        )


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


def convert_tensor_out_to_dict(tensor_out: torch.Tensor) -> DictSingleChannelVar:
    tensor_map = TensorMap.get_instance()
    assert tensor_out.ndim == 5
    assert tensor_out.shape[2] == len(tensor_map.prognostic_var_names)
    out_dict = {}
    for i, var in enumerate(tensor_map.prognostic_var_names):
        out_dict[var] = tensor_out[:, :, i]
    return out_dict


def get_aggregator_dicts(
    data: Prognostic | Input,
    wet: torch.Tensor,
    long_rollout: bool,
    input_type: Literal["prognostic", "input"] = "prognostic",
    num_prognostic_channels: int = 0,
    hist: int = 1,
) -> tuple[DictSingleChannelVar, DictSingleChannelVar]:
    normalize = Normalize.get_instance()
    # Remove boundary data if input
    if input_type == "input":
        data = data[:, :num_prognostic_channels]

    # Separate history from channels
    data_reshaped: SingleTimeSeriesOutput | BatchTimeSeriesOutput
    if long_rollout:
        # All batches are part of the same rollout during inference
        data_reshaped = rearrange(
            data, "n (hi c) h w -> (n hi) c h w", hi=hist + 1
        ).unsqueeze(0)  # add artificial batch dim
    else:
        # Batches are independent rollouts during validation
        data_reshaped = rearrange(data, "n (hi c) h w -> n hi c h w", hi=hist + 1)

    # Get normalized dict
    data_normalized = data_reshaped.clone()
    data_normalized = torch.where(wet == 0, float("nan"), data_normalized)
    data_dict = convert_tensor_out_to_dict(data_normalized)
    # Unnormalize
    data_unnorm = normalize.unnormalize_tensor_prognostic(
        data_reshaped, fill_value=float("nan")
    )
    # Get unnormalized dict
    data_unnorm_dict = convert_tensor_out_to_dict(data_unnorm)
    return data_dict, data_unnorm_dict


def get_anomalies_vars(var_names: BoundaryVarNames) -> tuple[str, ...]:
    """Get the variables that need to be computed for anomalies."""
    return tuple([var for var in var_names if var.endswith("_anomalies")])


def compute_anomalies(
    data_src: DataSource, anomalies_vars: tuple[str, ...]
) -> DataSource:
    """Compute anomalies for the given variables."""

    def _anom(data, means, stds):
        for var in anomalies_vars:
            base_var = var.replace("_anomalies", "")
            if var not in data.variables and base_var in data.variables:
                logging.info(f"Computing anomalies for {base_var}")
                climatology = (
                    data[base_var].groupby("time.dayofyear").mean("time").compute()
                )
                # Remove the seasonal cycle (climatology) from the detrended data
                day_of_year = data[base_var]["time"].dt.dayofyear
                data[var] = (
                    data[base_var] - climatology.sel(dayofyear=day_of_year)
                ).compute()
                data = data.drop(["dayofyear"])
                means[var] = data[var].mean().compute()
                stds[var] = data[var].std().compute()
        return data, means, stds

    return data_src.map(_anom, name=data_src.name + "_anomalies")


def with_level_index_vars(data: xr.Dataset) -> xr.Dataset:
    """
    Ensure variable names use a depth level index, not depth level value.
    """
    data_copy = data.copy()

    for var in data.variables:
        # OM4 data format has variables in the form: var_lev_{depthlevel}
        # ex. so_lev_1040_0. We need to convert into var_{depthlevelidx}
        var_str = str(var)
        if "_lev_" in var_str:
            var_split = var_str.split("_lev_")
            var = var_split[0]
            lev_in_depth = float(var_split[1].replace("_", "."))
            lev_in_depth_idx = DEPTH_LEVELS.index(lev_in_depth)
            data_copy = data_copy.rename({var_str: f"{var}_{lev_in_depth_idx!s}"})

    return data_copy


def with_lat_lon_coords(data: xr.Dataset) -> xr.Dataset:
    """Standardize dataset coordinates; prefer "lat"/"lon" over "y"/"x"."""
    data_copy = data.copy()
    # OM4 data has coordinates we don't need
    # We drop them and rename x, y dimensions to lon, lat
    if "lat" not in data_copy.dims:
        # Drop unnecessary coordinates and rename dimensions
        data_copy = data_copy.drop_vars(
            ["lat", "lon", "lat_b", "lon_b", "dayofyear"], errors="ignore"
        ).rename({"x": "lon", "y": "lat"})

    return data_copy


def validate_data(src: DataSource) -> DataSource:
    """Validate the data such that we have the correct format for training."""

    def _rename(data, means, stds):
        data = (
            data.pipe(flatten_masks)
            .pipe(with_level_index_vars)
            .pipe(with_lat_lon_coords)
        )

        # Check if data variables are in the right format
        # This check is to ensure we convert data to the correct format
        means = with_level_index_vars(means)
        stds = with_level_index_vars(stds)
        return data, means, stds

    src_ = src.map(_rename, name=src.name + "_validated")

    # Check if any anomalies are needed to be computed
    tensor_map = TensorMap.get_instance()
    anomalies_vars = get_anomalies_vars(tensor_map.boundary_var_names)

    out = compute_anomalies(src_, anomalies_vars) if anomalies_vars else src_

    return out


# TODO: Repetitive code. Refactor
class Normalize(Multiton):
    def _initialize(
        self,
        src: DataSource,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        wet_mask: torch.Tensor,
    ) -> None:
        """Store normalization parameters and pre-compute numpy arrays."""
        prog_src = src.filter(prognostic_var_names)
        bound_src = src.filter(boundary_var_names)
        self.prognostic_mean = prog_src.means
        self.prognostic_std = prog_src.stds
        self.boundary_mean = bound_src.means
        self.boundary_std = bound_src.stds
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
        norm = norm.to(data.dtype)
        return norm

    def unnormalize_tensor_prognostic(
        self, data: torch.Tensor, fill_value=float("nan")
    ) -> torch.Tensor:
        """Unnormalize prognostic tensor and apply fill value to land cells."""
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
        unnorm = torch.where(self.wet_mask.to(data.device) == 0, fill_value, unnorm)
        unnorm = unnorm.to(data.dtype)
        return unnorm
