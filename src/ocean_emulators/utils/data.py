import dataclasses
import logging
from collections.abc import Callable
from functools import cached_property
from typing import Literal, Self

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
    TIME_DELTA,
    BatchTimeSeriesOutput,
    BoundaryVarNames,
    DictSingleChannelVar,
    Grid,
    GridMask,
    Input,
    Prognostic,
    PrognosticMask,
    PrognosticVarNames,
    SingleTimeSeriesOutput,
    TensorMap,
)
from ocean_emulators.derived_variables import add_derived_variables
from ocean_emulators.utils.multiton import Multiton


@dataclasses.dataclass
class DataSource:
    """Data source for the model."""

    name: str
    data: xr.Dataset
    means: xr.Dataset
    stds: xr.Dataset

    @cached_property
    def is_compact(self) -> bool:
        """Check if the data source is compact."""
        return all(
            "_" not in str(v)
            for d in [self.data, self.means, self.stds]
            for v in d.keys()
            if "anom" not in str(v)
        )

    def filter(
        self,
        var_names: PrognosticVarNames | BoundaryVarNames,
        *,
        prefix: str,
    ) -> Self:
        """Filter the data source to only include the specified variables (and levels).

        If the dataset is compact, it will also filter the levels based on the
        variable names (which encode the level in the name).

        Args:
            var_names: Variable names to filter.
            prefix: Prefix for the new data source name.

        Returns:
            A new `DataSource` only with the filtered variables and levels.
        """
        name = f"{prefix}[{self.name}]"
        if self.is_compact:
            parsed_var_names, levels = [], []
            for mangled_var_name in var_names:
                if "_" not in mangled_var_name:
                    parsed_var_names.append(mangled_var_name)
                    continue
                tokens = mangled_var_name.split("_")
                var_name, level = tokens[0], int(tokens[1])

                parsed_var_names.append(var_name)
                # Build set of total levels
                if level not in levels:
                    levels.append(level)

            data = self.data[parsed_var_names]
            means = self.means[parsed_var_names]
            stds = self.stds[parsed_var_names]
            if levels:
                data = data.isel(lev=levels)
                means = means.isel(lev=levels)
                stds = stds.isel(lev=levels)

            return dataclasses.replace(
                self, name=name, data=data, means=means, stds=stds
            )

        data = self.data[var_names]
        means = self.means[var_names]
        stds = self.stds[var_names]

        return dataclasses.replace(self, name=name, data=data, means=means, stds=stds)

    def map(
        self,
        func: Callable[
            [xr.Dataset, xr.Dataset, xr.Dataset],
            tuple[xr.Dataset, xr.Dataset, xr.Dataset],
        ],
        *,
        suffix: str | None = None,
    ) -> Self:
        """Map the function over the data source."""
        if suffix is None:
            suffix = func.__qualname__

        data, means, stds = func(self.data.copy(), self.means.copy(), self.stds.copy())

        return dataclasses.replace(
            self, name=f"{self.name}_{suffix}", data=data, means=means, stds=stds
        )

    def map_data(
        self, func: Callable[[xr.Dataset], xr.Dataset], *, suffix: str | None = None
    ) -> Self:
        """Map the function over just data in DataSource."""
        if suffix is None:
            suffix = func.__qualname__
        data = func(self.data.copy())
        return dataclasses.replace(self, name=f"{self.name}_{suffix}", data=data)

    def slice(self, time: TimeConfig) -> Self:
        """Slice the data source to only include the specified time slice."""
        data = self.data.sel(time=time.time_slice)
        return dataclasses.replace(self, name=f"{time=}[{self.name}]", data=data)

    def normalize(self, fill_nan=True, fill_value=0.0) -> xr.Dataset:
        """Normalize input data."""
        norm = (self.data - self.means) / self.stds
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    def normalize_with(
        self,
        data: torch.Tensor,
        variable_axis: int = 0,
        fill_nan=True,
        fill_value=0.0,
    ) -> torch.Tensor:
        """Normalize input data treated as torch Tensors."""
        reshape_vars = [1] * data.ndim
        reshape_vars[variable_axis] = -1

        # TODO(alxmrs): Do we have to reshape twice?
        if "lev" in self.means.dims:
            means_np = (
                conditional_rearrange(
                    self.means,
                    "(variable lev)=var",
                    concat_dim="var",
                )
                .rename({"var": "variable"})
                .to_numpy()
                .reshape(-1)
            )
        else:
            means_np = self.means.to_array().to_numpy().reshape(-1)
        if "lev" in self.stds.dims:
            stds_np = (
                conditional_rearrange(
                    self.stds,
                    "(variable lev)=var",
                    concat_dim="var",
                )
                .rename({"var": "variable"})
                .to_numpy()
                .reshape(-1)
            )
        else:
            stds_np = self.stds.to_array().to_numpy().reshape(-1)

        means = torch.from_numpy(means_np).reshape(reshape_vars)
        stds = torch.from_numpy(stds_np).reshape(reshape_vars)

        norm = (data - means) / stds
        if fill_nan:
            norm = norm.nan_to_num(nan=fill_value)
        norm = norm.to(data.dtype)
        return norm

    @classmethod
    def from_config(cls, cfg: TrainConfig | EvalConfig, *, use_dask: bool) -> Self:
        chunks: dict[str, int] | None = {} if use_dask else None

        root = cfg.experiment.data_dir

        if "*" in cfg.data.data_path:
            data = xr.open_dataset(
                root / cfg.data.data_path,
                engine="netcdf4",
                chunks={"time": 1, "lat": 180, "lon": 360},
            )
        else:
            data = xr.open_dataset(
                root / cfg.data.data_path,
                chunks=chunks,
                consolidated=True,
                engine="zarr",
            )

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
            name=f"{cfg.experiment.name}-{cfg.experiment.data_dir.name}-{dask}",
            data=data,
            means=means,
            stds=stds,
        )


def conditional_rearrange(
    data: xr.Dataset, pattern: str, except_dim="lev", concat_dim="variable"
) -> xr.DataArray:
    """Rearrange a Dataset using an einsum notation with and without a dimension.

    When a dataset has variables with a mixture of dimensions and an einsum-like
    rearrange is applied on that dataset, it's common that the pattern will combinate
    one too many variables. Sometimes, it's desirable to apply the rearrange pattern
    on two versions of the data: one including variables with that dimension and one
    without, and then concatenate them along a new dimension.

    For example, surface level boundary variables, which only occur at t0, should not be
    combinatorially rearranged with depth variables that have multiple time steps. In
    such a situation, this function can be used to apply a standard einsum rearrangement
    to depth and surface variables, including and excluding variables who have a `time`
    dimension, respectively.

    Args:
        data: The dataset to rearrange.
        pattern: The einsum pattern to use for rearranging.
        except_dim: The dimension to exclude from the pattern.
        concat_dim: The dimension to concatenate along.

    Returns:
        The combined, rearranged dataset as a `xarray.DataArray`.
    """
    assert except_dim in pattern, f"{except_dim} must be in the pattern."

    vars_with_dim = [v for v in data if except_dim in data[v].dims]
    vars_without_dim = [v for v in data if except_dim not in data[v].dims]

    data_with_dim = (
        data[vars_with_dim]
        .to_array()
        .einops.rearrange(pattern, dask="allowed")
        .drop_vars(concat_dim, errors="ignore")
    )
    data_without_dim = (
        data[vars_without_dim]
        .to_array()
        .einops.rearrange(pattern.replace(except_dim, ""), dask="allowed")
        .drop_vars(concat_dim, errors="ignore")
    )
    return xr.concat([data_without_dim, data_with_dim], dim=concat_dim)


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


def get_inference_steps(time_config: TimeConfig, hist: int = 1):
    """
    Get the number of inference/rollout steps for the given time configuration.

    Args:
        time_config: Time configuration
        hist: Number of rollout steps

    Returns:
        num_steps: Number of rollout steps
    """
    time_delta = TIME_DELTA
    start_time_str = time_config.start
    start_year, start_month, start_day = start_time_str.split("-")
    start_time = cftime.DatetimeNoLeap(
        int(start_year), int(start_month), int(start_day), 0, 0, 0
    )

    end_time_str = time_config.end
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
    out_dict.update(add_derived_variables(tensor_out))
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

    return data_src.map(_anom, suffix="anomalies")


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


def validate_data(
    src: DataSource, boundary_vars: BoundaryVarNames, static_data_vars: list[str] | None = None
) -> DataSource:
    """Validate the data such that we have the correct format for training."""
    anomalies_vars = get_anomalies_vars(boundary_vars)

    if static_data_vars is not None:

        def _static_data_checks(data):
            for var in static_data_vars:
                assert var in data.variables, (
                    f"Static data variable {var} not found in data"
                )
                if "time" in data[var].dims:
                    data[var] = data[var].isel(time=0)

            return data

        src = src.map_data(_static_data_checks, suffix="static_data_checked")

    if src.is_compact:
        src_ = src.map_data(with_lat_lon_coords)
        return compute_anomalies(src_, anomalies_vars)

    def validated(data, means, stds):
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

    src_ = src.map(validated, suffix="validated")

    # Check if any anomalies are needed to be computed
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
        wet_mask_surface: torch.Tensor,
    ) -> None:
        """Store normalization parameters and pre-compute numpy arrays."""
        prognostic_src = src.filter(prognostic_var_names, prefix="prognostic")
        boundary_src = src.filter(boundary_var_names, prefix="boundary")
        self.prognostic_mean = prognostic_src.means
        self.prognostic_std = prognostic_src.stds
        self.boundary_mean = boundary_src.means
        self.boundary_std = boundary_src.stds
        self.wet_mask = wet_mask
        self.wet_mask_surface = wet_mask_surface

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

        expand_var_dim = [1] * data.ndim
        expand_var_dim[-3] = -1
        assert data.shape[-3] == self._prognostic_mean_np.shape[0]
        tensor_mean = tensor_mean.reshape(expand_var_dim)
        tensor_std = tensor_std.reshape(expand_var_dim)

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

        expand_var_dim = [1] * data.ndim
        expand_var_dim[-3] = -1
        assert data.shape[-3] == self._prognostic_mean_np.shape[0]
        tensor_mean = tensor_mean.reshape(expand_var_dim)
        tensor_std = tensor_std.reshape(expand_var_dim)

        unnorm = data * tensor_std + tensor_mean
        unnorm = torch.where(self.wet_mask.to(data.device) == 0, fill_value, unnorm)
        unnorm = unnorm.to(data.dtype)
        return unnorm

    def unnormalize_tensor_boundary(
        self, data: torch.Tensor, fill_value=float("nan")
    ) -> torch.Tensor:
        """Unnormalize boundary tensor."""
        tensor_mean = self._to_tensor(self._boundary_mean_np, data.device)
        tensor_std = self._to_tensor(self._boundary_std_np, data.device)

        expand_var_dim = [1] * data.ndim
        expand_var_dim[-3] = -1
        assert data.shape[-3] == self._boundary_mean_np.shape[0]
        tensor_mean = tensor_mean.reshape(expand_var_dim)
        tensor_std = tensor_std.reshape(expand_var_dim)

        unnorm = data * tensor_std + tensor_mean
        unnorm = torch.where(
            self.wet_mask_surface.to(data.device) == 0, fill_value, unnorm
        )
        unnorm = unnorm.to(data.dtype)
        return unnorm


@dataclasses.dataclass
class LoadStats:
    """Captures stats about loading a single TrainData object."""

    load_time_seconds: float

    @classmethod
    def accumulated(cls, stats: list["LoadStats"]) -> "LoadStats":
        """Accumulate the stats across multiple LoadStats objects in a batch."""
        return cls(sum(s.load_time_seconds for s in stats))
