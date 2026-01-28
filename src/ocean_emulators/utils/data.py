import dataclasses
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import cftime
import datetime as dt
import torch
import xarray as xr
from einops import rearrange
from jaxtyping import Bool
from torch.utils.dlpack import from_dlpack

if TYPE_CHECKING:
    from ocean_emulators.config import TimeConfig

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
from ocean_emulators.derived_variables import add_derived_variables
from ocean_emulators.utils.location import ResolvedLocation
from ocean_emulators.utils.multiton import Multiton
from ocean_emulators.utils.device import using_gpu

logger = logging.getLogger(__name__)
_XARRAY_BACKEND_LOGGED: set[str] = set()

try:
    import dask.array as da
except ImportError:
    da = None

try:
    import cupy as cp
except ImportError:
    cp = None


def _detect_cupy_nvrtc() -> bool:
    if cp is None:
        return False
    try:
        from cupy_backends.cuda.libs import nvrtc

        nvrtc.getVersion()
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning(
            "CuPy is installed but NVRTC is not available (%s). "
            "Disabling CuPy-backed dask conversion.",
            exc,
        )
        return False
    return True


_CUPY_ENABLED = _detect_cupy_nvrtc()


def _compute_if_needed(array):
    compute = getattr(array, "compute", None)
    if callable(compute):
        return compute()
    return array


def _xr_to_torch(
    array_like: xr.DataArray | xr.Variable | np.ndarray | torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if isinstance(array_like, (xr.DataArray, xr.Variable)):
        array = array_like.data
    else:
        array = array_like

    array = _compute_if_needed(array)

    if cp is not None and isinstance(array, cp.ndarray):
        tensor = from_dlpack(array)
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device, non_blocking=True)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    tensor = torch.as_tensor(array)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device=device, non_blocking=True)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _to_cupy_dask_array(array: object):
    if not _CUPY_ENABLED or da is None or not isinstance(array, da.Array):
        return array
    if isinstance(array._meta, cp.ndarray):
        return array
    meta = cp.array((), dtype=array.dtype)
    return array.map_blocks(cp.asarray, dtype=array.dtype, meta=meta)


def ensure_cupy_backend(dataset: xr.Dataset, *, allow_eager: bool = False) -> xr.Dataset:
    """Best-effort conversion of dataset variables to CuPy.

    If allow_eager=True and a variable is a NumPy-backed array, convert it to
    a CuPy array directly. This is intended for small datasets (means/stds).
    """
    if not using_gpu() or not _CUPY_ENABLED or da is None:
        return dataset

    changed = False
    data_vars: dict[str, xr.DataArray] = {}
    for name, var in dataset.data_vars.items():
        data = _to_cupy_dask_array(var.data)
        if data is var.data and allow_eager and cp is not None:
            if not isinstance(data, cp.ndarray):
                data = cp.asarray(data)
        if data is not var.data:
            changed = True
            data_vars[name] = xr.DataArray(
                data,
                dims=var.dims,
                coords=var.coords,
                attrs=var.attrs,
                name=var.name,
            )

    if not changed:
        return dataset

    updated = dataset.copy()
    for name, data_array in data_vars.items():
        updated[name] = data_array
    return updated


def is_xarray_cupy_backend(dataset: xr.Dataset) -> bool:
    if cp is None:
        return False
    if not dataset.data_vars:
        return False
    sample = next(iter(dataset.data_vars.values()))
    data = sample.data
    if da is not None and isinstance(data, da.Array):
        return isinstance(data._meta, cp.ndarray)
    return isinstance(data, cp.ndarray)


def log_xarray_backend(dataset: xr.Dataset, *, label: str) -> bool:
    """Log whether xarray is backed by CuPy (GPU) or NumPy (CPU) arrays."""
    if label in _XARRAY_BACKEND_LOGGED:
        return is_xarray_cupy_backend(dataset)
    _XARRAY_BACKEND_LOGGED.add(label)

    if not dataset.data_vars:
        logger.warning("xarray backend check for %s: no data variables found.", label)
        return False

    sample = next(iter(dataset.data_vars.values()))
    data = sample.data

    try:
        import dask.array as da
    except ImportError:
        da = None

    is_dask = da is not None and isinstance(data, da.Array)
    if is_dask:
        meta = data._meta
        is_cupy = cp is not None and isinstance(meta, cp.ndarray)
        backend = "cupy" if is_cupy else type(meta).__name__
        logger.info("xarray backend for %s: dask[%s]", label, backend)
    else:
        is_cupy = cp is not None and isinstance(data, cp.ndarray)
        backend = "cupy" if is_cupy else type(data).__name__
        logger.info("xarray backend for %s: %s", label, backend)

    if using_gpu():
        if is_cupy:
            logger.info("xarray backend for %s is GPU-backed (cupy).", label)
        elif not _CUPY_ENABLED:
            logger.warning(
                "GPU is enabled but NVRTC is unavailable; "
                "CuPy-backed dask conversion is disabled for %s.",
                label,
            )
        else:
            logger.warning(
                "GPU is enabled but xarray backend for %s is %s; "
                "data will stage on CPU. Install cupy + cupy-xarray and "
                "use dask/chunked zarr to enable GPU-backed arrays.",
                label,
                backend,
            )
    return is_cupy


def _coerce_time_bounds(
    time_coord: xr.DataArray, time_cfg: "TimeConfig"
) -> tuple[object, object]:
    """Coerce a TimeConfig to the same type as the dataset time coordinate."""
    if np.issubdtype(time_coord.dtype, np.datetime64):
        return (
            np.datetime64(str(time_cfg.start)),
            np.datetime64(str(time_cfg.end)),
        )
    if time_coord.dtype == object:
        data = time_coord.values
        sample = data.ravel()[0] if data.size else None
        if isinstance(sample, cftime.datetime):
            return time_cfg.start.datetime, time_cfg.end.datetime
        if isinstance(sample, (np.datetime64, dt.datetime, dt.date)):
            return (
                np.datetime64(str(time_cfg.start)),
                np.datetime64(str(time_cfg.end)),
            )
    units = time_coord.attrs.get("units")
    calendar = time_coord.attrs.get("calendar", "standard")
    if units is None:
        raise ValueError(
            "Time coordinate is numeric but missing 'units'; "
            "unable to compare with configured Julian dates."
        )
    start = cftime.date2num(time_cfg.start.datetime, units=units, calendar=calendar)
    end = cftime.date2num(time_cfg.end.datetime, units=units, calendar=calendar)
    return start, end


def _format_time_value(value: object, time_coord: xr.DataArray) -> str:
    if isinstance(value, cftime.datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="D")
    units = time_coord.attrs.get("units")
    calendar = time_coord.attrs.get("calendar", "standard")
    if units is not None:
        try:
            dt = cftime.num2date(value, units=units, calendar=calendar)
            if isinstance(dt, np.ndarray):
                dt = dt.item()
            if isinstance(dt, cftime.datetime):
                return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return str(value)


def _time_indices(
    time_coord: xr.DataArray, start: object, end: object
) -> np.ndarray:
    """Return numpy indices for the given time window without requiring indexes."""
    data = time_coord.data
    if cp is not None and isinstance(data, cp.ndarray):
        values = data.get()
    elif da is not None and isinstance(data, da.Array):
        values = data.compute()
        if cp is not None and isinstance(values, cp.ndarray):
            values = values.get()
    else:
        values = np.asarray(data)

    mask = (values >= start) & (values < end)
    return np.nonzero(mask)[0]


def _var_name_encode_level(var_name: str) -> bool:
    """Check if the variable name encodes the level."""
    var_name_encodes_level = re.compile(r"_[0-9]+")
    return bool(var_name_encodes_level.search(var_name))


def _is_compact(data: xr.Dataset, means: xr.Dataset, stds: xr.Dataset) -> bool:
    return all(
        not _var_name_encode_level(str(v))
        for d in [data, means, stds]
        for v in d.keys()
    )


@dataclasses.dataclass
class Masks:
    """A collection of masks to expose the ocean and mask land."""

    prognostic: PrognosticMask
    boundary: GridMask

    def __post_init__(self):
        self.prognostic = self.prognostic.bool()
        self.boundary = self.boundary.bool()

    def prognostic_with_hist(
        self, hist: int
    ) -> Bool[GridMask, " prognostic_vars*({hist}+1)"]:
        return torch.concat([self.prognostic] * (hist + 1), dim=0)


@dataclasses.dataclass
class DataSource:
    """Data source for the model."""

    name: str
    data: xr.Dataset
    means: xr.Dataset
    stds: xr.Dataset
    masks: Masks

    @cached_property
    def is_compact(self) -> bool:
        """Check if the data source is compact."""
        return _is_compact(self.data, self.means, self.stds)

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
                if not _var_name_encode_level(mangled_var_name):
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

    def slice(self, time: "TimeConfig") -> Self:
        """Slice the data source to only include the specified time slice."""
        start, end = _coerce_time_bounds(self.data.time, time)
        data_time_min = self.data.time.min().item()
        data_time_max = self.data.time.max().item()
        if start > data_time_max or end < data_time_min:
            raise ValueError(
                f"Time slice {time} is entirely outside the range of the data "
                f"{_format_time_value(data_time_min, self.data.time)} to "
                f"{_format_time_value(data_time_max, self.data.time)}"
            )

        if start < data_time_min or end > data_time_max:
            logger.warning(
                f"Time slice {time} is partially outside the range of the data "
                f"{_format_time_value(data_time_min, self.data.time)} to "
                f"{_format_time_value(data_time_max, self.data.time)}"
            )

        indices = _time_indices(self.data.time, start, end)
        data = self.data.isel(time=indices)
        return dataclasses.replace(self, name=f"{time=}[{self.name}]", data=data)

    # TODO(jder): delete this once we've de-duplicated InferenceDataset with TorchTrainDataset
    def normalize(self, fill_nan=True, fill_value=0.0) -> xr.Dataset:
        """Normalize input data."""
        means = ensure_cupy_backend(self.means, allow_eager=True)
        stds = ensure_cupy_backend(self.stds, allow_eager=True)
        norm = (self.data - means) / stds
        if fill_nan:
            norm = norm.fillna(fill_value)
        return norm

    # TODO(jder): delete this once we've de-duplicated InferenceDataset with TorchTrainDataset
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

        if "lev" in self.means.dims:
            means_arr = conditional_rearrange(
                self.means,
                "(variable lev)=var",
                concat_dim="var",
            ).rename({"var": "variable"})
        else:
            means_arr = self.means.to_array()
        if "lev" in self.stds.dims:
            stds_arr = conditional_rearrange(
                self.stds,
                "(variable lev)=var",
                concat_dim="var",
            ).rename({"var": "variable"})
        else:
            stds_arr = self.stds.to_array()

        means = _xr_to_torch(means_arr, device=data.device).reshape(reshape_vars)
        stds = _xr_to_torch(stds_arr, device=data.device).reshape(reshape_vars)

        norm = (data - means) / stds
        if fill_nan:
            norm = norm.nan_to_num(nan=fill_value)
        norm = norm.to(data.dtype)
        return norm

    @classmethod
    def from_locations(
        cls,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        *,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        static_data_vars: list[str] | None,
        use_dask: bool,
    ) -> Self:
        chunks: dict[str, int] | None = {} if use_dask else None
        data = data_location.open(chunks)
        means = means_location.open(chunks)
        stds = stds_location.open(chunks)

        return cls.from_datasets(
            data,
            means,
            stds,
            prognostic_var_names=prognostic_var_names,
            boundary_var_names=boundary_var_names,
            static_data_vars=static_data_vars,
            name=f"{data_location}-{use_dask}",
        )

    @classmethod
    def from_datasets(
        cls,
        data: xr.Dataset,
        means: xr.Dataset,
        stds: xr.Dataset,
        *,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        static_data_vars: list[str] | None = None,
        name: str = "DataSource",
    ) -> Self:
        data, means, stds = validate_data(
            data,
            means,
            stds,
            boundary_var_names=boundary_var_names,
            static_data_vars=static_data_vars,
        )
        masks = extract_wet_mask(data, prognostic_var_names)

        return cls(
            name=name,
            data=data,
            means=means,
            stds=stds,
            masks=masks,
        )


@dataclasses.dataclass
class DataContainer:
    source: DataSource
    source_using_dask: DataSource
    loader_version: LoaderVersion
    supports_fork: bool
    static_data: xr.Dataset | None = None


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

    This method is stable: even if it creates a new number of dimensions, it will
    preserve the order of the variables in the original dataset.

    Args:
        data: The dataset to rearrange.
        pattern: The einsum pattern to use for rearranging.
        except_dim: The dimension to exclude from the pattern.
        concat_dim: The dimension to concatenate along.

    Returns:
        The combined, rearranged dataset as a `xarray.DataArray`.
    """
    assert except_dim in pattern, f"{except_dim} must be in the pattern."

    all_vars = list(data.keys())

    vars_with_dim = [v for v in data if except_dim in data[v].dims]
    vars_without_dim = [v for v in data if except_dim not in data[v].dims]

    # Some of the `vars_without_dim` may need to appear before or behind `vars_with_dim`
    # in the final data array. These lists help preserve the correct order of the vars,
    # even after a rearrangement (i.e. merge to two or more dimensions).
    back = [
        v
        for v in vars_without_dim
        if all_vars.index(v) > all_vars.index(vars_with_dim[0])
    ]
    front = [
        v
        for v in vars_without_dim
        if all_vars.index(v) < all_vars.index(vars_with_dim[-1])
    ]

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

    da = xr.concat([data_with_dim, data_without_dim], dim=concat_dim)

    n_front = len(front)  # e.g. n_front=2
    n_center = data_with_dim.sizes[concat_dim]  # e.g. n_center=10
    n_back = len(back)  # e.g. n_back=3

    # In the `concat` above, we put all the `data_without_dim` vars at the end. Some of
    # these need to be moved to the front, and the rest stays at the back. Here, we
    # compute a list of indices that will sort the data in the correct order.
    #
    # e.g. with the example constants above, order would look like:
    #  array([10, 11,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 12, 13, 14])
    order = np.concatenate(
        (
            # Moves vars to the front
            np.roll(np.arange((new_front := n_center + n_front)), n_front),
            np.arange(new_front, new_front + n_back),  # rest of vars
        )
    )
    order_da = xr.DataArray(order, dims=concat_dim)

    return da.sortby(order_da)


def extract_wet_mask(
    data: xr.Dataset, prognostic_var_names: PrognosticVarNames
) -> Masks:
    """A mask for where the oceans are. Water is wet."""
    data_ = flatten_masks(data)
    wet_mask = data_[MASK_VARS]
    if "time" in wet_mask.dims:
        wet_mask_arr = wet_mask.isel(time=0).to_array()
        wet_surface_mask_arr = wet_mask[MASK_VARS[0]].isel(time=0)
    else:
        wet_mask_arr = wet_mask.to_array()
        wet_surface_mask_arr = wet_mask[MASK_VARS[0]]

    depth_ind = _parse_lev_from_output_var(prognostic_var_names)

    wet_mask_tensor = _xr_to_torch(wet_mask_arr)
    wet_inp = wet_mask_tensor[depth_ind]
    wet_surface = _xr_to_torch(wet_surface_mask_arr)
    return Masks(wet_inp.bool(), wet_surface.bool())


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
    lats = _xr_to_torch(data.lat, dtype=torch.float32)
    weights = torch.cos(torch.deg2rad(lats)).repeat(num_lon, 1).t()
    weights /= weights.sum()
    return weights


def get_inference_steps(data_source: DataSource, hist: int = 1):
    """
    Get the number of inference/rollout steps for the given time configuration.

    Args:
        data_source: The data source sliced to the inference time range
        hist: How many additional history samples we get per step

    Returns:
        num_steps: Total number of rolled-out inferences which fit into the time range
    """
    num_steps = data_source.data.time.size

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
    data: xr.Dataset,
    means: xr.Dataset,
    stds: xr.Dataset,
    anomalies_vars: tuple[str, ...],
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Compute anomalies for the given variables."""
    for var in anomalies_vars:
        base_var = var.replace("_anomalies", "")
        if var not in data.variables and base_var in data.variables:
            logger.info(f"Computing anomalies for {base_var}")
            climatology = (
                data[base_var].groupby("time.dayofyear").mean("time").compute()
            )
            # Remove the seasonal cycle (climatology) from the detrended data
            day_of_year = data[base_var]["time"].dt.dayofyear
            data[var] = (
                data[base_var] - climatology.sel(dayofyear=day_of_year)
            ).compute()
            data = data.drop_vars(["dayofyear"])
            means[var] = data[var].mean().compute()
            stds[var] = data[var].std().compute()
    return data, means, stds


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
    data: xr.Dataset,
    means: xr.Dataset,
    stds: xr.Dataset,
    boundary_var_names: BoundaryVarNames,
    static_data_vars: list[str] | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Validate the data such that we have the correct format for training."""
    is_compact = _is_compact(data, means, stds)
    if static_data_vars is not None:
        for var in static_data_vars:
            assert var in data.variables, (
                f"Static data variable {var} not found in data"
            )
            if "time" in data[var].dims:
                data[var] = data[var].isel(time=0)

    if is_compact:
        data = with_lat_lon_coords(data)
    else:
        data = (
            data.pipe(flatten_masks)
            .pipe(with_level_index_vars)
            .pipe(with_lat_lon_coords)
        )

        # Check if data variables are in the right format
        # This check is to ensure we convert data to the correct format
        means = with_level_index_vars(means)
        stds = with_level_index_vars(stds)

    # Check if any anomalies are needed to be computed
    anomalies_vars = get_anomalies_vars(boundary_var_names)
    out = (
        compute_anomalies(data, means, stds, anomalies_vars)
        if anomalies_vars
        else (data, means, stds)
    )

    return out


# TODO: Repetitive code. Refactor
class Normalize(Multiton):
    def _initialize(
        self,
        src: DataSource,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
    ) -> None:
        """Store normalization parameters and pre-compute tensor views."""
        prognostic_src = src.filter(prognostic_var_names, prefix="prognostic")
        boundary_src = src.filter(boundary_var_names, prefix="boundary")
        self.prognostic_mean = ensure_cupy_backend(
            prognostic_src.means, allow_eager=True
        )
        self.prognostic_std = ensure_cupy_backend(
            prognostic_src.stds, allow_eager=True
        )
        self.boundary_mean = ensure_cupy_backend(
            boundary_src.means, allow_eager=True
        )
        self.boundary_std = ensure_cupy_backend(
            boundary_src.stds, allow_eager=True
        )
        self.wet_mask = src.masks.prognostic
        self.wet_mask_surface = src.masks.boundary

        self._prognostic_mean_t = _xr_to_torch(
            self.prognostic_mean.to_array(),
            dtype=torch.float32,
        ).reshape(-1)
        self._prognostic_std_t = _xr_to_torch(
            self.prognostic_std.to_array(),
            dtype=torch.float32,
        ).reshape(-1)
        self._boundary_mean_t = _xr_to_torch(
            self.boundary_mean.to_array(),
            dtype=torch.float32,
        ).reshape(-1)
        self._boundary_std_t = _xr_to_torch(
            self.boundary_std.to_array(),
            dtype=torch.float32,
        ).reshape(-1)

        # Backward-compatible attribute names used in tests and downstream code.
        self._prognostic_mean_np = self._prognostic_mean_t
        self._prognostic_std_np = self._prognostic_std_t
        self._boundary_mean_np = self._boundary_mean_t
        self._boundary_std_np = self._boundary_std_t

    def normalize_tensor_prognostic(
        self, data: torch.Tensor, fill_nan=True, fill_value=0.0
    ) -> torch.Tensor:
        """Normalize prognostic tensor."""
        tensor_mean = self._prognostic_mean_t.to(data.device, non_blocking=True)
        tensor_std = self._prognostic_std_t.to(data.device, non_blocking=True)

        expand_var_dim = [1] * data.ndim
        expand_var_dim[-3] = -1
        assert data.shape[-3] == self._prognostic_mean_t.shape[0]
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
        tensor_mean = self._prognostic_mean_t.to(data.device, non_blocking=True)
        tensor_std = self._prognostic_std_t.to(data.device, non_blocking=True)

        expand_var_dim = [1] * data.ndim
        expand_var_dim[-3] = -1
        assert data.shape[-3] == self._prognostic_mean_t.shape[0]
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
        tensor_mean = self._boundary_mean_t.to(data.device, non_blocking=True)
        tensor_std = self._boundary_std_t.to(data.device, non_blocking=True)

        expand_var_dim = [1] * data.ndim
        expand_var_dim[-3] = -1
        assert data.shape[-3] == self._boundary_mean_t.shape[0]
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


def compact_dataset(ds: xr.Dataset) -> xr.Dataset:
    data = ds.copy()

    var_groups = defaultdict(list)
    for key in data.keys():
        if "_lev_" in (k := str(key)):
            base_name = k.split("_lev_")[0]
            var_groups[base_name].append(k)

    def _parse_level(x) -> float:
        return float(x.split("_lev_")[1].replace("_", "."))

    for base_var, vars_ in var_groups.items():
        sorted_vars = sorted(vars_, key=_parse_level)
        levels = [_parse_level(var) for var in sorted_vars]
        if hasattr(data, "lev"):
            levels = data.lev.values
        da = xr.concat([data[var] for var in sorted_vars], dim="lev").assign_coords(
            lev=("lev", levels)
        )
        data[base_var] = da
        data = data.drop_vars(vars_)

    return data
