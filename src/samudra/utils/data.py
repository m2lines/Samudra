# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, final

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from jaxtyping import Bool

if TYPE_CHECKING:
    from samudra.config import TimeConfig

from samudra.constants import (
    BatchTimeSeriesOutput,
    BoundaryVarNames,
    DataLayout,
    DictSingleChannelVar,
    Grid,
    GridMask,
    GridSize,
    Input,
    Lat,
    LoaderVersion,
    Lon,
    Prognostic,
    PrognosticMask,
    PrognosticVarNames,
    SingleTimeSeriesOutput,
    construct_metadata,
)
from samudra.derived_variables import add_derived_variables

logger = logging.getLogger(__name__)

_CANONICAL_MASK_PREFIX = "mask_"
_STACKED_WET_MASK_VAR = "wetmask"


@dataclasses.dataclass(frozen=True)
class Masks:
    """Read-only mask metadata used to expose the ocean and mask land.

    Tensor contents are shared between canonical views for efficiency and must
    be treated as immutable by callers.
    """

    prognostic: PrognosticMask
    boundary: GridMask

    def __post_init__(self):
        object.__setattr__(self, "prognostic", self.prognostic.bool())
        object.__setattr__(self, "boundary", self.boundary.bool())

    def prognostic_with_hist(
        self, hist: int
    ) -> Bool[GridMask, " prognostic_vars*({hist}+1)"]:
        return torch.concat([self.prognostic] * (hist + 1), dim=0)


@dataclasses.dataclass(frozen=True)
class CanonicalReadRequest:
    """A storage-independent request for canonical ocean-data planes.

    The shape of ``time_indices`` defines the leading dimensions of the returned
    planes. Keeping this core request to NumPy makes it usable by Python and native
    readers without importing xarray concepts into the boundary.
    """

    time_indices: np.ndarray
    channels: tuple[str, ...]

    def __post_init__(self) -> None:
        indices = np.asarray(self.time_indices)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Canonical time indices must be integers")
        immutable_indices = indices.astype(np.int64, copy=True)
        immutable_indices.setflags(write=False)
        object.__setattr__(self, "time_indices", immutable_indices)
        object.__setattr__(self, "channels", tuple(self.channels))


@dataclasses.dataclass(frozen=True)
class ChannelStatistics:
    """Normalization statistics aligned one-for-one with canonical channels."""

    mean: np.ndarray
    std: np.ndarray


class CanonicalReader(Protocol):
    """Narrow storage seam implemented by xarray now and native readers later."""

    @property
    def channels(self) -> tuple[str, ...]: ...

    @property
    def time(self) -> xr.DataArray: ...

    @property
    def resolution(self) -> tuple[Lat, Lon]: ...

    def statistics(self, channels: tuple[str, ...]) -> ChannelStatistics: ...

    @property
    def attrs(self) -> Mapping[str, Any]: ...

    def slice_time(self, time: "TimeConfig") -> Self: ...

    def read(self, request: CanonicalReadRequest) -> np.ndarray: ...

    def coordinates(self) -> Mapping[str, xr.DataArray]: ...

    def metadata(self, data_layout: DataLayout) -> dict: ...


@dataclasses.dataclass(frozen=True)
class _XarrayCanonicalReader:
    """Private xarray implementation of the canonical read contract."""

    data: xr.Dataset
    means: xr.Dataset
    stds: xr.Dataset
    channels: tuple[str, ...]

    @property
    def time(self) -> xr.DataArray:
        return self.data.time.copy(deep=True)

    @property
    def resolution(self) -> tuple[Lat, Lon]:
        return (
            torch.from_numpy(self.data.lat.values).clone(),
            torch.from_numpy(self.data.lon.values).clone(),
        )

    def statistics(self, channels: tuple[str, ...]) -> ChannelStatistics:
        self._validate_channels(channels)
        return ChannelStatistics(
            _flatten(self.means[list(channels)]),
            _flatten(self.stds[list(channels)]),
        )

    @property
    def attrs(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.data.attrs))

    def _validate_channels(self, channels: tuple[str, ...]) -> None:
        missing = set(channels).difference(self.channels)
        if missing:
            raise KeyError(f"Canonical channels not found: {sorted(missing)}")

    def slice_time(self, time: "TimeConfig") -> Self:
        return dataclasses.replace(self, data=self.data.sel(time=time.time_slice))

    def read(self, request: CanonicalReadRequest) -> np.ndarray:
        self._validate_channels(request.channels)
        index_dims = [f"index_{i}" for i in range(request.time_indices.ndim)]
        index = xr.DataArray(request.time_indices, dims=index_dims)
        selected = self.data[list(request.channels)].isel(time=index)

        # Materialize one combined graph, rather than loading canonical channels
        # independently. Compact level views share their base-array Dask keys, so
        # the scheduler can read/decompress each physical chunk once per request.
        values = (
            selected.to_array(dim="channel")
            .transpose(*index_dims, "channel", "lat", "lon")
            .to_numpy()
            .astype(np.float32, copy=False)
        )
        return values

    def coordinates(self) -> Mapping[str, xr.DataArray]:
        return MappingProxyType(
            {
                str(name): coordinate.copy(deep=True)
                for name, coordinate in self.data.coords.items()
            }
        )

    def metadata(self, data_layout: DataLayout) -> dict:
        return construct_metadata(self.data, data_layout)


@final
@dataclasses.dataclass(frozen=True)
class CanonicalSource:
    """A structurally immutable, read-capable view of canonical ocean data.

    Physical xarray layout is private to the reader. In particular, callers see
    the same ordered channels for flat and compact OM4 stores. Channel selection
    and time slicing return new views and never mutate the source. Tensor-valued
    masks are shared, read-only metadata; mutating their contents is unsupported.
    """

    name: str
    _reader: CanonicalReader
    masks: Masks
    data_layout: DataLayout

    @property
    def reader(self) -> CanonicalReader:
        """Return the storage reader so backends can decorate its read behavior."""
        return self._reader

    def with_reader(self, reader: CanonicalReader) -> Self:
        """Return an equivalent source backed by a replacement reader."""
        if reader.channels != self.channels:
            raise ValueError(
                "Replacement reader channels must match the canonical source: "
                f"expected {self.channels}, got {reader.channels}"
            )
        return dataclasses.replace(self, _reader=reader)

    @classmethod
    def from_canonical_datasets(
        cls,
        name: str,
        data: xr.Dataset,
        means: xr.Dataset,
        stds: xr.Dataset,
        masks: Masks,
        data_layout: DataLayout,
    ) -> Self:
        """Construct from datasets that are already in canonical channel form.

        Raw OM4 callers should use :meth:`from_datasets`. This factory remains
        useful for focused in-memory tests and named preprocessing stages.
        """
        channels = tuple(str(name) for name in means.data_vars)
        if any("lev" in dataset.dims for dataset in (data, means, stds)):
            raise ValueError("Canonical datasets cannot expose a 'lev' dimension")
        if set(channels) - set(data.data_vars) or set(channels) - set(stds.data_vars):
            raise ValueError("Canonical data, means, and stds have different channels")
        return cls(
            name=name,
            _reader=_XarrayCanonicalReader(
                data,
                means[list(channels)],
                stds[list(channels)],
                channels,
            ),
            masks=masks,
            data_layout=data_layout,
        )

    @property
    def channels(self) -> tuple[str, ...]:
        return self._reader.channels

    @property
    def time(self) -> xr.DataArray:
        return self._reader.time

    def statistics(self, channels: Sequence[str]) -> ChannelStatistics:
        return self._reader.statistics(tuple(channels))

    @property
    def attrs(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(dict(self._reader.attrs))

    @property
    def resolution(self) -> tuple[Lat, Lon]:
        # Readers return defensive coordinate tensors. Do not cache and expose a
        # mutable tensor that could silently alter future callers' grid context.
        return self._reader.resolution

    @cached_property
    def grid_size(self) -> GridSize:
        res = self.resolution
        return res[0].shape[0], res[1].shape[0]

    @cached_property
    def spherical_area_weights(self) -> Grid:
        lat, lon = self.resolution
        weights = torch.cos(torch.deg2rad(lat)).repeat(lon.shape[0], 1).t()
        return weights / weights.sum()

    @cached_property
    def metadata(self) -> dict:
        return self._reader.metadata(self.data_layout)

    def slice_time(self, time: "TimeConfig") -> Self:
        """Slice the data source to only include the specified time slice."""
        data_time_min = self.time.values.min()
        data_time_max = self.time.values.max()
        time_start = time.time_slice.start
        time_end = time.time_slice.stop
        if time_start > data_time_max or time_end < data_time_min:
            raise ValueError(
                f"Time slice {time} is entirely outside the range of the data "
                f"{str(data_time_min)[:10]} to {str(data_time_max)[:10]}"
            )

        if time_start < data_time_min or time_end > data_time_max:
            logger.warning(
                f"Time slice {time} is partially outside the range of the data "
                f"{str(data_time_min)[:10]} to {str(data_time_max)[:10]}"
            )

        return dataclasses.replace(
            self,
            name=f"{time=}[{self.name}]",
            _reader=self._reader.slice_time(time),
        )

    def read(self, time_indices: np.ndarray, channels: Sequence[str]) -> np.ndarray:
        """Read canonical channels at integer time indices."""
        return self._reader.read(CanonicalReadRequest(time_indices, tuple(channels)))

    def coordinates(self) -> dict[str, xr.DataArray]:
        return dict(self._reader.coordinates())

    def _xarray_datasets_for_testing(
        self,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """Expose xarray fixtures without making them part of the public contract."""
        if not isinstance(self._reader, _XarrayCanonicalReader):
            raise TypeError("This canonical dataset is not backed by xarray")
        return self._reader.data, self._reader.means, self._reader.stds

    @classmethod
    def from_datasets(
        cls,
        data: xr.Dataset,
        means: xr.Dataset,
        stds: xr.Dataset,
        *,
        data_layout: DataLayout,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        name: str = "CanonicalSource",
    ) -> Self:
        """Build a canonical reader from already-canonicalized xarray datasets."""
        channels = tuple(dict.fromkeys((*prognostic_var_names, *boundary_var_names)))
        for dataset in (data, means, stds):
            level_channels = [
                name
                for name in channels
                if name in dataset and "lev" in dataset[name].dims
            ]
            if level_channels:
                raise ValueError(
                    "Canonical channels cannot expose a 'lev' dimension: "
                    f"{level_channels}"
                )
        masks = extract_wet_mask(
            data,
            prognostic_var_names,
            boundary_var_names,
        )

        missing_data = set(channels).difference(data.data_vars)
        missing_means = set(channels).difference(means.data_vars)
        missing_stds = set(channels).difference(stds.data_vars)
        if missing_data or missing_means or missing_stds:
            raise ValueError(
                "Canonical channels are missing: "
                f"data={sorted(missing_data)}, means={sorted(missing_means)}, "
                f"stds={sorted(missing_stds)}"
            )

        return cls(
            name=name,
            _reader=_XarrayCanonicalReader(
                data=data,
                means=means[list(channels)],
                stds=stds[list(channels)],
                channels=channels,
            ),
            masks=masks,
            data_layout=data_layout,
        )


@dataclasses.dataclass
class SourceSplits:
    train: CanonicalSource
    val: CanonicalSource
    inference: CanonicalSource | None


@dataclasses.dataclass
class DataBundle:
    train_sources: list[CanonicalSource]
    val_sources: list[CanonicalSource]
    inference_source: CanonicalSource | None
    loader_version: LoaderVersion
    data_layout: DataLayout


def _flatten(ds: xr.Dataset) -> np.ndarray:
    """Flatten scalar statistics already aligned to canonical channel order."""
    if "lev" in ds.dims:
        raise ValueError("Canonical statistics cannot expose a 'lev' dimension")
    return ds.to_array().to_numpy().reshape(-1)


def _level_index_from_var_name(var_name: str) -> int:
    suffix = var_name.rsplit("_", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def _var_without_level(var_name: str) -> str:
    suffix = var_name.rsplit("_", maxsplit=1)[-1]
    return var_name.rsplit("_", maxsplit=1)[0] if suffix.isdigit() else var_name


def _preferred_available_var(
    data: xr.Dataset, candidates: tuple[str, ...]
) -> str | None:
    return next((name for name in candidates if name in data.data_vars), None)


def _mask_var_for_data_var(
    data: xr.Dataset,
    var_name: str,
) -> str:
    level = _level_index_from_var_name(var_name)
    base_var = _var_without_level(var_name)
    llc_staggered_masks = {
        "U": (f"mask_w_{level}", f"hFacW_{level}"),
        "oceTAUX": ("mask_w_0", "hFacW_0"),
        "V": (f"mask_s_{level}", f"hFacS_{level}"),
        "oceTAUY": ("mask_s_0", "hFacS_0"),
    }
    if base_var in llc_staggered_masks:
        if mask_var := _preferred_available_var(data, llc_staggered_masks[base_var]):
            return mask_var
    return f"{_CANONICAL_MASK_PREFIX}{level}"


def _mask_array_for_data_var(
    data: xr.Dataset,
    var_name: str,
) -> np.ndarray:
    mask = data[_mask_var_for_data_var(data, var_name)]
    if "time" in mask.dims:
        mask = mask.isel(time=0)
    return mask.to_numpy()


def extract_wet_mask(
    data: xr.Dataset,
    prognostic_var_names: PrognosticVarNames,
    boundary_var_names: BoundaryVarNames,
) -> Masks:
    """A mask for where the oceans are. Water is wet."""
    data_ = flatten_masks(data)
    wet_inp_np = np.stack(
        [_mask_array_for_data_var(data_, var_name) for var_name in prognostic_var_names]
    )
    boundary_mask_vars = [
        _mask_var_for_data_var(data_, var_name) for var_name in boundary_var_names
    ]
    boundary_masks = [
        _mask_array_for_data_var(data_, var_name) for var_name in boundary_var_names
    ]
    wet_surface_mask_np = (
        np.stack(boundary_masks)
        if len(set(boundary_mask_vars)) > 1
        else boundary_masks[0]
    )

    wet_inp = torch.from_numpy(wet_inp_np)
    wet_surface = torch.from_numpy(wet_surface_mask_np)
    return Masks(wet_inp.bool(), wet_surface.bool())


def flatten_masks(
    data: xr.Dataset,
) -> xr.Dataset:
    """Adds level-wise mask variables from the stacked wet mask."""
    data_ = data.copy()
    if f"{_CANONICAL_MASK_PREFIX}0" not in data_.variables:
        assert _STACKED_WET_MASK_VAR in data_.variables, (
            "Wet mask cannot be constructed without "
            "either the wetmask variable or the level-wise masks"
        )

        wet_mask = data_[_STACKED_WET_MASK_VAR]
        for i in range(wet_mask.sizes["lev"]):
            data_[f"{_CANONICAL_MASK_PREFIX}{i}"] = wet_mask.isel(lev=i)

        data_ = data_.drop_vars(_STACKED_WET_MASK_VAR)

    return data_


def unflatten_masks(
    data: xr.Dataset,
    num_levels: int,
) -> xr.Dataset:
    """Adds a stacked wet mask `xarray.DataArray` from level-wise mask variables."""
    data_ = data.copy()
    mask_vars = [f"{_CANONICAL_MASK_PREFIX}{i}" for i in range(num_levels)]
    if _STACKED_WET_MASK_VAR not in data_.variables:
        assert mask_vars[0] in data_.variables, "Wet mask must have masks as data vars!"

        wetmask = data_[mask_vars].to_array(dim="lev", name=_STACKED_WET_MASK_VAR)

        lev = data_.coords.get("lev", np.arange(len(mask_vars)))
        data_[_STACKED_WET_MASK_VAR] = wetmask.assign_coords(lev=lev)
        data_ = data_.drop_vars(mask_vars)

    return data_


def spherical_area_weights(data: xr.Dataset) -> Grid:
    num_lon = data.lon.size
    lats = torch.from_numpy(data.lat.to_numpy())
    weights = torch.cos(torch.deg2rad(lats)).repeat(num_lon, 1).t()
    weights /= weights.sum()
    return weights


def spherical_area(data: xr.Dataset) -> Grid:
    """
    Compute real grid cell areas on a spherical Earth.

    Uses the spherical geometry formula:
    A = R² × Δλ × (sin(φ₂) - sin(φ₁))

    where:
    - R is Earth's radius (6371 km)
    - Δλ is the longitude spacing in radians
    - φ₁, φ₂ are the latitude bounds of the cell in radians

    Args:
        data: Dataset containing lat/lon coordinates

    Returns:
        Grid cell areas in m²
    """
    R = 6371000  # Earth radius in meters

    lats = data.lat.to_numpy()
    lons = data.lon.to_numpy()

    # Compute grid spacing (assuming uniform spacing)
    dlat = np.abs(np.diff(lats).mean())
    dlon = np.abs(np.diff(lons).mean())

    # Convert to radians
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)
    lats_rad = np.deg2rad(lats)

    # Compute cell areas: A = R² × dlon × (sin(lat + dlat/2) - sin(lat - dlat/2))
    areas = np.zeros((len(lats), len(lons)))
    for i, lat_rad in enumerate(lats_rad):
        lat_north = lat_rad + dlat_rad / 2
        lat_south = lat_rad - dlat_rad / 2
        area = R**2 * dlon_rad * (np.sin(lat_north) - np.sin(lat_south))
        areas[i, :] = area

    return torch.from_numpy(areas)


def get_inference_steps(data_source: CanonicalSource, hist: int = 1):
    """
    Get the number of inference/rollout steps for the given time configuration.

    Args:
        data_source: The data source sliced to the inference time range
        hist: How many additional history samples we get per step

    Returns:
        num_steps: Total number of rolled-out inferences which fit into the time range
    """
    num_steps = data_source.time.size

    # Might have extra remaining days, so we remove them
    mod = num_steps % (hist + 1)
    num_steps = num_steps - mod
    return num_steps


def convert_tensor_out_to_dict(
    tensor_out: torch.Tensor,
    *,
    data_layout: DataLayout,
) -> DictSingleChannelVar:
    assert tensor_out.ndim == 5
    assert tensor_out.shape[2] == len(data_layout.prognostic_var_names)
    out_dict = {}
    for i, var in enumerate(data_layout.prognostic_var_names):
        out_dict[var] = tensor_out[:, :, i]
    out_dict.update(add_derived_variables(tensor_out, data_layout=data_layout))
    return out_dict


def get_aggregator_dicts(
    data: Prognostic | Input,
    preprocessor: "BatchPreprocessor",
    data_layout: DataLayout,
    wet: torch.Tensor,
    long_rollout: bool,
    input_type: Literal["prognostic", "input"] = "prognostic",
    num_prognostic_channels: int = 0,
    hist: int = 1,
) -> tuple[DictSingleChannelVar, DictSingleChannelVar]:
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
    elif data.ndim != 5:
        # Batches are independent rollouts during validation
        data_reshaped = rearrange(data, "n (hi c) h w -> n hi c h w", hi=hist + 1)
    else:
        # This case comes up in tests; typically, data is not in the desired shape automatically.
        data_reshaped = data

    # Get normalized dict
    data_normalized = data_reshaped.clone()
    data_normalized = torch.where(wet == 0, float("nan"), data_normalized)
    data_dict = convert_tensor_out_to_dict(data_normalized, data_layout=data_layout)
    # Unnormalize
    data_unnorm = preprocessor.unnormalize_tensor_prognostic(
        data_reshaped, fill_value=float("nan")
    )
    # Get unnormalized dict
    data_unnorm_dict = convert_tensor_out_to_dict(data_unnorm, data_layout=data_layout)
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


def with_level_index_vars(
    data: xr.Dataset,
    depth_levels: Sequence[float],
) -> xr.Dataset:
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
            lev_in_depth_idx = depth_levels.index(lev_in_depth)
            data_copy = data_copy.rename({var_str: f"{var}_{lev_in_depth_idx!s}"})

    return data_copy


def with_depth_value_vars(
    data: xr.Dataset,
    data_layout: DataLayout,
) -> xr.Dataset:
    """Inverse of `with_level_index_vars`: name 3D variables by depth value.

    Renames the depth-resolved prognostic variables (``<var>_<level_index>``)
    back to the OM4 ``<var>_lev_<depth>`` form (e.g. ``thetao_0`` ->
    ``thetao_lev_2_5``). Which variables to rename is read directly off
    ``data_layout.prognostic_var_names`` rather than inferred from the data, so
    per-level masks (``mask_<i>``) and level-free prognostics (e.g. ``zos``) are
    never mistaken for depth-resolved variables by name alone.
    """
    renames = {}
    for var_name in data_layout.prognostic_var_names:
        base, _, idx = var_name.rpartition("_")
        if not (base and idx.isdigit()):
            continue  # level-free prognostic variable, e.g. zos
        depth = data_layout.depth_levels[int(idx)]
        depth_str = str(depth).replace(".", "_")
        if var_name in data.variables:
            renames[var_name] = f"{base}_lev_{depth_str}"

    return data.rename(renames)


def with_lat_lon_coords(data: xr.Dataset) -> xr.Dataset:
    """Standardize dataset coordinates; prefer "lat"/"lon" over "y"/"x"."""
    data_copy = data.copy()
    if "lat" not in data_copy.dims:
        # Preserve the 2-D geographic coords under a non-colliding name, then
        # rename the x/y dims to the 1-D lat/lon we standardize on.
        preserve = {n: f"{n}_2d" for n in ("lat", "lon") if n in data_copy.coords}
        data_copy = data_copy.rename(preserve).rename({"x": "lon", "y": "lat"})

    return data_copy


class BatchPreprocessor:
    def __init__(
        self,
        source: CanonicalSource,
        prognostic_var_names: Sequence[str],
        boundary_var_names: Sequence[str],
        *,
        normalize_before_mask: bool = True,
        masked_fill_value: float = 0.0,
    ) -> None:
        """Prepare canonical host tensors for models and restore physical values."""
        self.prognostic_mask = source.masks.prognostic
        self.boundary_mask = source.masks.boundary
        self.normalize_before_mask = normalize_before_mask
        self.masked_fill_value = masked_fill_value

        # Pre-compute arrays for faster tensor normalization. Canonicalization has
        # already aligned every scalar statistic to one ordered logical channel.
        prognostic_statistics = source.statistics(prognostic_var_names)
        boundary_statistics = source.statistics(boundary_var_names)
        self._prognostic_mean_np = prognostic_statistics.mean
        self._prognostic_std_np = prognostic_statistics.std
        self._boundary_mean_np = boundary_statistics.mean
        self._boundary_std_np = boundary_statistics.std

    @staticmethod
    def _reshape_statistics(statistics: torch.Tensor, ndim: int) -> torch.Tensor:
        shape = [1] * ndim
        shape[-3] = -1
        return statistics.reshape(shape)

    @classmethod
    def _normalize_tensor(
        cls,
        data: torch.Tensor,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        fill_nan: bool = True,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        if data.shape[-3] != mean.shape[0]:
            raise ValueError(
                f"Expected {mean.shape[0]} variable channels, got {data.shape[-3]}"
            )
        tensor_mean = cls._reshape_statistics(
            torch.from_numpy(mean).to(data.device, data.dtype), data.ndim
        )
        tensor_std = cls._reshape_statistics(
            torch.from_numpy(std).to(data.device, data.dtype), data.ndim
        )
        normalized = (data - tensor_mean) / tensor_std
        if fill_nan:
            normalized = normalized.nan_to_num(nan=fill_value)
        return normalized.to(data.dtype)

    def _prepare(
        self,
        data: torch.Tensor,
        *,
        mean: np.ndarray,
        std: np.ndarray,
        mask: torch.Tensor,
        device: torch.device,
    ) -> Input:
        tensor = data.to(device, non_blocking=True)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 5:
            raise ValueError(f"Expected 4D or 5D canonical planes, got {tensor.ndim}D")
        mask = mask.to(device, non_blocking=True)
        if self.normalize_before_mask:
            tensor = self._normalize_tensor(tensor, mean, std)
        tensor = torch.where(mask, tensor, self.masked_fill_value)
        if not self.normalize_before_mask:
            tensor = self._normalize_tensor(tensor, mean, std)
        return rearrange(
            tensor, "batch time variable lat lon -> batch (time variable) lat lon"
        )

    def prepare_prognostic(
        self, data: torch.Tensor, device: torch.device
    ) -> Prognostic:
        return self._prepare(
            data,
            mean=self._prognostic_mean_np,
            std=self._prognostic_std_np,
            mask=self.prognostic_mask,
            device=device,
        )

    def prepare_boundary(self, data: torch.Tensor, device: torch.device) -> Input:
        return self._prepare(
            data,
            mean=self._boundary_mean_np,
            std=self._boundary_std_np,
            mask=self.boundary_mask,
            device=device,
        )

    def normalize_tensor_prognostic(
        self, data: torch.Tensor, fill_nan=True, fill_value=0.0
    ) -> torch.Tensor:
        """Normalize a prognostic tensor without masking or flattening."""
        return self._normalize_tensor(
            data,
            self._prognostic_mean_np,
            self._prognostic_std_np,
            fill_nan=fill_nan,
            fill_value=fill_value,
        )

    def unnormalize_tensor_prognostic(
        self, data: torch.Tensor, fill_value=float("nan")
    ) -> torch.Tensor:
        """Unnormalize prognostic tensor and apply fill value to land cells."""
        tensor_mean = torch.from_numpy(self._prognostic_mean_np).to(
            data.device, data.dtype
        )
        tensor_std = torch.from_numpy(self._prognostic_std_np).to(
            data.device, data.dtype
        )

        expand_var_dim = [1] * data.ndim
        expand_var_dim[-3] = -1
        assert data.shape[-3] == self._prognostic_mean_np.shape[0]
        tensor_mean = tensor_mean.reshape(expand_var_dim)
        tensor_std = tensor_std.reshape(expand_var_dim)

        unnorm = data * tensor_std + tensor_mean
        unnorm = torch.where(
            self.prognostic_mask.to(data.device) == 0, fill_value, unnorm
        )
        unnorm = unnorm.to(data.dtype)
        return unnorm

    def unnormalize_tensor_boundary(
        self, data: torch.Tensor, fill_value=float("nan")
    ) -> torch.Tensor:
        """Unnormalize boundary tensor."""
        tensor_mean = torch.from_numpy(self._boundary_mean_np).to(
            data.device, data.dtype
        )
        tensor_std = torch.from_numpy(self._boundary_std_np).to(data.device, data.dtype)

        expand_var_dim = [1] * data.ndim
        expand_var_dim[-3] = -1
        assert data.shape[-3] == self._boundary_mean_np.shape[0]
        tensor_mean = tensor_mean.reshape(expand_var_dim)
        tensor_std = tensor_std.reshape(expand_var_dim)

        unnorm = data * tensor_std + tensor_mean
        unnorm = torch.where(
            self.boundary_mask.to(data.device) == 0, fill_value, unnorm
        )
        unnorm = unnorm.to(data.dtype)
        return unnorm


@dataclasses.dataclass
class LoadStats:
    """Captures stats about loading a single ModelBatch object."""

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


def stack_levels(
    data: xr.Dataset,
    data_layout: DataLayout,
) -> xr.Dataset:
    """Reassemble a flattened OM4 dataset into analysis-ready, depth-stacked form.

    Inverts the preprocessing flattening so downstream analysis does not have to:
    per-level prognostic channels (``thetao_0`` ...) become ``thetao(lev, ...)``
    and the per-level ``mask_i`` become a single stacked ``wetmask``. Grid
    coordinates are preserved. This is the dataset-level counterpart to the eval
    writer's reassembly, for putting ground-truth inputs on the same footing as
    predictions.

    Implemented as the inverse of `with_level_index_vars` followed by the existing
    `compact_dataset` (stacks the ``_lev_`` form) and `unflatten_masks`.
    """
    data = with_depth_value_vars(data, data_layout)
    data = compact_dataset(data)
    if "mask_0" in data.variables:
        data = unflatten_masks(data, num_levels=len(data_layout.depth_levels))
    return data
