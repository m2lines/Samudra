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
from jaxtyping import Bool, Float

if TYPE_CHECKING:
    from samudra.config import TimeConfig

from samudra.constants import (
    BatchTimeSeriesOutput,
    BoundaryVarNames,
    DatasetSpec,
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
    TensorMap,
    construct_metadata,
)
from samudra.derived_variables import add_derived_variables
from samudra.utils.location import ResolvedLocation

logger = logging.getLogger(__name__)


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

    def __post_init__(self) -> None:
        indices = np.asarray(self.time_indices)
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Canonical time indices must be integers")
        immutable_indices = indices.astype(np.int64, copy=True)
        immutable_indices.setflags(write=False)
        object.__setattr__(self, "time_indices", immutable_indices)


@dataclasses.dataclass(frozen=True)
class LoadedPlanes:
    """Canonical planes ordered as requested, with channels before lat/lon."""

    values: np.ndarray


@dataclasses.dataclass(frozen=True)
class ChannelStatistics:
    """Normalization statistics aligned one-for-one with canonical channels."""

    mean: np.ndarray
    std: np.ndarray


class CanonicalReader(Protocol):
    """Narrow storage seam implemented by xarray now and native readers later."""

    channels: tuple[str, ...]

    @property
    def time(self) -> xr.DataArray: ...

    @property
    def resolution(self) -> tuple[Lat, Lon]: ...

    @property
    def statistics(self) -> ChannelStatistics: ...

    @property
    def attrs(self) -> Mapping[str, Any]: ...

    def select_channels(self, channels: tuple[str, ...]) -> Self: ...

    def slice_time(self, time: "TimeConfig") -> Self: ...

    def read(self, request: CanonicalReadRequest) -> LoadedPlanes: ...

    def coordinates(self) -> Mapping[str, xr.DataArray]: ...

    def read_static(self, names: Sequence[str]) -> xr.Dataset: ...

    def metadata(self, dataset_spec: DatasetSpec) -> dict: ...


@dataclasses.dataclass(frozen=True)
class _XarrayCanonicalReader:
    """Private xarray implementation of the canonical read contract."""

    data: xr.Dataset
    means: xr.Dataset
    stds: xr.Dataset
    channels: tuple[str, ...]
    data_channels: frozenset[str]

    @property
    def time(self) -> xr.DataArray:
        return self.data.time.copy(deep=True)

    @property
    def resolution(self) -> tuple[Lat, Lon]:
        return (
            torch.from_numpy(self.data.lat.values).clone(),
            torch.from_numpy(self.data.lon.values).clone(),
        )

    @property
    def statistics(self) -> ChannelStatistics:
        return ChannelStatistics(_flatten(self.means), _flatten(self.stds))

    @property
    def attrs(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.data.attrs))

    def select_channels(self, channels: tuple[str, ...]) -> Self:
        missing = set(channels).difference(self.channels)
        if missing:
            raise KeyError(f"Canonical channels not found: {sorted(missing)}")
        return dataclasses.replace(
            self,
            means=self.means[list(channels)],
            stds=self.stds[list(channels)],
            channels=channels,
        )

    def slice_time(self, time: "TimeConfig") -> Self:
        return dataclasses.replace(self, data=self.data.sel(time=time.time_slice))

    def read(self, request: CanonicalReadRequest) -> LoadedPlanes:
        index_dims = [f"index_{i}" for i in range(request.time_indices.ndim)]
        index = xr.DataArray(request.time_indices, dims=index_dims)
        selected = self.data[list(self.channels)].isel(time=index)

        # Materialize one combined graph, rather than loading canonical channels
        # independently. Compact level views share their base-array Dask keys, so
        # the scheduler can read/decompress each physical chunk once per request.
        values = (
            selected.to_array(dim="channel")
            .transpose(*index_dims, "channel", "lat", "lon")
            .to_numpy()
            .astype(np.float32, copy=False)
        )
        return LoadedPlanes(values)

    def coordinates(self) -> Mapping[str, xr.DataArray]:
        return MappingProxyType(
            {
                str(name): coordinate.copy(deep=True)
                for name, coordinate in self.data.coords.items()
            }
        )

    def read_static(self, names: Sequence[str]) -> xr.Dataset:
        overlap = set(names).intersection(self.data_channels)
        if overlap:
            raise ValueError(
                f"Training channels cannot be read as static fields: {sorted(overlap)}"
            )
        # Static model configuration is explicitly xarray-based today. Return an
        # independent materialized copy so mutation cannot affect future reads.
        return self.data[list(names)].copy(deep=True).load()

    def metadata(self, dataset_spec: DatasetSpec) -> dict:
        return construct_metadata(self.data, dataset_spec)


def _expand_canonical_channels(dataset: xr.Dataset) -> xr.Dataset:
    """Expand physical level dimensions into independent canonical channels."""
    canonical = xr.Dataset(attrs=dataset.attrs)
    for coord in ("time", "lat", "lon"):
        if coord in dataset.coords:
            canonical = canonical.assign_coords({coord: dataset.coords[coord]})

    for name, variable in dataset.data_vars.items():
        if "lev" not in variable.dims:
            canonical[str(name)] = variable
            continue
        for level in range(variable.sizes["lev"]):
            canonical[f"{name}_{level}"] = variable.isel(lev=level, drop=True)
    return canonical


def _canonicalize_om4_datasets(
    data: xr.Dataset,
    means: xr.Dataset,
    stds: xr.Dataset,
    *,
    dataset_spec: DatasetSpec,
    boundary_var_names: BoundaryVarNames,
    static_data_vars: list[str] | None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Canonicalize flat or compact OM4 into the same channel representation."""
    data = data.copy()
    means = means.copy()
    stds = stds.copy()
    if static_data_vars is not None:
        for var in static_data_vars:
            if var not in data.variables:
                raise ValueError(f"Static data variable {var} not found in data")
            if "time" in data[var].dims:
                data[var] = data[var].isel(time=0)

    data = with_lat_lon_coords(data)
    # Both layouts follow the same canonicalization pipeline: flat OM4 names are
    # normalized and compact variables are then expanded along their ``lev`` axis.
    # There is intentionally no format flag, even inside the xarray reader.
    data = data.pipe(with_level_index_vars, dataset_spec=dataset_spec)
    means = with_level_index_vars(means, dataset_spec=dataset_spec)
    stds = with_level_index_vars(stds, dataset_spec=dataset_spec)

    data = flatten_masks(data, dataset_spec=dataset_spec)
    anomalies_vars = get_anomalies_vars(boundary_var_names)
    if anomalies_vars:
        data, means, stds = compute_anomalies(data, means, stds, anomalies_vars)

    return tuple(_expand_canonical_channels(dataset) for dataset in (data, means, stds))  # type: ignore[return-value]


@final
@dataclasses.dataclass(frozen=True)
class CanonicalDataset:
    """A structurally immutable, read-capable view of canonical ocean data.

    Physical xarray layout is private to the reader. In particular, callers see
    the same ordered channels for flat and compact OM4 stores. Channel selection
    and time slicing return new views and never mutate the source. Tensor-valued
    masks are shared, read-only metadata; mutating their contents is unsupported.
    """

    name: str
    _reader: CanonicalReader
    masks: Masks
    dataset_spec: DatasetSpec

    @classmethod
    def from_canonical_datasets(
        cls,
        name: str,
        data: xr.Dataset,
        means: xr.Dataset,
        stds: xr.Dataset,
        masks: Masks,
        dataset_spec: DatasetSpec,
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
                frozenset(channels),
            ),
            masks=masks,
            dataset_spec=dataset_spec,
        )

    @property
    def channels(self) -> tuple[str, ...]:
        return self._reader.channels

    @property
    def time(self) -> xr.DataArray:
        return self._reader.time

    @property
    def statistics(self) -> ChannelStatistics:
        return self._reader.statistics

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
        return self._reader.metadata(self.dataset_spec)

    def select_channels(
        self,
        var_names: Sequence[str],
        *,
        prefix: str,
    ) -> Self:
        channels = tuple(str(name) for name in var_names)
        return dataclasses.replace(
            self,
            name=f"{prefix}[{self.name}]",
            _reader=self._reader.select_channels(channels),
        )

    def slice_time(self, time: "TimeConfig") -> Self:
        """Slice the data source to only include the specified time slice."""
        data_time_min = self.time.min().item()
        data_time_max = self.time.max().item()
        if time.start.datetime > data_time_max or time.end.datetime < data_time_min:
            raise ValueError(
                f"Time slice {time} is entirely outside the range of the data "
                f"{data_time_min.strftime('%Y-%m-%d')} to "
                f"{data_time_max.strftime('%Y-%m-%d')}"
            )

        if time.start.datetime < data_time_min or time.end.datetime > data_time_max:
            logger.warning(
                f"Time slice {time} is partially outside the range of the data "
                f"{data_time_min.strftime('%Y-%m-%d')} to "
                f"{data_time_max.strftime('%Y-%m-%d')}"
            )

        return dataclasses.replace(
            self,
            name=f"{time=}[{self.name}]",
            _reader=self._reader.slice_time(time),
        )

    def read(self, request: CanonicalReadRequest) -> LoadedPlanes:
        return self._reader.read(request)

    def coordinates(self) -> dict[str, xr.DataArray]:
        return dict(self._reader.coordinates())

    def read_static(self, names: Sequence[str]) -> xr.Dataset:
        """Read named static fields without exposing the backing training arrays."""
        return self._reader.read_static(names)

    def _xarray_datasets_for_testing(
        self,
    ) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """Expose xarray fixtures without making them part of the public contract."""
        if not isinstance(self._reader, _XarrayCanonicalReader):
            raise TypeError("This canonical dataset is not backed by xarray")
        return self._reader.data, self._reader.means, self._reader.stds

    @classmethod
    def from_locations(
        cls,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        *,
        dataset_spec: DatasetSpec,
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
            dataset_spec=dataset_spec,
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
        dataset_spec: DatasetSpec,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        static_data_vars: list[str] | None = None,
        name: str = "CanonicalDataset",
    ) -> Self:
        data, means, stds = _canonicalize_om4_datasets(
            data,
            means,
            stds,
            dataset_spec=dataset_spec,
            boundary_var_names=boundary_var_names,
            static_data_vars=static_data_vars,
        )
        masks = extract_wet_mask(data, prognostic_var_names, dataset_spec=dataset_spec)

        channels = tuple(dict.fromkeys((*prognostic_var_names, *boundary_var_names)))
        missing_data = set(channels).difference(data.data_vars)
        missing_means = set(channels).difference(means.data_vars)
        missing_stds = set(channels).difference(stds.data_vars)
        if missing_data or missing_means or missing_stds:
            raise ValueError(
                "Canonical OM4 channels are missing: "
                f"data={sorted(missing_data)}, means={sorted(missing_means)}, "
                f"stds={sorted(missing_stds)}"
            )

        return cls(
            name=name,
            _reader=_XarrayCanonicalReader(
                data=data,
                means=means,
                stds=stds,
                channels=channels,
                data_channels=frozenset(str(name) for name in means.data_vars),
            ),
            masks=masks,
            dataset_spec=dataset_spec,
        )


@dataclasses.dataclass
class OceanData:
    """A slice of ocean data (boundary or prognostic) with normalization statistics.

    This dataclass bundles raw tensor data with the statistics needed to normalize it
    and the mask needed to handle land/invalid values. It serves as an intermediary
    representation used when constructing training `Example`s from raw xarray data.

    The typical workflow is:
        1. Load canonical planes from a `CanonicalDataset`
        2. Slice to the desired time range with `with_time()`
        3. Apply normalization and masking with `normalize_and_mask()`
        4. Flatten time/variable dims to create the final `Input` or `Prognostic` tensor

    Attributes:
        data: Raw ocean variable values with shape (batch, time, variable, lat, lon).
        means: Per-variable means for normalization, shape (variable,).
        stds: Per-variable standard deviations for normalization, shape (variable,).
        mask: Boolean mask indicating valid ocean points (True) vs land (False),
            broadcast-compatible with the variable dimension.
    """

    data: Float[torch.Tensor, "batch time variable lat lon"]
    means: Float[torch.Tensor, " variable"]
    stds: Float[torch.Tensor, " variable"]
    mask: Bool[torch.Tensor, " variable"]

    @classmethod
    def from_data_source(
        cls,
        data: Float[torch.Tensor, "batch time variable lat lon"],
        mask: Float[torch.Tensor, " variable"],
        src: CanonicalDataset,
    ) -> Self:
        means_torch = torch.from_numpy(src.statistics.mean)
        stds_torch = torch.from_numpy(src.statistics.std)
        return cls(data, means_torch, stds_torch, mask)

    def with_time(self, time_range: slice) -> Self:
        """Slice data across the time dimension."""
        return dataclasses.replace(self, data=self.data[:, time_range, :, :, :])

    def _normalize(
        self,
        data: Float[torch.Tensor, "batch time var lat lon"],
        fill_nan: bool = True,
        fill_value: float = 0.0,
    ) -> Float[torch.Tensor, "batch time var lat lon"]:
        """Normalize input data treated as torch Tensors."""
        norm = (data - self.means.view(1, 1, -1, 1, 1)) / self.stds.view(1, 1, -1, 1, 1)
        if fill_nan:
            norm = norm.nan_to_num(nan=fill_value)
        norm = norm.to(data.dtype)
        return norm

    def normalize_and_mask(
        self, normalize_before_mask: bool, masked_fill_value: float
    ) -> Float[torch.Tensor, "batch time var lat lon"]:
        """Normalize and mask tensors."""
        tensor = self.data
        if normalize_before_mask:
            tensor = self._normalize(tensor)
        tensor = torch.where(self.mask, tensor, masked_fill_value)
        if not normalize_before_mask:
            tensor = self._normalize(tensor)
        return tensor

    def to(self, device: torch.device, non_blocking: bool = True) -> Self:
        return dataclasses.replace(
            self,
            data=self.data.to(device, non_blocking=non_blocking),
            means=self.means.to(device, non_blocking=non_blocking),
            stds=self.stds.to(device, non_blocking=non_blocking),
            mask=self.mask.to(device, non_blocking=non_blocking),
        )


@dataclasses.dataclass
class DataContainer:
    sources: list[CanonicalDataset]
    inference_source: CanonicalDataset
    loader_version: LoaderVersion
    dataset_spec: DatasetSpec
    # TODO(559): static_data should belong to the CanonicalDataset, since we now
    #  deal with multiple resolutions.
    static_data: xr.Dataset | None = None

    @property
    def primary_source(self) -> CanonicalDataset:
        return self.sources[0]


def _flatten(ds: xr.Dataset) -> np.ndarray:
    """Flatten scalar statistics already aligned to canonical channel order."""
    if "lev" in ds.dims:
        raise ValueError("Canonical statistics cannot expose a 'lev' dimension")
    return ds.to_array().to_numpy().reshape(-1)


def extract_wet_mask(
    data: xr.Dataset,
    prognostic_var_names: PrognosticVarNames,
    *,
    dataset_spec: DatasetSpec,
) -> Masks:
    """A mask for where the oceans are. Water is wet."""
    data_ = flatten_masks(data, dataset_spec=dataset_spec)
    wet_mask = data_[list(dataset_spec.mask_vars)]
    if "time" in wet_mask.dims:
        wet_mask_np = wet_mask.isel(time=0).to_array().to_numpy()
        wet_surface_mask_np = (
            wet_mask[dataset_spec.mask_vars[0]].isel(time=0).to_numpy()
        )
    else:
        wet_mask_np = wet_mask.to_array().to_numpy()
        wet_surface_mask_np = wet_mask[dataset_spec.mask_vars[0]].to_numpy()

    depth_ind = _parse_lev_from_output_var(
        prognostic_var_names, dataset_spec=dataset_spec
    )

    wet_inp = torch.from_numpy(wet_mask_np[depth_ind])
    wet_surface = torch.from_numpy(wet_surface_mask_np)
    return Masks(wet_inp.bool(), wet_surface.bool())


def _parse_lev_from_output_var(
    prognostic_var_names: PrognosticVarNames,
    *,
    dataset_spec: DatasetSpec,
) -> list[int]:
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


def flatten_masks(
    data: xr.Dataset,
    dataset_spec: DatasetSpec,
) -> xr.Dataset:
    """Adds level-wise mask variables from the stacked wet mask."""
    data_ = data.copy()
    mask_vars = list(dataset_spec.mask_vars)
    if mask_vars[0] not in data_.variables:
        assert dataset_spec.mask_all_levels_var in data_.variables, (
            "Wet mask cannot be constructed without "
            "either the wetmask variable or the level-wise masks"
        )

        wet_mask = data_[dataset_spec.mask_all_levels_var]
        for i, mask_var in enumerate(mask_vars):
            data_[mask_var] = wet_mask.isel(lev=i)

        data_ = data_.drop_vars(dataset_spec.mask_all_levels_var)

    return data_


def unflatten_masks(
    data: xr.Dataset,
    dataset_spec: DatasetSpec,
) -> xr.Dataset:
    """Adds a stacked wet mask `xarray.DataArray` from level-wise mask variables."""
    data_ = data.copy()
    mask_vars = list(dataset_spec.mask_vars)
    if dataset_spec.mask_all_levels_var not in data_.variables:
        assert mask_vars[0] in data_.variables, "Wet mask must have masks as data vars!"

        wetmask = data_[mask_vars].to_array(
            dim="lev", name=dataset_spec.mask_all_levels_var
        )

        lev = data_.coords.get("lev", np.arange(len(mask_vars)))
        data_[dataset_spec.mask_all_levels_var] = wetmask.assign_coords(lev=lev)
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


def get_inference_steps(data_source: CanonicalDataset, hist: int = 1):
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
    tensor_map: TensorMap,
) -> DictSingleChannelVar:
    assert tensor_out.ndim == 5
    assert tensor_out.shape[2] == len(tensor_map.prognostic_var_names)
    out_dict = {}
    for i, var in enumerate(tensor_map.prognostic_var_names):
        out_dict[var] = tensor_out[:, :, i]
    out_dict.update(add_derived_variables(tensor_out, tensor_map=tensor_map))
    return out_dict


def get_aggregator_dicts(
    data: Prognostic | Input,
    normalize: "Normalize",
    tensor_map: TensorMap,
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
    data_dict = convert_tensor_out_to_dict(data_normalized, tensor_map=tensor_map)
    # Unnormalize
    data_unnorm = normalize.unnormalize_tensor_prognostic(
        data_reshaped, fill_value=float("nan")
    )
    # Get unnormalized dict
    data_unnorm_dict = convert_tensor_out_to_dict(data_unnorm, tensor_map=tensor_map)
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
    dataset_spec: DatasetSpec,
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
            lev_in_depth_idx = dataset_spec.depth_levels.index(lev_in_depth)
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


class Normalize:
    def __init__(
        self,
        src: CanonicalDataset,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
    ) -> None:
        """Store normalization parameters and pre-compute numpy arrays."""
        prognostic_src = src.select_channels(prognostic_var_names, prefix="prognostic")
        boundary_src = src.select_channels(boundary_var_names, prefix="boundary")
        self.wet_mask = src.masks.prognostic
        self.wet_mask_surface = src.masks.boundary

        # Pre-compute arrays for faster tensor normalization. Canonicalization has
        # already aligned every scalar statistic to one ordered logical channel.
        self._prognostic_mean_np = prognostic_src.statistics.mean
        self._prognostic_std_np = prognostic_src.statistics.std
        self._boundary_mean_np = boundary_src.statistics.mean
        self._boundary_std_np = boundary_src.statistics.std
        self._wet_mask_np = self.wet_mask.numpy()

    def normalize_tensor_prognostic(
        self, data: torch.Tensor, fill_nan=True, fill_value=0.0
    ) -> torch.Tensor:
        """Normalize prognostic tensor."""
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

        norm = (data - tensor_mean) / tensor_std
        if fill_nan:
            norm = norm.nan_to_num(nan=fill_value)
        norm = norm.to(data.dtype)
        return norm

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
        unnorm = torch.where(self.wet_mask.to(data.device) == 0, fill_value, unnorm)
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
