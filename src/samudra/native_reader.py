# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Canonical OM4 channels over interchangeable synchronous plane readers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, Self

import numpy as np
import xarray as xr
from jaxtyping import Int

from samudra.constants import CanonicalPlanes, DataLayout
from samudra.utils.data import (
    CanonicalReadRequest,
    ChannelStatistics,
    XarrayCanonicalReader,
)
from samudra.utils.location import LocalLocation


@dataclass(frozen=True)
class PhysicalVariable:
    """A physical array, optionally selecting a depth index."""

    name: str
    level: int | None = None


class PlaneReader(Protocol):
    @property
    def shape(self) -> tuple[int, int, int]: ...

    def read_into(
        self,
        time_indices: list[int],
        variables: Sequence[PhysicalVariable],
        output: CanonicalPlanes,
    ) -> None:
        """Fill float32 (time, channel, lat, lon) caller-owned storage.

        All writes must finish before returning OR raising, including cancellation
        and BaseException. The caller can release/reuse the buffer immediately.
        Implementations must also support zero-length requests.
        """
        ...


class Om4IoRuntime(Protocol):
    """Own one backend's process-local I/O resources across all sources."""

    def open(
        self, path: Path, variables: Sequence[PhysicalVariable]
    ) -> PlaneReader: ...


@dataclass(frozen=True)
class NativeOm4Reader:
    """Xarray metadata and native reads over a physical OM4 plane reader."""

    xarray_reader: XarrayCanonicalReader
    channels: tuple[str, ...]
    path: str
    _planes: PlaneReader
    _reader_variables: dict[str, PhysicalVariable]
    _physical_time_indices: Int[np.ndarray, " source_time"]
    _spatial_shape: tuple[int, int]

    @property
    def time(self) -> xr.DataArray:
        return self.xarray_reader.time

    @property
    def resolution(self):
        return self.xarray_reader.resolution

    def statistics(self, channels: tuple[str, ...]) -> ChannelStatistics:
        return self.xarray_reader.statistics(channels)

    @property
    def attrs(self):
        return self.xarray_reader.attrs

    @property
    def storage_id(self) -> int:
        return id(self._planes)

    def slice_time(self, time) -> Self:
        xarray_reader = self.xarray_reader.slice_time(time)
        positions = self.time.to_index().get_indexer(xarray_reader.time.to_index())
        if np.any(positions < 0):
            raise AssertionError("Canonical time slice could not be mapped to storage")
        physical = self._physical_time_indices[positions].copy()
        physical.setflags(write=False)
        return replace(
            self,
            xarray_reader=xarray_reader,
            _physical_time_indices=physical,
        )

    def read(self, request: CanonicalReadRequest) -> CanonicalPlanes:
        shape = (
            *request.time_indices.shape,
            len(request.channels),
            *self.spatial_shape,
        )
        output = np.empty(shape, dtype=np.float32)
        self.read_into(
            request.time_indices.reshape(-1),
            request.channels,
            output.reshape(
                request.time_indices.size, len(request.channels), *self.spatial_shape
            ),
        )
        return output

    def coordinates(self):
        return self.xarray_reader.coordinates()

    def metadata(self, data_layout):
        return self.xarray_reader.metadata(data_layout)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self._spatial_shape

    def read_into(
        self,
        time_indices: Int[np.ndarray, " time"],
        channels: tuple[str, ...],
        output: CanonicalPlanes,
    ) -> None:
        """Read positions relative to this reader's current time slice."""
        if time_indices.ndim != 1:
            raise ValueError("Canonical time indices must be one-dimensional")
        missing = set(channels).difference(self._reader_variables)
        if missing:
            raise KeyError(f"Canonical channels not found: {sorted(missing)}")
        self._planes.read_into(
            self._physical_time_indices[time_indices].tolist(),
            [self._reader_variables[name] for name in channels],
            output,
        )


def _validate_native_encoding(
    variable: xr.DataArray, location: LocalLocation, backend: str
) -> None:
    """Reject CF transforms that direct physical plane reads do not implement."""
    # Xarray moves decoded CF metadata out of attrs and into encoding. Inspect
    # metadata only: reading values here could load an entire ocean field.
    encoding = variable.attrs | variable.encoding
    unsupported = [name for name in ("scale_factor", "add_offset") if name in encoding]
    for name in ("_FillValue", "missing_value"):
        fill = encoding.get(name)
        if fill is not None and not np.all(np.isnan(fill)):
            unsupported.append(name)
    if unsupported:
        raise ValueError(
            f"loading.type={backend!r} does not support CF encoding "
            f"{unsupported} on variable {variable.name!r} in {location.path}; "
            "native reads require unscaled float32 values and NaN missing-value "
            "sentinels. Use loading.type='cpu' to apply Xarray's CF decoding."
        )


def build_om4_reader(
    xarray_reader: XarrayCanonicalReader,
    location: LocalLocation,
    data_layout: DataLayout,
    runtime: Om4IoRuntime,
    *,
    backend: str,
) -> NativeOm4Reader:
    """Construct an OM4 reader from canonical metadata and an I/O runtime."""
    physical = location.open({})
    reader_variables: dict[str, PhysicalVariable] = {}
    for logical_name in xarray_reader.channels:
        if logical_name in physical.data_vars:
            reader_variables[logical_name] = PhysicalVariable(logical_name)
            continue
        base, separator, level_text = logical_name.rpartition("_")
        if separator and level_text.isdigit() and base in physical.data_vars:
            variable = physical[base]
            if "lev" in variable.dims:
                reader_variables[logical_name] = PhysicalVariable(base, int(level_text))
                continue
        matches = []
        for physical_name in (
            physical.data_vars if separator and level_text.isdigit() else ()
        ):
            name = str(physical_name)
            if not name.startswith(f"{base}_lev_"):
                continue
            depth = float(name.split("_lev_", 1)[1].replace("_", "."))
            if data_layout.depth_levels.index(depth) == int(level_text):
                matches.append(name)
        if len(matches) == 1:
            reader_variables[logical_name] = PhysicalVariable(matches[0])
            continue
        raise ValueError(
            f"Could not map canonical OM4 channel {logical_name!r} in {location.path}"
        )

    for physical_name in dict.fromkeys(
        value.name for value in reader_variables.values()
    ):
        _validate_native_encoding(physical[physical_name], location, backend)

    native = runtime.open(
        location.path, tuple(dict.fromkeys(reader_variables.values()))
    )

    physical_time = physical["time"].to_index()
    canonical_time = xarray_reader.time.to_index()
    if not physical_time.is_unique:
        raise ValueError(f"Native store {location.path} has duplicate time coordinates")
    if not canonical_time.is_unique:
        raise ValueError("Canonical dataset has duplicate time coordinates")
    physical_indices = physical_time.get_indexer(canonical_time).astype(
        np.int64, copy=False
    )
    if np.any(physical_indices < 0):
        missing = canonical_time[physical_indices < 0]
        raise ValueError(
            f"Canonical dataset times are missing from Native store {location.path}: "
            f"{list(missing[:3])}"
        )

    time_size, lat, lon = native.shape
    if time_size != len(physical_time):
        raise ValueError(
            f"Native store {location.path} reports {time_size} rows, but its time "
            f"coordinate has {len(physical_time)}"
        )
    if (lat, lon) != tuple(len(axis) for axis in xarray_reader.resolution):
        raise ValueError(
            f"Native store {location.path} has spatial shape {(lat, lon)}, but the "
            f"canonical dataset has {tuple(len(axis) for axis in xarray_reader.resolution)}"
        )
    physical_indices = physical_indices.copy()
    physical_indices.setflags(write=False)
    return NativeOm4Reader(
        xarray_reader=xarray_reader,
        channels=xarray_reader.channels,
        path=str(location.path),
        _planes=native,
        _reader_variables=reader_variables,
        _physical_time_indices=physical_indices,
        _spatial_shape=(lat, lon),
    )
