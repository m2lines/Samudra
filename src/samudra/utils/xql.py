# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental Xarray-SQL reader for canonical ocean data.

Install the optional dependencies with ``uv sync --group xql``.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pyarrow as pa
import xarray as xr

from samudra.constants import DataLayout
from samudra.utils.data import (
    CanonicalReader,
    CanonicalReadRequest,
    CanonicalSource,
    ChannelStatistics,
)

if TYPE_CHECKING:
    from samudra.config import TimeConfig


_CONTEXTS: dict[tuple[int, str], Any] = {}
_CONTEXTS_LOCK = threading.Lock()


def _quoted_identifier(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _require_xarray_sql() -> Any:
    try:
        import xarray_sql as xql
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "XQL data loading requires the optional dependencies; run "
            "`uv sync --group xql`."
        ) from error
    return xql


@dataclasses.dataclass(frozen=True)
class XqlCanonicalReader:
    """Query canonical planes through a process-local Xarray-SQL context.

    Context creation is delayed until the first read.  Samudra constructs its
    sources in the parent process and reads them in PyTorch DataLoader workers,
    so this avoids inheriting DataFusion's Tokio runtime across ``fork()`` (the
    failure tracked by xarray-sql issue #145).
    """

    semantic: CanonicalReader
    data: xr.Dataset
    time_chunk_size: int = 1
    _cache_key: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.time_chunk_size <= 0:
            raise ValueError("XQL time_chunk_size must be positive")
        missing = set(self.channels).difference(self.data.data_vars)
        if missing:
            raise ValueError(
                f"XQL data is missing canonical channels: {sorted(missing)}"
            )
        unexpected_dims = {
            name: self.data[name].dims
            for name in self.channels
            if self.data[name].dims != ("time", "lat", "lon")
        }
        if unexpected_dims:
            raise ValueError(
                "XQL canonical channels must have dimensions "
                f"('time', 'lat', 'lon'): {unexpected_dims}"
            )

    @property
    def channels(self) -> tuple[str, ...]:
        return self.semantic.channels

    @property
    def time(self) -> xr.DataArray:
        return self.semantic.time

    @property
    def resolution(self):
        return self.semantic.resolution

    def statistics(self, channels: tuple[str, ...]) -> ChannelStatistics:
        return self.semantic.statistics(channels)

    @property
    def attrs(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.semantic.attrs))

    def slice_time(self, time: TimeConfig) -> Self:
        return dataclasses.replace(
            self,
            semantic=self.semantic.slice_time(time),
            data=self.data.sel(time=time.time_slice),
            _cache_key=uuid.uuid4().hex,
        )

    def _context(self) -> Any:
        # A parent and a forked worker have different PIDs, and consequently can
        # never share a DataFusion context even though their Python object IDs
        # and memory initially match.
        key = (os.getpid(), self._cache_key)
        with _CONTEXTS_LOCK:
            if key not in _CONTEXTS:
                xql = _require_xarray_sql()
                query_data = self.data[list(self.channels)]
                auxiliary_coords = [
                    name for name in query_data.coords if name not in query_data.dims
                ]
                if auxiliary_coords:
                    query_data = query_data.drop_vars(auxiliary_coords)
                # TODO(Codex): Would it be better for XQL to implement https://github.com/xqlsystems/xarray-sql/pull/219?
                query_data = query_data.assign_coords(
                    {
                        dim: np.arange(size, dtype=np.int64)
                        for dim, size in query_data.sizes.items()
                    }
                )
                context = xql.XarrayContext()
                context.from_dataset(
                    "canonical",
                    query_data,
                    chunks={"time": self.time_chunk_size},
                )
                _CONTEXTS[key] = context
            return _CONTEXTS[key]

    def read(self, request: CanonicalReadRequest) -> np.ndarray:
        missing = set(request.channels).difference(self.channels)
        if missing:
            raise KeyError(f"Canonical channels not found: {sorted(missing)}")

        try:
            physical_indices = np.arange(self.time.size)[request.time_indices]
        except IndexError as error:
            raise IndexError("Canonical time index is out of bounds") from error

        output_shape = (
            *request.time_indices.shape,
            len(request.channels),
            self.data.sizes["lat"],
            self.data.sizes["lon"],
        )
        if physical_indices.size == 0 or not request.channels:
            return np.empty(output_shape, dtype=np.float32)

        unique_indices, inverse = np.unique(
            physical_indices.reshape(-1), return_inverse=True
        )
        query_channels = tuple(dict.fromkeys(request.channels))
        selected_columns = ", ".join(
            _quoted_identifier(name) for name in query_channels
        )
        selected_times = ", ".join(str(int(index)) for index in unique_indices)
        query = (
            f'SELECT "time", "lat", "lon", {selected_columns} '
            'FROM "canonical" '
            f'WHERE "time" IN ({selected_times}) '
            'ORDER BY "time", "lat", "lon"'
        )
        batches = self._context().sql(query).collect()
        table = pa.Table.from_batches(batches)

        spatial_shape = (self.data.sizes["lat"], self.data.sizes["lon"])
        expected_rows = len(unique_indices) * np.prod(spatial_shape)
        if table.num_rows != expected_rows:
            raise RuntimeError(
                f"XQL returned {table.num_rows} rows; expected {expected_rows}"
            )
        values_by_channel = {
            name: table.column(name)
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .reshape(len(unique_indices), *spatial_shape)
            for name in query_channels
        }
        unique_values = np.stack(
            [values_by_channel[name] for name in request.channels],
            axis=1,
        ).astype(np.float32, copy=False)
        return unique_values[inverse].reshape(output_shape)

    def coordinates(self) -> Mapping[str, xr.DataArray]:
        return self.semantic.coordinates()

    def metadata(self, data_layout: DataLayout) -> dict:
        return self.semantic.metadata(data_layout)


def with_xql_reader(
    source: CanonicalSource, *, time_chunk_size: int = 1
) -> CanonicalSource:
    """Return ``source`` with canonical reads routed through Xarray-SQL."""
    return source.with_reader(
        XqlCanonicalReader(
            semantic=source.reader,
            data=source.to_xarray_dataset(),
            time_chunk_size=time_chunk_size,
        )
    )
