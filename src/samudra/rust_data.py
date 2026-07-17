# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Local OM4 batch loading through the optional Rust extension."""

from __future__ import annotations

import importlib
import math
import time
import weakref
from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Lock
from typing import Any, Protocol, Self

import numpy as np
import torch
import xarray as xr

from samudra.datasets import BatchReadUse, ModelBatch, TrainBatchPreparer, TrainingShard
from samudra.utils.data import (
    CanonicalReader,
    CanonicalReadRequest,
    CanonicalSource,
    ChannelStatistics,
    LoadStats,
)
from samudra.utils.location import LocalLocation


def _load_extension() -> Any:
    try:
        extension = importlib.import_module("samudra_rust_loader")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Rust data loading requires the optional extension; in a source "
            "checkout run `uv sync --extra rust`, or install the matching "
            "samudra-rust-loader platform wheel."
        ) from error
    return extension


@dataclass(frozen=True)
class RustIoRuntime:
    """Process-local native I/O executor shared by canonical OM4 readers."""

    read_pool: Any


def create_rust_io_runtime(max_concurrent_reads: int) -> RustIoRuntime:
    return RustIoRuntime(_load_extension().FlatOm4ReadPool(max_concurrent_reads))


class BatchSampler(Protocol):
    def __iter__(self) -> Iterator[list[int]]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class HostPrefetch:
    pin_memory: bool


@dataclass(frozen=True)
class CudaPrefetch:
    """Prefetch through pinned host buffers onto a dedicated CUDA stream."""


type RustPrefetchPolicy = HostPrefetch | CudaPrefetch


@dataclass(frozen=True)
class FlatOm4Variable:
    physical_name: str

    def extension_selector(self) -> str:
        return self.physical_name


@dataclass(frozen=True)
class CompactOm4Variable:
    physical_name: str
    level: int | None

    def extension_selector(self) -> tuple[str, int | None]:
        return self.physical_name, self.level


type ReaderVariable = FlatOm4Variable | CompactOm4Variable


class _PinnedTensorPool:
    """Reuse pinned tensors after their CUDA consumer event has completed."""

    _MAX_FREE_TENSORS = 3

    def __init__(self) -> None:
        self._free: list[torch.Tensor] = []
        self._pending: deque[tuple[torch.cuda.Event, torch.Tensor]] = deque()
        self._lock = Lock()

    @staticmethod
    def _capacity(tensor: torch.Tensor) -> int:
        return tensor.untyped_storage().nbytes() // tensor.element_size()

    def _cache_locked(self, tensor: torch.Tensor) -> None:
        tensor.resize_((self._capacity(tensor),))
        self._free.append(tensor)
        if len(self._free) > self._MAX_FREE_TENSORS:
            smallest = min(
                range(len(self._free)),
                key=lambda index: self._capacity(self._free[index]),
            )
            self._free.pop(smallest)

    def _reclaim_locked(self) -> None:
        remaining: deque[tuple[torch.cuda.Event, torch.Tensor]] = deque()
        while self._pending:
            event, tensor = self._pending.popleft()
            if event.query():
                self._cache_locked(tensor)
            else:
                remaining.append((event, tensor))
        self._pending = remaining

    def acquire(self, shape: tuple[int, ...]) -> torch.Tensor:
        with self._lock:
            self._reclaim_locked()
            required = math.prod(shape)
            candidates = [
                (self._capacity(tensor), index)
                for index, tensor in enumerate(self._free)
                if self._capacity(tensor) >= required
            ]
            if candidates:
                _, index = min(candidates)
                tensor = self._free.pop(index)
                return tensor.resize_(shape)
            return torch.empty(shape, dtype=torch.float32, pin_memory=True)

    def release_tensors(
        self,
        tensors: list[torch.Tensor],
        event: torch.cuda.Event | None = None,
    ) -> None:
        with self._lock:
            if event is None:
                for tensor in tensors:
                    self._cache_locked(tensor)
            else:
                self._pending.extend((event, tensor) for tensor in tensors)

    def lease(self) -> _PinnedBufferLease:
        return _PinnedBufferLease(self)


class _PinnedBufferLease:
    """Own pinned buffers until their CUDA consumer has completed."""

    def __init__(self, pool: _PinnedTensorPool) -> None:
        self._pool = pool
        self._tensors: list[torch.Tensor] = []
        self._released = False

    def acquire(self, shape: tuple[int, ...]) -> torch.Tensor:
        if self._released:
            raise RuntimeError("Cannot acquire from a released pinned-buffer lease")
        tensor = self._pool.acquire(shape)
        self._tensors.append(tensor)
        return tensor

    def release(self, event: torch.cuda.Event | None = None) -> None:
        if self._released:
            return
        self._released = True
        self._pool.release_tensors(self._tensors, event)
        self._tensors = []


@dataclass(frozen=True)
class _NativeOm4CanonicalReader:
    """Canonical semantics with a private persistent native plane reader.

    The xarray delegate remains the source of coordinates, statistics, metadata,
    and ordinary reads. The Rust loader discovers only the narrow native methods
    below; physical layout and time indices never become CanonicalSource state.
    """

    semantic: CanonicalReader
    channels: tuple[str, ...]
    path: str
    _native: Any
    _reader_variables: dict[str, ReaderVariable]
    _physical_time_indices: np.ndarray
    _spatial_shape: tuple[int, int]

    @property
    def time(self) -> xr.DataArray:
        return self.semantic.time

    @property
    def resolution(self):
        return self.semantic.resolution

    def statistics(self, channels: tuple[str, ...]) -> ChannelStatistics:
        return self.semantic.statistics(channels)

    @property
    def attrs(self):
        return self.semantic.attrs

    @property
    def storage_id(self) -> int:
        return id(self._native)

    def slice_time(self, time) -> Self:
        semantic = self.semantic.slice_time(time)
        positions = self.time.to_index().get_indexer(semantic.time.to_index())
        if np.any(positions < 0):
            raise AssertionError("Canonical time slice could not be mapped to storage")
        physical = self._physical_time_indices[positions].copy()
        physical.setflags(write=False)
        return replace(
            self,
            semantic=semantic,
            _physical_time_indices=physical,
        )

    def read(self, request: CanonicalReadRequest) -> np.ndarray:
        # Keep the ordinary CanonicalSource path independent from the optimized
        # training loader. This also gives parity tests a true xarray reference.
        return self.semantic.read(request)

    def coordinates(self):
        return self.semantic.coordinates()

    def metadata(self, data_layout):
        return self.semantic.metadata(data_layout)

    def physical_indices(self, relative: np.ndarray) -> np.ndarray:
        return self._physical_time_indices[relative]

    def read_unique(
        self,
        physical_time_indices: np.ndarray,
        *,
        channels: tuple[str, ...],
        buffer_factory: Callable[[tuple[int, ...]], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Read unique physical planes for this canonical channel view."""
        if physical_time_indices.ndim != 1:
            raise ValueError("Unique physical time indices must be one-dimensional")
        missing = set(channels).difference(self._reader_variables)
        if missing:
            raise KeyError(f"Canonical channels not found: {sorted(missing)}")
        reader_variables = [
            self._reader_variables[name].extension_selector() for name in channels
        ]
        shape = (len(physical_time_indices), len(channels), *self._spatial_shape)
        tensor = (
            buffer_factory(shape)
            if buffer_factory is not None
            else torch.empty(shape, dtype=torch.float32)
        )
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            raise ValueError(
                f"Rust unique read buffer must have shape {shape} and dtype float32; "
                f"got shape {tuple(tensor.shape)} and dtype {tensor.dtype}"
            )
        if buffer_factory is not None and not tensor.is_pinned():
            raise ValueError("Rust buffer factory must return pinned memory")
        self._native.read_into(
            physical_time_indices.tolist(), reader_variables, tensor.numpy()
        )
        return tensor


def native_om4_source(
    dataset: CanonicalSource,
    location: LocalLocation,
    runtime: RustIoRuntime,
) -> CanonicalSource:
    """Attach native OM4 plane I/O without changing canonical dataset semantics."""
    physical = location.open({})
    reader_variables: dict[str, ReaderVariable] = {}
    uses_level_axis = False
    for logical_name in dataset.channels:
        if logical_name in physical.data_vars:
            reader_variables[logical_name] = FlatOm4Variable(logical_name)
            continue
        base, separator, level_text = logical_name.rpartition("_")
        if separator and level_text.isdigit() and base in physical.data_vars:
            variable = physical[base]
            if "lev" in variable.dims:
                reader_variables[logical_name] = CompactOm4Variable(
                    base, int(level_text)
                )
                uses_level_axis = True
                continue
        matches = []
        for physical_name in physical.data_vars:
            name = str(physical_name)
            if not name.startswith(f"{base}_lev_"):
                continue
            depth = float(name.split("_lev_", 1)[1].replace("_", "."))
            if dataset.data_layout.depth_levels.index(depth) == int(level_text):
                matches.append(name)
        if len(matches) == 1:
            reader_variables[logical_name] = FlatOm4Variable(matches[0])
            continue
        raise ValueError(
            f"Could not map canonical OM4 channel {logical_name!r} in {location.path}"
        )

    extension = _load_extension()
    unique = list(dict.fromkeys(reader_variables.values()))
    if uses_level_axis:
        selectors = [
            value
            if isinstance(value, CompactOm4Variable)
            else CompactOm4Variable(value.physical_name, None)
            for value in unique
        ]
        native = extension.CompactOm4Reader(
            location.path,
            [value.extension_selector() for value in selectors],
            runtime.read_pool,
        )
        reader_variables = {
            name: value
            if isinstance(value, CompactOm4Variable)
            else CompactOm4Variable(value.physical_name, None)
            for name, value in reader_variables.items()
        }
    else:
        if not all(isinstance(value, FlatOm4Variable) for value in unique):
            raise AssertionError("Flat OM4 mapping unexpectedly contains a level")
        native = extension.FlatOm4Reader(
            location.path,
            [value.physical_name for value in unique],
            runtime.read_pool,
        )

    physical_time = physical["time"].to_index()
    canonical_time = dataset.time.to_index()
    if not physical_time.is_unique:
        raise ValueError(f"Rust store {location.path} has duplicate time coordinates")
    if not canonical_time.is_unique:
        raise ValueError("Canonical dataset has duplicate time coordinates")
    physical_indices = physical_time.get_indexer(canonical_time).astype(
        np.int64, copy=False
    )
    if np.any(physical_indices < 0):
        missing = canonical_time[physical_indices < 0]
        raise ValueError(
            f"Canonical dataset times are missing from Rust store {location.path}: "
            f"{list(missing[:3])}"
        )

    time_size, lat, lon = native.shape
    if time_size != len(physical_time):
        raise ValueError(
            f"Rust store {location.path} reports {time_size} rows, but its time "
            f"coordinate has {len(physical_time)}"
        )
    if (lat, lon) != dataset.grid_size:
        raise ValueError(
            f"Rust store {location.path} has spatial shape {(lat, lon)}, but the "
            f"canonical dataset has {dataset.grid_size}"
        )
    physical_indices = physical_indices.copy()
    physical_indices.setflags(write=False)
    reader = _NativeOm4CanonicalReader(
        semantic=dataset.reader,
        channels=dataset.channels,
        path=str(location.path),
        _native=native,
        _reader_variables=reader_variables,
        _physical_time_indices=physical_indices,
        _spatial_shape=(lat, lon),
    )
    return dataset.with_reader(reader)


def _native_reader(source: CanonicalSource) -> _NativeOm4CanonicalReader:
    reader = source.reader
    if not isinstance(reader, _NativeOm4CanonicalReader):
        raise ValueError(
            "Rust training requires a canonical dataset built by the native OM4 "
            "backend factory"
        )
    return reader


@dataclass
class _RustChunkUse:
    group_index: int
    rows: torch.Tensor
    policy: BatchReadUse


@dataclass
class _RustChunkStep:
    input: _RustChunkUse
    boundary: _RustChunkUse
    label: _RustChunkUse


@dataclass
class _RustChunkBatch:
    groups: list[torch.Tensor]
    steps: list[_RustChunkStep]
    load_stats: LoadStats
    lease: _PinnedBufferLease | None


@dataclass
class _PendingChunkUse:
    key: tuple[int, tuple[str, ...]]
    physical_indices: np.ndarray
    policy: BatchReadUse


@dataclass
class _PendingChunkStep:
    input: _PendingChunkUse
    boundary: _PendingChunkUse
    label: _PendingChunkUse


class _RustBatchDataset:
    """Batch-oriented native reader and preparer for one training shard."""

    def __init__(
        self,
        shard: TrainingShard,
    ) -> None:
        self.shard = shard
        self.preparer = TrainBatchPreparer(shard)
        self._input_reader = _native_reader(self.shard.input_source)
        self._boundary_reader = self._input_reader
        self._label_reader = _native_reader(self.shard.label_source)

    def load_chunk_batch(
        self,
        indices: list[int],
        *,
        buffer_pool: _PinnedTensorPool | None = None,
    ) -> _RustChunkBatch:
        """Load every physical plane once and retain logical rollout maps."""
        if not indices:
            raise ValueError("Cannot load an empty Rust batch")
        start_time = time.perf_counter()
        lease = buffer_pool.lease() if buffer_pool is not None else None
        group_specs: dict[tuple[int, tuple[str, ...]], _NativeOm4CanonicalReader] = {}
        pending_steps: list[_PendingChunkStep] = []

        def buffer_factory(shape: tuple[int, ...]) -> torch.Tensor:
            assert lease is not None
            return lease.acquire(shape)

        def pending_use(
            reader: _NativeOm4CanonicalReader,
            use: BatchReadUse,
        ) -> _PendingChunkUse:
            key = (reader.storage_id, use.request.channels)
            group_specs.setdefault(key, reader)
            return _PendingChunkUse(
                key=key,
                physical_indices=reader.physical_indices(use.request.time_indices),
                policy=use,
            )

        try:
            plan = self.shard.window_plan(indices)
            for step in plan.steps:
                pending_steps.append(
                    _PendingChunkStep(
                        input=pending_use(self._input_reader, step.input),
                        boundary=pending_use(self._boundary_reader, step.boundary),
                        label=pending_use(self._label_reader, step.label),
                    )
                )

            uses_by_key: dict[tuple[int, tuple[str, ...]], list[_PendingChunkUse]] = {
                key: [] for key in group_specs
            }
            for pending_step in pending_steps:
                for use in (
                    pending_step.input,
                    pending_step.boundary,
                    pending_step.label,
                ):
                    uses_by_key[use.key].append(use)

            groups: list[torch.Tensor] = []
            group_metadata: dict[
                tuple[int, tuple[str, ...]], tuple[int, np.ndarray]
            ] = {}
            for key, reader in group_specs.items():
                unique_indices = np.unique(
                    np.concatenate(
                        [use.physical_indices.reshape(-1) for use in uses_by_key[key]]
                    )
                ).astype(np.int64, copy=False)
                values = reader.read_unique(
                    unique_indices,
                    channels=key[1],
                    buffer_factory=buffer_factory if buffer_pool else None,
                )
                group_metadata[key] = (len(groups), unique_indices)
                groups.append(values)

            def finalize(use: _PendingChunkUse) -> _RustChunkUse:
                group_index, unique_indices = group_metadata[use.key]
                rows = np.searchsorted(unique_indices, use.physical_indices)
                if not np.array_equal(unique_indices[rows], use.physical_indices):
                    raise AssertionError("Rust chunk plan lost a physical time index")
                return _RustChunkUse(
                    group_index=group_index,
                    rows=torch.from_numpy(rows.astype(np.int64, copy=False)),
                    policy=use.policy,
                )

            steps = [
                _RustChunkStep(
                    input=finalize(step.input),
                    boundary=finalize(step.boundary),
                    label=finalize(step.label),
                )
                for step in pending_steps
            ]
        except BaseException:
            if lease is not None:
                lease.release()
            raise

        return _RustChunkBatch(
            groups=groups,
            steps=steps,
            load_stats=LoadStats(time.perf_counter() - start_time),
            lease=lease,
        )


class RustTrainDataLoader:
    """Batch-sampler preserving loader with bounded Rust host prefetch."""

    def __init__(
        self,
        shards: list[TrainingShard],
        batch_sampler: BatchSampler,
        device: torch.device,
        *,
        prefetch_batches: int,
        prefetch: RustPrefetchPolicy,
    ) -> None:
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches must be positive")
        if not shards:
            raise ValueError("RustTrainDataLoader requires at least one dataset")
        if not hasattr(batch_sampler, "__iter__") or not hasattr(
            batch_sampler, "__len__"
        ):
            raise TypeError("batch_sampler must be iterable and sized")

        self._shards = shards
        self._batch_datasets = [_RustBatchDataset(shard) for shard in shards]
        self._batch_sampler = batch_sampler
        self._device = device
        self._prefetch_batches = prefetch_batches
        self._active_iterator: (
            weakref.ReferenceType[_PreparedIterator | _CudaPrefetchIterator] | None
        ) = None
        self._prefetch_to_device = isinstance(prefetch, CudaPrefetch)
        if self._prefetch_to_device and device.type != "cuda":
            raise ValueError("CUDA prefetch requires a CUDA training device")
        pin_memory = True if isinstance(prefetch, CudaPrefetch) else prefetch.pin_memory
        self._pinned_pool = _PinnedTensorPool() if pin_memory else None
        self._cumulative_sizes = np.cumsum([len(shard) for shard in shards]).tolist()

    def _resolve_batch(self, global_indices: list[int]) -> tuple[int, list[int]]:
        if not global_indices:
            raise ValueError("The batch sampler emitted an empty batch")

        resolved: list[tuple[int, int]] = []
        for global_index in global_indices:
            if global_index < 0 or global_index >= self._cumulative_sizes[-1]:
                raise IndexError(
                    f"Global dataset index {global_index} is out of range for "
                    f"length {self._cumulative_sizes[-1]}"
                )
            dataset_index = bisect_right(self._cumulative_sizes, global_index)
            dataset_start = (
                0 if dataset_index == 0 else self._cumulative_sizes[dataset_index - 1]
            )
            resolved.append((dataset_index, global_index - dataset_start))

        dataset_indices = {dataset_index for dataset_index, _ in resolved}
        if len(dataset_indices) != 1:
            # Preserve collate_raw_train_data's existing dataset_id invariant.
            raise AssertionError("we don't support heterogenous batches yet")
        dataset_index = resolved[0][0]
        return dataset_index, [local_index for _, local_index in resolved]

    def _load_raw_batch(
        self, global_indices: list[int]
    ) -> tuple[_RustBatchDataset, _RustChunkBatch]:
        dataset_index, local_indices = self._resolve_batch(global_indices)
        batch_dataset = self._batch_datasets[dataset_index]
        raw = batch_dataset.load_chunk_batch(
            local_indices,
            buffer_pool=self._pinned_pool,
        )
        return batch_dataset, raw

    def _prepare_batch(
        self, loaded: tuple[_RustBatchDataset, _RustChunkBatch]
    ) -> ModelBatch:
        batch_dataset, raw = loaded
        preparer = batch_dataset.preparer
        device_groups = [
            group.to(device=self._device, non_blocking=True) for group in raw.groups
        ]
        transformed: dict[tuple[int, int, int], torch.Tensor] = {}

        def materialize(use: _RustChunkUse) -> torch.Tensor:
            transform_key = (
                use.group_index,
                id(use.policy.source),
                id(use.policy.mask),
            )
            values = transformed.get(transform_key)
            if values is None:
                values = preparer.normalize_and_mask_device_planes(
                    use.policy,
                    device_groups[use.group_index],
                    self._device,
                )
                transformed[transform_key] = values
            rows = use.rows.to(device=self._device, non_blocking=True)
            batch_size, time_size = rows.shape
            selected = values.index_select(0, rows.reshape(-1)).reshape(
                batch_size,
                time_size,
                values.shape[1],
                values.shape[2],
                values.shape[3],
            )
            return selected.flatten(1, 2)

        train_data = preparer.new_model_batch(self._device)
        for step in raw.steps:
            train_data.append(
                materialize(step.input),
                materialize(step.boundary),
                materialize(step.label),
            )
        train_data.load_stats = raw.load_stats
        return train_data

    def _release_raw(
        self, raw: _RustChunkBatch, event: torch.cuda.Event | None = None
    ) -> None:
        if raw.lease is not None:
            raw.lease.release(event)

    def __iter__(self) -> Iterator[ModelBatch]:
        self.close()
        # Snapshot sampler RNG and rank scheduling before the producer starts.
        schedule = [list(batch) for batch in self._batch_sampler]
        host_iterator = _RustHostPrefetchIterator(self, schedule)
        if self._prefetch_to_device:
            iterator: _PreparedIterator | _CudaPrefetchIterator = _CudaPrefetchIterator(
                self, host_iterator
            )
        else:
            iterator = _PreparedIterator(self, host_iterator)
        # The iterator owns the loader operations it needs, while the loader only
        # tracks it weakly so abandoning a partial iteration can finalize and
        # stop the host executor immediately.
        self._active_iterator = weakref.ref(iterator)
        return iterator

    def __len__(self) -> int:
        return len(self._batch_sampler)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._batch_sampler, "set_epoch"):
            self._batch_sampler.set_epoch(epoch)

    def close(self) -> None:
        active_iterator = self._active_iterator
        self._active_iterator = None
        if active_iterator is not None:
            iterator = active_iterator()
            if iterator is not None:
                iterator.close()


class _RustHostPrefetchIterator(Iterator[tuple[_RustBatchDataset, _RustChunkBatch]]):
    def __init__(self, loader: RustTrainDataLoader, schedule: list[list[int]]) -> None:
        self._loader = loader
        self._schedule = iter(schedule)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="samudra-rust-prefetch"
        )
        self._pending: deque[Future[tuple[_RustBatchDataset, _RustChunkBatch]]] = (
            deque()
        )
        self._closed = False
        self._fill()

    def _fill(self) -> None:
        while len(self._pending) < self._loader._prefetch_batches:
            try:
                batch = next(self._schedule)
            except StopIteration:
                break
            self._pending.append(
                self._executor.submit(self._loader._load_raw_batch, batch)
            )

    def __next__(self) -> tuple[_RustBatchDataset, _RustChunkBatch]:
        if self._closed or not self._pending:
            self.close()
            raise StopIteration
        future = self._pending.popleft()
        try:
            loaded = future.result()
            self._fill()
            return loaded
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        pending = list(self._pending)
        self._pending.clear()
        for future in pending:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        for future in pending:
            if future.cancelled():
                continue
            try:
                loaded = future.result()
            except BaseException:
                continue
            self._loader._release_raw(loaded[1])

    def __del__(self) -> None:
        self.close()


class _PreparedIterator(Iterator[ModelBatch]):
    def __init__(
        self, loader: RustTrainDataLoader, host: _RustHostPrefetchIterator
    ) -> None:
        self._loader = loader
        self._host = host

    def __next__(self) -> ModelBatch:
        loaded = next(self._host)
        event: torch.cuda.Event | None = None
        try:
            return self._loader._prepare_batch(loaded)
        finally:
            if self._loader._device.type == "cuda":
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self._loader._device))
            self._loader._release_raw(loaded[1], event)

    def close(self) -> None:
        self._host.close()

    def __del__(self) -> None:
        self.close()


class _CudaPrefetchIterator(Iterator[ModelBatch]):
    """Prepare one batch ahead on a dedicated PyTorch CUDA stream."""

    def __init__(
        self, loader: RustTrainDataLoader, host: _RustHostPrefetchIterator
    ) -> None:
        self._loader = loader
        self._host = host
        self._stream = torch.cuda.Stream(device=loader._device)
        self._next_data: ModelBatch | None = None
        self._next_event: torch.cuda.Event | None = None
        self._closed = False
        self._preload()

    def _preload(self) -> None:
        try:
            loaded = next(self._host)
        except StopIteration:
            self._next_data = None
            self._next_event = None
            return

        with torch.cuda.stream(self._stream):
            try:
                self._next_data = self._loader._prepare_batch(loaded)
            finally:
                event = torch.cuda.Event()
                event.record(self._stream)
                self._loader._release_raw(loaded[1], event)
            self._next_event = event

    def __next__(self) -> ModelBatch:
        if self._closed or self._next_data is None or self._next_event is None:
            self.close()
            raise StopIteration

        current_stream = torch.cuda.current_stream(self._loader._device)
        current_stream.wait_event(self._next_event)
        data = self._next_data
        data.record_stream(current_stream)
        self._preload()
        return data

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._host.close()
        self._next_data = None
        self._next_event = None

    def __del__(self) -> None:
        self.close()
