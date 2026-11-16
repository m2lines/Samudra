# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared native batching, host prefetch and CUDA buffer lifetimes."""

from __future__ import annotations

import math
import time
import weakref
from bisect import bisect_right
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import numpy as np
import torch

from samudra.constants import TimeIndices
from samudra.datasets import BatchPreparer, BatchReadUse, ModelBatch, TrainingWindows
from samudra.utils.data import BulkCanonicalReader, LoadStats
from samudra.utils.samplers import BatchSchedule


@dataclass(frozen=True)
class HostPrefetch:
    """Read ahead into RAM; prepare on the consumer's CPU or CUDA stream."""


@dataclass(frozen=True)
class CudaPrefetch:
    """Prefetch through pinned host buffers onto a dedicated CUDA stream."""


type PrefetchPolicy = HostPrefetch | CudaPrefetch


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


@dataclass
class _ChunkUse:
    group_index: int
    rows: torch.Tensor
    policy: BatchReadUse


@dataclass
class _ChunkStep:
    input: _ChunkUse
    boundary: _ChunkUse
    label: _ChunkUse


@dataclass
class _ChunkBatch:
    groups: list[torch.Tensor]
    steps: list[_ChunkStep]
    load_stats: LoadStats
    lease: _PinnedBufferLease | None


@dataclass
class _PendingChunkUse:
    key: tuple[int, tuple[str, ...]]
    time_indices: TimeIndices
    policy: BatchReadUse


@dataclass
class _PendingChunkStep:
    input: _PendingChunkUse
    boundary: _PendingChunkUse
    label: _PendingChunkUse


class _NativeBatchReader:
    """Batch-oriented native reader and preparer for one set of training windows."""

    def __init__(
        self,
        windows: TrainingWindows,
    ) -> None:
        self.windows = windows
        self.preparer = BatchPreparer(windows)
        self._input_reader = self.windows.input_source.bulk_reader
        self._label_reader = self.windows.label_source.bulk_reader

    def load_chunk_batch(
        self,
        indices: list[int],
        *,
        buffer_pool: _PinnedTensorPool | None = None,
    ) -> _ChunkBatch:
        """Load every physical plane once and retain logical rollout maps."""
        if not indices:
            raise ValueError("Cannot load an empty native batch")
        start_time = time.perf_counter()
        lease = buffer_pool.lease() if buffer_pool is not None else None
        group_specs: dict[tuple[int, tuple[str, ...]], BulkCanonicalReader] = {}
        pending_steps: list[_PendingChunkStep] = []

        def pending_use(
            reader: BulkCanonicalReader,
            use: BatchReadUse,
        ) -> _PendingChunkUse:
            # TrainingWindows requires input and label time axes to match, so
            # source-relative positions are comparable within each read group.
            key = (reader.storage_id, use.request.channels)
            group_specs.setdefault(key, reader)
            return _PendingChunkUse(
                key=key,
                time_indices=use.request.time_indices,
                policy=use,
            )

        try:
            plan = self.windows.window_plan(indices)
            for step in plan.steps:
                pending_steps.append(
                    _PendingChunkStep(
                        input=pending_use(self._input_reader, step.input),
                        boundary=pending_use(self._input_reader, step.boundary),
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
                tuple[int, tuple[str, ...]], tuple[int, TimeIndices]
            ] = {}
            for key, reader in group_specs.items():
                unique_indices = np.unique(
                    np.concatenate(
                        [use.time_indices.reshape(-1) for use in uses_by_key[key]]
                    )
                ).astype(np.int64, copy=False)
                shape = (len(unique_indices), len(key[1]), *reader.spatial_shape)
                values = (
                    lease.acquire(shape)
                    if lease is not None
                    else torch.empty(shape, dtype=torch.float32)
                )
                reader.read_into(unique_indices, key[1], values.numpy())
                group_metadata[key] = (len(groups), unique_indices)
                groups.append(values)

            def finalize(use: _PendingChunkUse) -> _ChunkUse:
                group_index, unique_indices = group_metadata[use.key]
                rows = np.searchsorted(unique_indices, use.time_indices)
                if not np.array_equal(unique_indices[rows], use.time_indices):
                    raise AssertionError(
                        "Native chunk plan lost a canonical time index"
                    )
                return _ChunkUse(
                    group_index=group_index,
                    rows=torch.from_numpy(rows.astype(np.int64, copy=False)),
                    policy=use.policy,
                )

            steps = [
                _ChunkStep(
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

        return _ChunkBatch(
            groups=groups,
            steps=steps,
            load_stats=LoadStats(time.perf_counter() - start_time),
            lease=lease,
        )


class NativeBatchLoader:
    """Batch-sampler preserving loader with bounded native host prefetch."""

    def __init__(
        self,
        windows: list[TrainingWindows],
        batch_sampler: BatchSchedule,
        device: torch.device,
        *,
        prefetch_batches: int,
        prefetch: PrefetchPolicy,
    ) -> None:
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches must be positive")
        if not windows:
            raise ValueError("NativeBatchLoader requires at least one dataset")
        if not hasattr(batch_sampler, "__iter__") or not hasattr(
            batch_sampler, "__len__"
        ):
            raise TypeError("batch_sampler must be iterable and sized")

        self._batch_readers = [_NativeBatchReader(window) for window in windows]
        self._batch_sampler = batch_sampler
        self._device = device
        self._prefetch_batches = prefetch_batches
        self._active_iterator: (
            weakref.ReferenceType[_PreparedIterator | _CudaPrefetchIterator] | None
        ) = None
        self._prefetch_to_device = isinstance(prefetch, CudaPrefetch)
        if self._prefetch_to_device and device.type != "cuda":
            raise ValueError("CUDA prefetch requires a CUDA training device")
        self._pinned_pool = _PinnedTensorPool() if device.type == "cuda" else None
        self._cumulative_sizes = np.cumsum([len(window) for window in windows]).tolist()

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
            # Preserve HostBatch collation's existing dataset_id invariant.
            raise AssertionError("we don't support heterogenous batches yet")
        dataset_index = resolved[0][0]
        return dataset_index, [local_index for _, local_index in resolved]

    def _load_raw_batch(
        self, global_indices: list[int]
    ) -> tuple[_NativeBatchReader, _ChunkBatch]:
        dataset_index, local_indices = self._resolve_batch(global_indices)
        batch_reader = self._batch_readers[dataset_index]
        raw = batch_reader.load_chunk_batch(
            local_indices,
            buffer_pool=self._pinned_pool,
        )
        return batch_reader, raw

    def _prepare_batch(
        self, loaded: tuple[_NativeBatchReader, _ChunkBatch]
    ) -> ModelBatch:
        batch_reader, raw = loaded
        preparer = batch_reader.preparer
        device_groups = [
            group.to(device=self._device, non_blocking=True) for group in raw.groups
        ]
        transformed: dict[tuple[int, int, int], torch.Tensor] = {}

        def materialize(use: _ChunkUse) -> torch.Tensor:
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
        self, raw: _ChunkBatch, event: torch.cuda.Event | None = None
    ) -> None:
        if raw.lease is not None:
            raw.lease.release(event)

    def __iter__(self) -> Iterator[ModelBatch]:
        self.close()
        # Snapshot sampler RNG and rank scheduling before the producer starts.
        schedule = [list(batch) for batch in self._batch_sampler]
        host_iterator = _HostPrefetchIterator(self, schedule)
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


class _HostPrefetchIterator(Iterator[tuple[_NativeBatchReader, _ChunkBatch]]):
    def __init__(self, loader: NativeBatchLoader, schedule: list[list[int]]) -> None:
        self._loader = loader
        self._schedule = iter(schedule)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="samudra-native-prefetch"
        )
        self._pending: deque[Future[tuple[_NativeBatchReader, _ChunkBatch]]] = deque()
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

    def __next__(self) -> tuple[_NativeBatchReader, _ChunkBatch]:
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
    def __init__(self, loader: NativeBatchLoader, host: _HostPrefetchIterator) -> None:
        self._loader = loader
        self._host = host

    def __next__(self) -> ModelBatch:
        try:
            loaded = next(self._host)
            try:
                return self._loader._prepare_batch(loaded)
            finally:
                event: torch.cuda.Event | None = None
                if self._loader._device.type == "cuda":
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(self._loader._device))
                self._loader._release_raw(loaded[1], event)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._host.close()

    def __del__(self) -> None:
        self.close()


class _CudaPrefetchIterator(Iterator[ModelBatch]):
    """Prepare one batch ahead on a dedicated PyTorch CUDA stream."""

    def __init__(self, loader: NativeBatchLoader, host: _HostPrefetchIterator) -> None:
        self._loader = loader
        self._host = host
        self._next_data: ModelBatch | None = None
        self._next_event: torch.cuda.Event | None = None
        self._closed = False
        try:
            self._stream = torch.cuda.Stream(device=loader._device)
            self._preload()
        except BaseException:
            self.close()
            raise

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

        try:
            current_stream = torch.cuda.current_stream(self._loader._device)
            current_stream.wait_event(self._next_event)
            data = self._next_data
            data.record_stream(current_stream)
            self._preload()
            return data
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._host.close()
        self._next_data = None
        self._next_event = None

    def __del__(self) -> None:
        self.close()
