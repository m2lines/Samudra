# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Local flat-OM4 batch loading through the optional Rust extension."""

from __future__ import annotations

import importlib
import time
from bisect import bisect_right
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from samudra.constants import DatasetSpec
from samudra.datasets import RawTrainData, TorchTrainDataset, TrainData
from samudra.utils.data import DataSource, LoadStats
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


def create_rust_read_pool(max_concurrent_reads: int) -> Any:
    """Create the one bounded native read pool shared by a training process."""
    return _load_extension().FlatOm4ReadPool(max_concurrent_reads)


class BatchSampler(Protocol):
    def __iter__(self) -> Iterator[list[int]]: ...

    def __len__(self) -> int: ...


def _encode_om4_depth(depth: float) -> str:
    return str(depth).replace(".", "_")


def _flat_om4_variable_candidates(
    logical_name: str, dataset_spec: DatasetSpec
) -> tuple[str, ...]:
    """Return supported physical names for one canonical flat-OM4 variable."""
    base_name, separator, level_text = logical_name.rpartition("_")
    if not separator or not level_text.isdigit():
        return (logical_name,)

    level_index = int(level_text)
    if level_index >= len(dataset_spec.depth_levels):
        raise ValueError(
            f"OM4 variable {logical_name!r} selects level {level_index}, but only "
            f"{len(dataset_spec.depth_levels)} levels are configured"
        )
    depth = _encode_om4_depth(dataset_spec.depth_levels[level_index])
    # Some stores are already canonical while older flat stores encode depth values.
    return logical_name, f"{base_name}_lev_{depth}"


def _array_metadata_exists(root: Path, variable: str) -> bool:
    array_path = root / variable
    return (array_path / ".zarray").is_file() or (array_path / "zarr.json").is_file()


class FlatOm4Store:
    """Logical-variable facade over one persistent Rust flat-OM4 reader."""

    def __init__(
        self,
        source: DataSource,
        logical_variables: list[str],
        read_pool: Any,
    ) -> None:
        if source.is_compact:
            raise ValueError("loading.type='rust' does not yet support OM4-compact")
        if not isinstance(source.data_location, LocalLocation):
            raise ValueError(
                "loading.type='rust' currently requires a local data location"
            )
        if not logical_variables:
            raise ValueError("A Rust OM4 store requires at least one variable")

        self.path = source.data_location.path
        self._physical_by_logical: dict[str, str] = {}
        for logical_name in dict.fromkeys(logical_variables):
            candidates = _flat_om4_variable_candidates(
                logical_name, source.dataset_spec
            )
            physical_name = next(
                (
                    candidate
                    for candidate in candidates
                    if _array_metadata_exists(self.path, candidate)
                ),
                None,
            )
            if physical_name is None:
                expected = ", ".join(repr(candidate) for candidate in candidates)
                raise ValueError(
                    f"Could not find flat-OM4 variable {logical_name!r} in "
                    f"{self.path}; tried {expected}"
                )
            self._physical_by_logical[logical_name] = physical_name

        extension = _load_extension()
        self._reader: Any = extension.FlatOm4Reader(
            self.path,
            list(dict.fromkeys(self._physical_by_logical.values())),
            read_pool,
        )
        _, physical_lat, physical_lon = self._reader.shape
        self._spatial_shape = (physical_lat, physical_lon)
        if (physical_lat, physical_lon) != source.grid_size:
            raise ValueError(
                f"Rust store {self.path} has spatial shape "
                f"{(physical_lat, physical_lon)}, but the DataSource has "
                f"{source.grid_size}; derived/in-memory grids are unsupported"
            )

    def read(
        self,
        physical_time_indices: np.ndarray,
        logical_variables: list[str],
        *,
        pin_memory: bool,
    ) -> torch.Tensor:
        if physical_time_indices.ndim != 2:
            raise ValueError("Batch time indices must have shape (batch, time)")
        physical_variables = [
            self._physical_by_logical[variable] for variable in logical_variables
        ]
        batch_size, time_size = physical_time_indices.shape
        shape = (
            batch_size,
            time_size,
            len(logical_variables),
            *self._spatial_shape,
        )
        tensor = torch.empty(shape, dtype=torch.float32, pin_memory=pin_memory)
        self._reader.read_into(
            physical_time_indices.reshape(-1).tolist(),
            physical_variables,
            tensor.reshape(
                batch_size * time_size,
                len(logical_variables),
                *self._spatial_shape,
            ).numpy(),
        )
        return tensor


class RustBatchDataset:
    """Batch-oriented Rust reader for one existing ``TorchTrainDataset``."""

    def __init__(
        self,
        dataset: TorchTrainDataset,
        *,
        max_concurrent_reads: int,
        read_pool: Any | None = None,
    ) -> None:
        self.dataset = dataset
        self.read_pool = read_pool or create_rust_read_pool(max_concurrent_reads)
        input_source = dataset.prognostic_srcs[0]
        label_source = dataset.prognostic_srcs[-1]
        self._prognostic_variables = list(input_source.data.data_vars)
        self._boundary_variables = list(dataset.boundary_src.data.data_vars)

        self._input_store = FlatOm4Store(
            input_source,
            self._prognostic_variables + self._boundary_variables,
            self.read_pool,
        )
        if (
            isinstance(label_source.data_location, LocalLocation)
            and label_source.data_location.path == self._input_store.path
        ):
            if label_source.grid_size != input_source.grid_size:
                raise ValueError(
                    "Input and label DataSources share one physical store but have "
                    "different grids; derived/in-memory grids are unsupported"
                )
            self._label_store = self._input_store
        else:
            self._label_store = FlatOm4Store(
                label_source,
                self._prognostic_variables,
                self.read_pool,
            )

    @staticmethod
    def _physical_indices(source: DataSource, relative: np.ndarray) -> np.ndarray:
        if source.physical_time_indices is None:
            raise ValueError("DataSource is missing physical time-index metadata")
        return source.physical_time_indices[relative]

    def load_batch(
        self, indices: list[int], *, pin_memory: bool = False
    ) -> RawTrainData:
        if not indices:
            raise ValueError("Cannot load an empty Rust batch")
        start_time = time.perf_counter()
        raw = RawTrainData(self.dataset.id)

        for step in range(self.dataset.steps):
            relative = np.stack(
                [self.dataset._get_x_index(index, step).to_numpy() for index in indices]
            ).astype(np.int64, copy=False)
            current = relative[:, : self.dataset.hist + 1]
            forecast = relative[:, self.dataset.hist + 1 :]

            input_indices = self._physical_indices(
                self.dataset.prognostic_srcs[0], current
            )
            boundary_indices = self._physical_indices(
                self.dataset.boundary_src, current
            )
            label_indices = self._physical_indices(
                self.dataset.prognostic_srcs[-1], forecast
            )

            input_ = self._input_store.read(
                input_indices,
                self._prognostic_variables,
                pin_memory=pin_memory,
            )
            boundary = self._input_store.read(
                boundary_indices,
                self._boundary_variables,
                pin_memory=pin_memory,
            )
            label = self._label_store.read(
                label_indices,
                self._prognostic_variables,
                pin_memory=pin_memory,
            )
            raw.insert(input_, boundary, label)

        raw.load_stats = LoadStats(time.perf_counter() - start_time)
        return raw


class RustTrainDataLoader:
    """Batch-sampler preserving loader with bounded Rust host prefetch."""

    def __init__(
        self,
        datasets: list[TorchTrainDataset],
        batch_sampler: BatchSampler,
        device: torch.device,
        *,
        max_concurrent_reads: int,
        prefetch_batches: int,
        pin_memory: bool,
        prefetch_to_device: bool = False,
        read_pool: Any | None = None,
    ) -> None:
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches must be positive")
        if not datasets:
            raise ValueError("RustTrainDataLoader requires at least one dataset")
        if not hasattr(batch_sampler, "__iter__") or not hasattr(
            batch_sampler, "__len__"
        ):
            raise TypeError("batch_sampler must be iterable and sized")

        self._datasets = datasets
        self._read_pool = read_pool or create_rust_read_pool(max_concurrent_reads)
        self._batch_datasets = [
            RustBatchDataset(
                dataset,
                max_concurrent_reads=max_concurrent_reads,
                read_pool=self._read_pool,
            )
            for dataset in datasets
        ]
        self._batch_sampler = batch_sampler
        self._device = device
        self._prefetch_batches = prefetch_batches
        self._pin_memory = pin_memory
        self._prefetch_to_device = prefetch_to_device and device.type == "cuda"
        if self._prefetch_to_device and not pin_memory:
            raise ValueError("CUDA device prefetch requires pinned host memory")
        self._concat_dataset: ConcatDataset[RawTrainData] = ConcatDataset(datasets)
        self._cumulative_sizes = self._concat_dataset.cumulative_sizes

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
    ) -> tuple[TorchTrainDataset, RawTrainData]:
        dataset_index, local_indices = self._resolve_batch(global_indices)
        raw = self._batch_datasets[dataset_index].load_batch(
            local_indices, pin_memory=self._pin_memory
        )
        return self._datasets[dataset_index], raw

    def _prepare_batch(
        self, loaded: tuple[TorchTrainDataset, RawTrainData]
    ) -> TrainData:
        dataset, raw = loaded
        return dataset.to_train_data(raw, self._device)

    def __iter__(self) -> Iterator[TrainData]:
        # Snapshot sampler RNG and rank scheduling before the producer starts.
        schedule = [list(batch) for batch in self._batch_sampler]
        host_iterator = _RustHostPrefetchIterator(self, schedule)
        if self._prefetch_to_device:
            return _CudaPrefetchIterator(self, host_iterator)
        return _PreparedIterator(self, host_iterator)

    def __len__(self) -> int:
        return len(self._batch_sampler)

    def __getitem__(self, index: int):
        loaded = self._load_raw_batch([index])
        return self._prepare_batch(loaded)

    @property
    def dataset(self):
        return self._concat_dataset

    @property
    def sampler(self):
        return self._batch_sampler


class _RustHostPrefetchIterator(Iterator[tuple[TorchTrainDataset, RawTrainData]]):
    def __init__(self, loader: RustTrainDataLoader, schedule: list[list[int]]) -> None:
        self._loader = loader
        self._schedule = iter(schedule)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="samudra-rust-prefetch"
        )
        self._pending: deque[Future[tuple[TorchTrainDataset, RawTrainData]]] = deque()
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

    def __next__(self) -> tuple[TorchTrainDataset, RawTrainData]:
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
        for future in self._pending:
            future.cancel()
        self._pending.clear()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __del__(self) -> None:
        self.close()


class _PreparedIterator(Iterator[TrainData]):
    def __init__(
        self, loader: RustTrainDataLoader, host: _RustHostPrefetchIterator
    ) -> None:
        self._loader = loader
        self._host = host

    def __next__(self) -> TrainData:
        return self._loader._prepare_batch(next(self._host))

    def close(self) -> None:
        self._host.close()


def _record_train_data_stream(data: TrainData, stream: torch.cuda.Stream) -> None:
    data.ctx.label_mask.record_stream(stream)
    for prognostic, boundary, label in data.example_by_step:
        prognostic.record_stream(stream)
        boundary.record_stream(stream)
        label.record_stream(stream)


class _CudaPrefetchIterator(Iterator[TrainData]):
    """Prepare one batch ahead on a dedicated PyTorch CUDA stream."""

    def __init__(
        self, loader: RustTrainDataLoader, host: _RustHostPrefetchIterator
    ) -> None:
        self._loader = loader
        self._host = host
        self._stream = torch.cuda.Stream(device=loader._device)
        self._next_data: TrainData | None = None
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
            self._next_data = self._loader._prepare_batch(loaded)
            self._next_event = torch.cuda.Event()
            self._next_event.record(self._stream)

    def __next__(self) -> TrainData:
        if self._closed or self._next_data is None or self._next_event is None:
            self.close()
            raise StopIteration

        current_stream = torch.cuda.current_stream(self._loader._device)
        current_stream.wait_event(self._next_event)
        data = self._next_data
        _record_train_data_stream(data, current_stream)
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
