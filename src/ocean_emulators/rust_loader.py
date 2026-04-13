from __future__ import annotations

import bisect
import dataclasses
import importlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import xarray as xr

if TYPE_CHECKING:
    from ocean_emulators.datasets import TorchTrainDataset

_tide_module: Any | None = None
_TIDE_IMPORT_ERROR: ImportError | None = None
try:
    _tide_module = importlib.import_module("tide")
except ImportError as exc:  # pragma: no cover - exercised via runtime error path
    _TIDE_IMPORT_ERROR = exc

type TensorPlacement = Literal["cpu", "torch_device"]


def _require_tide() -> Any:
    if _tide_module is None:
        raise RuntimeError(
            "The optional Rust loader extension `tide` is not installed. "
            "Build it with `uvx maturin develop --uv --manifest-path "
            "rust/tide/Cargo.toml`."
        ) from _TIDE_IMPORT_ERROR
    return _tide_module


def _as_local_path(path: str | None, *, label: str) -> str:
    if path is None:
        raise NotImplementedError(f"{label} does not expose a filesystem path for tide")

    resolved = Path(path)
    if not resolved.is_absolute():
        raise NotImplementedError(
            f"{label} must be backed by an absolute local path, got {path!r}"
        )
    return str(resolved)


@dataclasses.dataclass(frozen=True)
class ExampleSpec:
    input_time_indices: list[list[int]]
    label_time_indices: list[list[int]]


class TideDatasetHandle:
    def __init__(
        self,
        dataset: TorchTrainDataset,
    ) -> None:
        tide_mod = _require_tide()
        if dataset.prognostic_srcs[0].is_compact:
            raise NotImplementedError("tide v0 does not support compact datasets")
        if len(dataset.prognostic_srcs) != 1:
            raise NotImplementedError(
                "tide v0 only supports standard single-source training datasets"
            )

        self.dataset = dataset
        data_path = _as_local_path(
            dataset.prognostic_srcs[0].data_path, label="data source"
        )
        self._time_index = self._build_time_index(
            data_path,
            dataset.prognostic_srcs[0].data.time.values,
        )
        self.backend = tide_mod.Dataset(
            data_path,
            list(dataset.prognostic_srcs[0].data.data_vars),
            list(dataset.boundary_src.data.data_vars),
        )

    @staticmethod
    def _build_time_index(data_path: str, sliced_times: np.ndarray) -> np.ndarray:
        full_times = np.asarray(xr.open_zarr(data_path, chunks=None).time.values)
        if sliced_times.size == 0:
            return np.asarray([], dtype=np.int64)

        matches = np.flatnonzero(full_times == sliced_times[0])
        if matches.size == 0:
            raise ValueError("Could not locate sliced time range in tide backing store")

        offset = int(matches[0])
        end = offset + sliced_times.size
        if end <= full_times.size and np.array_equal(
            full_times[offset:end], sliced_times
        ):
            return np.arange(offset, end, dtype=np.int64)

        lookup = {time: index for index, time in enumerate(full_times)}
        return np.asarray([lookup[time] for time in sliced_times], dtype=np.int64)

    def make_batch(
        self,
        local_indices: Sequence[int],
        device: torch.device,
    ) -> RustTrainBatch:
        examples = [self._example_spec(idx) for idx in local_indices]
        rust_batch = self.backend.open_batch(
            [example.input_time_indices for example in examples],
            [example.label_time_indices for example in examples],
        )
        return RustTrainBatch(
            rust_batch=rust_batch,
            device=device,
        )

    def _example_spec(self, idx: int) -> ExampleSpec:
        input_time_indices = []
        label_time_indices = []
        for step in range(self.dataset.steps):
            x_index = self.dataset._get_x_index(idx, step)
            input_times = x_index.isel(time=slice(0, self.dataset.hist + 1)).values
            label_times = x_index.isel(time=slice(self.dataset.hist + 1, None)).values
            input_time_indices.append(
                [int(self._time_index[int(v)]) for v in input_times]
            )
            label_time_indices.append(
                [int(self._time_index[int(v)]) for v in label_times]
            )
        return ExampleSpec(input_time_indices, label_time_indices)


class RustTrainBatch:
    def __init__(
        self,
        *,
        rust_batch,
        device: torch.device,
    ) -> None:
        self._rust_batch = rust_batch
        self._device = device
        self._raw_step0_parts: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None
        self._raw_later_steps: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return self._rust_batch.num_steps()

    def get_raw_step0_prognostic(
        self, placement: TensorPlacement = "cpu"
    ) -> torch.Tensor:
        prognostic, _, _ = self._ensure_raw_step0_parts()
        return self._place_tensor(prognostic, placement)

    def get_raw_step0_boundary(
        self, placement: TensorPlacement = "cpu"
    ) -> torch.Tensor:
        _, boundary, _ = self._ensure_raw_step0_parts()
        return self._place_tensor(boundary, placement)

    def get_raw_step0_label(self, placement: TensorPlacement = "cpu") -> torch.Tensor:
        _, _, label = self._ensure_raw_step0_parts()
        return self._place_tensor(label, placement)

    def get_raw_step0_parts(
        self, placement: TensorPlacement = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prognostic, boundary, label = self._ensure_raw_step0_parts()
        return (
            self._place_tensor(prognostic, placement),
            self._place_tensor(boundary, placement),
            self._place_tensor(label, placement),
        )

    def get_raw_boundary(
        self, step: int, placement: TensorPlacement = "cpu"
    ) -> torch.Tensor:
        if step == 0:
            return self.get_raw_step0_boundary(placement)
        boundary, _ = self._ensure_raw_later_step(step)
        return self._place_tensor(boundary, placement)

    def get_raw_label(
        self, step: int, placement: TensorPlacement = "cpu"
    ) -> torch.Tensor:
        if step == 0:
            return self.get_raw_step0_label(placement)
        _, label = self._ensure_raw_later_step(step)
        return self._place_tensor(label, placement)

    def _ensure_raw_step0_parts(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._raw_step0_parts is not None:
            return self._raw_step0_parts
        prognostic_np, boundary_np, label_np = self._rust_batch.raw_step0_parts()
        self._raw_step0_parts = (
            self._to_cpu_tensor(prognostic_np),
            self._to_cpu_tensor(boundary_np),
            self._to_cpu_tensor(label_np),
        )
        return self._raw_step0_parts

    def _ensure_raw_later_step(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._raw_later_steps.get(step)
        if cached is not None:
            return cached
        boundary_np, label_np = self._rust_batch.raw_step_parts(step)
        cached = (
            self._to_cpu_tensor(boundary_np),
            self._to_cpu_tensor(label_np),
        )
        self._raw_later_steps[step] = cached
        return cached

    def _to_cpu_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).to(dtype=torch.float32)

    def _place_tensor(
        self, tensor: torch.Tensor, placement: TensorPlacement
    ) -> torch.Tensor:
        match placement:
            case "cpu":
                return tensor
            case "torch_device":
                return tensor.to(
                    device=self._device, dtype=torch.float32, non_blocking=True
                )
            case _:
                raise ValueError(f"unknown tensor placement {placement!r}")


class RustTrainDataLoader:
    def __init__(
        self,
        *,
        batch_sampler,
        datasets: Sequence[TorchTrainDataset],
        device: torch.device,
    ) -> None:
        self._batch_sampler = batch_sampler
        self._datasets = list(datasets)
        self._device = device
        self._offsets = []
        cumsum = 0
        self._handles: dict[str, TideDatasetHandle] = {}
        for dataset in self._datasets:
            self._offsets.append(cumsum)
            cumsum += len(dataset)
            self._handles[dataset.id] = TideDatasetHandle(dataset)

    def __iter__(self) -> Iterator[RustTrainBatch]:
        for batch_indices in self._batch_sampler:
            yield self._batch_from_indices(batch_indices)

    def __len__(self) -> int:
        return len(self._batch_sampler)

    def __getitem__(self, index: int) -> RustTrainBatch:
        batch_indices = list(iter(self._batch_sampler))[index]
        return self._batch_from_indices(batch_indices)

    @property
    def sampler(self):
        return self._batch_sampler

    @property
    def datasets(self) -> tuple[TorchTrainDataset, ...]:
        return tuple(self._datasets)

    def _batch_from_indices(self, batch_indices: Sequence[int]) -> RustTrainBatch:
        by_dataset: dict[str, list[int]] = {}
        for global_index in batch_indices:
            dataset_index = bisect.bisect_right(self._offsets, global_index) - 1
            dataset = self._datasets[dataset_index]
            local_index = global_index - self._offsets[dataset_index]
            by_dataset.setdefault(dataset.id, []).append(local_index)

        if len(by_dataset) != 1:
            raise NotImplementedError(
                "tide v0 only supports batches sourced from a single dataset"
            )

        dataset_id, local_indices = next(iter(by_dataset.items()))
        return self._handles[dataset_id].make_batch(local_indices, self._device)
