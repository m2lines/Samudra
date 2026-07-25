# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from collections.abc import Iterator
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from typing import ClassVar, Protocol, final

import cftime
import numpy as np
import torch
import xarray as xr
from jaxtyping import Float
from torch.utils.data import Dataset

from samudra.constants import (
    Boundary,
    BoundaryVarNames,
    LoaderVersion,
    Prognostic,
    PrognosticVarNames,
    RolloutStep,
)
from samudra.utils.ctx import BatchGrid
from samudra.utils.data import (
    BatchPreprocessor,
    CanonicalReadRequest,
    CanonicalSource,
    LoadStats,
)
from samudra.utils.device import using_gpu
from samudra.utils.logging import elapsed

logger = logging.getLogger(__name__)


def temporal_fourier_embedding(
    times: list[object],
    *,
    num_scales: int,
    min_hours: float,
    max_hours: float,
) -> torch.Tensor:
    """Encode real timestamps exactly as Otter's continuous time embedding."""
    if num_scales == 0:
        return torch.empty((len(times), 0), dtype=torch.float32)
    hours = []
    for value in times:
        if isinstance(value, cftime.datetime):
            hours.append(
                cftime.date2num(
                    value,
                    units="hours since 1979-01-01 00:00:00",
                    calendar=value.calendar,
                )
            )
        elif isinstance(value, np.datetime64):
            hours.append(
                (value - np.datetime64("1979-01-01T00:00:00"))
                .astype("timedelta64[h]")
                .astype(int)
            )
        else:
            raise TypeError(f"Unsupported time coordinate type: {type(value)}")
    scales = torch.logspace(
        np.log10(min_hours),
        np.log10(max_hours),
        num_scales,
        dtype=torch.float64,
    )
    phase = 2 * torch.pi * torch.tensor(hours, dtype=torch.float64)[:, None] / scales
    return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1).float()


class InferenceDataset(Dataset):
    """This class is used for inference rollouts.

    It creates rolling indices to keep track of histories/past states.
    For example,
    Hist=0 ; 0->[0, 1]; 1->[1, 2]; 2->[2, 3]; 3->[3, 4]
    Hist=1 ; 0->[[0, 1], [2, 3]]; 1->[[2, 3], [4, 5]];
            2->[[4, 5], [6, 7]]; 3->[[6, 7], [8, 9]]
    Hist=2 ; 0->[[0, 1, 2], [3, 4, 5]];
            1->[[3, 4, 5], [6, 7, 8]];
            2->[[6, 7, 8], [9, 10, 11]];
            3->[[9, 10, 11], [12, 13, 14]]
    """

    @elapsed
    def __init__(
        self,
        source: CanonicalSource,
        prognostic_var_names,
        boundary_var_names,
        hist,
        normalize_before_mask,
        masked_fill_value,
        long_rollout,
        output_steps: int | None = None,
        time_embedding_num_scales: int = 0,
        time_embedding_min_hours: float = 3.0,
        time_embedding_max_hours: float = 8760.0,
    ):
        super().__init__()
        # NOTE: Keep tensors on CPU during initialization. This allows the dataset
        # to be passed between DataLoader worker processes. Call to(device) before
        # using the dataset for inference.

        self.hist = hist
        self.output_steps = hist + 1 if output_steps is None else output_steps
        self.time_embedding_num_scales = time_embedding_num_scales
        self.time_embedding_min_hours = time_embedding_min_hours
        self.time_embedding_max_hours = time_embedding_max_hours
        self.prognostic_var_names = tuple(prognostic_var_names)
        self.boundary_var_names = tuple(boundary_var_names)

        self.num_prognostic_channels = (hist + 1) * len(prognostic_var_names)
        self.input_resolution = source.resolution
        self._device = torch.device("cpu")
        self._source = source
        self._times = source.time
        self.preprocessor = BatchPreprocessor(
            source,
            self.prognostic_var_names,
            self.boundary_var_names,
            normalize_before_mask=normalize_before_mask,
            masked_fill_value=masked_fill_value,
        )

        time_indices = np.arange(source.time.size)
        input_steps = self.hist + 1
        total_steps = input_steps + self.output_steps
        starts = np.arange(0, len(time_indices) - total_steps + 1, self.output_steps)
        rolling = starts[:, None] + np.arange(total_steps)[None, :]
        self.rolling_indices = xr.DataArray(
            rolling,
            dims=["window_dim", "time"],
        ).astype(int)

        if long_rollout:
            logger.info(
                f"Long rollout will use input at time {source.time.values[0]} and produce"
                f" output at {source.time.values[self.hist + 1]}"
            )

        self.wet_label = source.masks.prognostic_with_hist(self.output_steps - 1)
        self.size = len(self.rolling_indices)

        if using_gpu():
            self.wet_label = self.wet_label.pin_memory()

        # Inference only currently supports the same output resolution as the input
        # resolution.
        self.ctx = BatchGrid(
            self.wet_label, self.input_resolution, self.input_resolution
        )

    def __len__(self):
        return self.size

    @property
    def num_timeline_steps(self) -> int:
        """Physical timesteps recorded: initial context plus every prediction."""
        return self.hist + 1 + self.size * self.output_steps

    def to(self, device: torch.device) -> "InferenceDataset":
        """Move the dataset's context tensors to the specified device.

        Call this before using the dataset for inference to ensure tensors
        are on the correct device (GPU).
        """
        self.ctx = self.ctx.to(device)
        self._device = device
        self.wet_label = self.wet_label.to(device, non_blocking=True)
        return self

    @property
    def initial_prognostic(self):
        x_index = self._get_x_index(0)
        data_in = self._get_prognostic(x_index)
        return data_in

    def inference_target(self, step: int | slice):
        x_index = self._get_x_index(step)
        label = self._get_label(x_index)
        return label

    def get_initial_input(self) -> tuple[Prognostic, Boundary]:
        prog, boundary, _ = self.__getitem__(0)
        return prog, boundary

    def get_target_time(self, start_step: int, num_steps: int):
        x_index = self._get_x_index(start_step)
        batch_index = x_index.values[0]
        start_target_index = batch_index[self.hist + 1]

        return self._times.isel(
            time=slice(
                start_target_index, start_target_index + num_steps * self.output_steps
            )
        )

    def get_boundary(self, step: int) -> Boundary:
        """Return boundary at the requested step."""
        x_index = self._get_x_index(step)
        boundary = self._get_boundary(x_index)
        return boundary

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx):
        x_index = self._get_x_index(idx)
        data_in_prog = self._get_prognostic(x_index)
        data_in_boundary = self._get_boundary(x_index)
        label = self._get_label(x_index)
        return (data_in_prog, data_in_boundary, label)

    def _get_x_index(self, idx):
        if isinstance(idx, slice):
            if (
                (idx.start is not None and idx.start < 0)
                or (idx.stop is not None and idx.stop < 0)
                or (idx.step is not None and idx.step < 0)
            ):
                raise IndexError("Sorry, negative indexing is not supported!")
            if idx.step is None:
                idx = slice(idx.start, idx.stop, 1)
            if idx.start is None and idx.stop is None:
                idx = slice(0, self.size, idx.step)
            elif idx.start is None:
                idx = slice(0, idx.stop, idx.step)
            elif idx.stop is None:
                idx = slice(idx.start, self.size, idx.step)
        elif isinstance(idx, int):
            if idx < 0:
                raise IndexError("Sorry, negative indexing is not supported!")
            elif idx >= self.size:
                raise IndexError(f"Index {idx} out of range with size {self.size}")
            idx = slice(idx, idx + 1, 1)

        rolling_idx = self.rolling_indices.isel(window_dim=idx)
        x_index = xr.Variable(["window_dim", "time"], rolling_idx)

        return x_index

    def _get_prognostic(self, x_index):
        return self._read_and_prepare(
            x_index.values[:, : self.hist + 1],
            channels=self.prognostic_var_names,
            prognostic=True,
        )

    def _get_boundary(self, x_index):
        """
        This function returns the boundary condition for the current time step.

        With hist > 0, the boundary condition considered is always the last step of
        the input.
        """
        boundary = self._read_and_prepare(
            x_index.values[:, : self.hist + 1],
            channels=self.boundary_var_names,
            prognostic=False,
        )
        zero_indices = np.asarray(x_index.isel(time=self.hist)).reshape(-1)
        zero_times = [self._times.values[int(index)] for index in zero_indices]
        time_embedding = temporal_fourier_embedding(
            zero_times,
            num_scales=self.time_embedding_num_scales,
            min_hours=self.time_embedding_min_hours,
            max_hours=self.time_embedding_max_hours,
        ).to(boundary.device)
        if time_embedding.shape[1]:
            expanded = time_embedding[:, :, None, None].expand(
                -1, -1, boundary.shape[-2], boundary.shape[-1]
            )
            boundary = torch.cat((boundary, expanded), dim=1)
        return boundary

    def _get_label(self, x_index):
        return self._read_and_prepare(
            x_index.values[:, self.hist + 1 :],
            channels=self.prognostic_var_names,
            prognostic=True,
        )

    def _read_and_prepare(
        self,
        time_indices: np.ndarray,
        *,
        channels: tuple[str, ...],
        prognostic: bool,
    ) -> torch.Tensor:
        data = torch.from_numpy(self._source.read(time_indices, channels))
        prepare = (
            self.preprocessor.prepare_prognostic
            if prognostic
            else self.preprocessor.prepare_boundary
        )
        return prepare(data, self._device)

    def get_coords_dict(self):
        return self._source.coordinates()


class InferenceDatasets(Dataset):
    def __init__(self, datasets: list[InferenceDataset], lengths: list[int]):
        self.datasets = datasets
        self.lengths = lengths

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return (self.datasets[idx], self.lengths[idx])


class HostBatch:
    def __init__(self, dataset_id: "TorchTrainDataset.Id"):
        self.dataset_id: TorchTrainDataset.Id = dataset_id
        self.steps: list[RolloutStep] = []
        self.time_embeddings: list[torch.Tensor] = []
        self.load_stats: LoadStats | None = None

    def append(
        self,
        input_: torch.Tensor,
        boundary: torch.Tensor,
        label: torch.Tensor,
        time_embedding: torch.Tensor | None = None,
    ):
        """Add a prognostic input, boundary, and prognostic label as the last step."""
        self.steps.append(RolloutStep(input_, boundary, label))
        self.time_embeddings.append(
            torch.empty(0) if time_embedding is None else time_embedding
        )

    def pin_memory(self):
        self.steps = [
            RolloutStep(
                input_.pin_memory(),
                boundary.pin_memory(),
                label.pin_memory(),
            )
            for input_, boundary, label in self.steps
        ]
        self.time_embeddings = [
            time_embedding.pin_memory() for time_embedding in self.time_embeddings
        ]
        return self


class ModelBatch:
    """A single batch of training data.

    A single batch contains multiple steps worth of ``RolloutStep`` entries, each
    of which is a ``(prognostic_input, boundary_input, label)`` triple. The
    prognostic and boundary tensors are carried separately because the
    samudra-multi model encodes them separately (Samudra just concatenates them later).
    """

    def __init__(self, ctx: BatchGrid):
        self.ctx = ctx
        self.steps: list[RolloutStep] = []
        self.load_stats: LoadStats | None = None

    def append(
        self, prognostic_input: Prognostic, boundary_input: Boundary, label: Prognostic
    ) -> None:
        """Add another RolloutStep as a new step."""
        self.steps.append(RolloutStep(prognostic_input, boundary_input, label))

    def get_initial_input(self) -> tuple[Prognostic, Boundary]:
        return self.get_input(0)

    def get_input(self, step: int) -> tuple[Prognostic, Boundary]:
        prog, boundary, _ = self.steps[step]
        return prog, boundary

    def get_label(self, step: int) -> Prognostic:
        return self.steps[step][2]

    def __getitem__(self, step: int) -> RolloutStep:
        """Converts index (step) into (prognostic, boundary, label) triple."""
        return self.steps[step]

    def __len__(self) -> int:
        return len(self.steps)

    def prefix(self, steps: int) -> "ModelBatch":
        """Return a lightweight view containing the first rollout steps."""
        if not 1 <= steps <= len(self):
            raise ValueError(f"Expected 1..{len(self)} rollout steps, got {steps}.")
        result = ModelBatch(self.ctx)
        result.steps = self.steps[:steps]
        result.load_stats = self.load_stats
        return result

    def __iter__(self):
        return iter(self.steps)

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Keep batch storage alive until work queued on ``stream`` completes."""
        self.ctx.label_mask.record_stream(stream)
        for prognostic, boundary, label in self.steps:
            prognostic.record_stream(stream)
            boundary.record_stream(stream)
            label.record_stream(stream)


class TrainBatchLoader(Protocol):
    """Training-loop contract shared by Torch and native batch loaders."""

    def __iter__(self) -> Iterator[ModelBatch]: ...

    def __len__(self) -> int: ...

    def set_epoch(self, epoch: int) -> None: ...

    def close(self) -> None: ...


def close_pytorch_dataloader(dataloader: torch.utils.data.DataLoader) -> None:
    """Deterministically stop persistent workers created by a PyTorch loader."""
    iterator = dataloader._iterator
    if iterator is not None:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if shutdown_workers is not None:
            shutdown_workers()
        dataloader._iterator = None


@dataclass(frozen=True)
class BatchReadUse:
    """One canonical source read and its model-facing preparation policy."""

    source: CanonicalSource
    request: CanonicalReadRequest
    mask: torch.Tensor
    cache_name: str


@dataclass(frozen=True)
class BatchReadStep:
    input: BatchReadUse
    boundary: BatchReadUse
    label: BatchReadUse
    time_embedding: torch.Tensor


@dataclass(frozen=True)
class BatchReadPlan:
    """Storage-independent reads for a homogeneous batch and full rollout."""

    dataset_id: str
    steps: tuple[BatchReadStep, ...]


class TrainingShard:
    """Training-window semantics shared by sample and native batch readers."""

    def __init__(
        self,
        input_source: CanonicalSource,
        label_source: CanonicalSource | None,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        hist: int,
        steps: int,
        normalize_before_mask: bool,
        masked_fill_value: float,
        stride: int,
        output_steps: int | None = None,
        time_embedding_num_scales: int = 0,
        time_embedding_min_hours: float = 3.0,
        time_embedding_max_hours: float = 8760.0,
        shard_id: str | None = None,
    ) -> None:
        self.id = shard_id or f"TrainingShard_{id(self)}"
        self.input_source = input_source
        self.label_source = label_source or input_source
        self.hist = hist
        self.output_steps = hist + 1 if output_steps is None else output_steps
        self.time_embedding_num_scales = time_embedding_num_scales
        self.time_embedding_min_hours = time_embedding_min_hours
        self.time_embedding_max_hours = time_embedding_max_hours
        self.steps = steps
        self.stride = stride
        self.normalize_before_mask = normalize_before_mask
        self.masked_fill_value = masked_fill_value
        self.prognostic_var_names = tuple(prognostic_var_names)
        self.boundary_var_names = tuple(boundary_var_names)
        self.num_prognostic_channels = (hist + 1) * len(prognostic_var_names)
        self.num_boundary_channels = (hist + 1) * len(boundary_var_names) + (
            2 * time_embedding_num_scales
        )
        if not np.array_equal(self.input_source.time, self.label_source.time):
            raise ValueError("Input and label sources have different time slices")

        total_times = hist + 1 + self.output_steps
        num_windows = input_source.time.size - (total_times - 1) * stride
        indices = xr.DataArray(np.arange(num_windows), dims=["window"])
        offsets = xr.DataArray(np.arange(total_times), dims=["time"])
        self.rolling_indices: Float[xr.DataArray, "window time"] = (
            indices + stride * offsets
        )
        self.input_prognostic_mask = input_source.masks.prognostic
        self.label_prognostic_mask = self.label_source.masks.prognostic
        self.boundary_mask = input_source.masks.boundary
        self.ctx = BatchGrid(
            label_mask=self.label_source.masks.prognostic_with_hist(
                self.output_steps - 1
            ),
            input_resolution_cpu=input_source.resolution,
            output_resolution_cpu=self.label_source.resolution,
        )
        self.size = (
            input_source.time.size - steps * self.output_steps * stride - hist * stride
        )

    def __len__(self) -> int:
        return self.size

    @property
    def sources(self) -> tuple[CanonicalSource, ...]:
        if self.label_source is self.input_source:
            return (self.input_source,)
        return self.input_source, self.label_source

    @property
    def batch_compatibility_key(self) -> str:
        """Preserve the current homogeneous dataset-ID batching contract."""
        return self.id

    def window_indices(self, index: int, step: int) -> np.ndarray:
        if index < 0:
            raise IndexError("Negative training-window indices are not supported")
        if index >= len(self):
            raise IndexError("Training-window index out of range")
        window = index + step * self.output_steps * self.stride
        return self.rolling_indices.isel(window=window, drop=True).to_numpy()

    def window_plan(self, indices: list[int]) -> BatchReadPlan:
        """Plan all canonical reads for a batch and full rollout."""
        if not indices:
            raise ValueError("Cannot plan an empty training batch")
        planned_steps = []
        for step in range(self.steps):
            relative = np.stack(
                [self.window_indices(index, step) for index in indices]
            ).astype(np.int64, copy=False)
            current = relative[:, : self.hist + 1]
            forecast = relative[:, self.hist + 1 :]
            zero_times = [
                self.input_source.time.values[int(index)] for index in current[:, -1]
            ]
            planned_steps.append(
                BatchReadStep(
                    input=BatchReadUse(
                        self.input_source,
                        CanonicalReadRequest(current, self.prognostic_var_names),
                        self.input_prognostic_mask,
                        "prognostic_input",
                    ),
                    boundary=BatchReadUse(
                        self.input_source,
                        CanonicalReadRequest(current, self.boundary_var_names),
                        self.boundary_mask,
                        "boundary_input",
                    ),
                    label=BatchReadUse(
                        self.label_source,
                        CanonicalReadRequest(forecast, self.prognostic_var_names),
                        self.label_prognostic_mask,
                        "prognostic_label",
                    ),
                    time_embedding=temporal_fourier_embedding(
                        zero_times,
                        num_scales=self.time_embedding_num_scales,
                        min_hours=self.time_embedding_min_hours,
                        max_hours=self.time_embedding_max_hours,
                    ),
                )
            )
        return BatchReadPlan(self.id, tuple(planned_steps))


class TrainBatchPreparer:
    """Rank-local normalization, masking, shaping, and device-static caches."""

    def __init__(self, shard: TrainingShard) -> None:
        self.shard = shard
        self._input = BatchPreprocessor(
            shard.input_source,
            shard.prognostic_var_names,
            shard.boundary_var_names,
            normalize_before_mask=shard.normalize_before_mask,
            masked_fill_value=shard.masked_fill_value,
        )
        self._label = BatchPreprocessor(
            shard.label_source,
            shard.prognostic_var_names,
            shard.boundary_var_names,
            normalize_before_mask=shard.normalize_before_mask,
            masked_fill_value=shard.masked_fill_value,
        )
        self._device_static: dict[
            tuple[torch.device, str],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._device_ctx: dict[torch.device, BatchGrid] = {}

    def prepare_host_batch(
        self, host_batch: HostBatch, device: torch.device
    ) -> ModelBatch:
        model_batch = self.new_model_batch(device)
        for (input_, boundary, label), time_embedding in zip(
            host_batch.steps, host_batch.time_embeddings, strict=True
        ):
            prepared_boundary = self._input.prepare_boundary(boundary, device)
            model_batch.append(
                self._input.prepare_prognostic(input_, device),
                self.append_time_embedding(
                    prepared_boundary,
                    time_embedding,
                    device,
                ),
                self._label.prepare_prognostic(label, device),
            )
        model_batch.load_stats = host_batch.load_stats
        return model_batch

    def new_model_batch(self, device: torch.device) -> ModelBatch:
        device_ctx = self._device_ctx.get(device)
        if device_ctx is None:
            device_ctx = self.shard.ctx.to(device)
            self._device_ctx[device] = device_ctx
        return ModelBatch(device_ctx)

    @staticmethod
    def append_time_embedding(
        boundary: torch.Tensor,
        time_embedding: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        time_embedding = time_embedding.to(device=device, non_blocking=True)
        if not time_embedding.shape[-1]:
            return boundary
        expanded = time_embedding[:, :, None, None].expand(
            -1, -1, boundary.shape[-2], boundary.shape[-1]
        )
        return torch.cat((boundary, expanded), dim=1)

    def normalize_and_mask_device_planes(
        self,
        use: BatchReadUse,
        data: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Prepare unique ``(time, variable, lat, lon)`` planes on device."""
        if data.device.type != device.type or (
            device.index is not None and data.device.index != device.index
        ):
            raise ValueError(
                f"Unique ocean planes are on {data.device}; expected {device}"
            )

        cache_key = (device, use.cache_name)
        static = self._device_static.get(cache_key)
        if static is None:
            statistics = use.source.statistics(use.request.channels)
            mean = torch.from_numpy(statistics.mean).to(device=device, dtype=data.dtype)
            std = torch.from_numpy(statistics.std).to(device=device, dtype=data.dtype)
            shape = (1, len(use.request.channels), 1, 1)
            static = (
                mean.reshape(shape),
                std.reshape(shape),
                use.mask.to(device=device, non_blocking=True),
            )
            self._device_static[cache_key] = static
        mean, std, mask = static

        def normalize(tensor: torch.Tensor) -> torch.Tensor:
            return ((tensor - mean) / std).nan_to_num(nan=0.0)

        if self.shard.normalize_before_mask:
            data = normalize(data)
        data = torch.where(mask, data, self.shard.masked_fill_value)
        if not self.shard.normalize_before_mask:
            data = normalize(data)
        return data


@final
class TorchTrainDataset(Dataset[HostBatch]):
    """
    This class is used for training and validation.

    It creates rolling indices to keep track of histories/past states. But different
    from InferenceDataset, as it creates rolling indices based on stride. By default,
    the sliding window / stride is 1.

    We make use of ModelBatch class to store a single sample.

    For example,
    Hist=0 ; step=0->[0, 1]; step=1->[1, 2]; step=2->[2, 3]; step=3->[3, 4]
    Hist=1 ; step=0->[[0, 1], [2, 3]]; step=1->[[2, 3], [4, 5]];
            step=2->[[4, 5], [6, 7]]; step=3->[[6, 7], [8, 9]]
    Hist=2 ; step=0->[[0, 1, 2], [3, 4, 5]];
            step=1->[[3, 4, 5], [6, 7, 8]];
            step=2->[[6, 7, 8], [9, 10, 11]];
            step=3->[[9, 10, 11], [12, 13, 14]]
    """

    type Id = str

    FLAG = LoaderVersion.OM4_TORCH

    # Shared across all instances within a process. Created lazily on first
    # __getitem__ call so that each DataLoader worker gets its own clean
    # executor.
    _shared_executor: ClassVar[ThreadPoolExecutor | None] = None

    @classmethod
    def _get_executor(cls) -> ThreadPoolExecutor:
        if cls._shared_executor is None:
            cls._shared_executor = ThreadPoolExecutor(
                max_workers=None, thread_name_prefix="concurrent_compute"
            )
        return cls._shared_executor

    @elapsed
    def __init__(
        self,
        input_source: CanonicalSource,
        label_source: CanonicalSource | None,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        hist: int,
        steps: int,
        normalize_before_mask: bool,
        masked_fill_value: float,
        stride: int = 1,
        concurrent_compute_: bool = False,
        output_steps: int | None = None,
        time_embedding_num_scales: int = 0,
        time_embedding_min_hours: float = 3.0,
        time_embedding_max_hours: float = 8760.0,
        shard_id: str | None = None,
    ):
        super().__init__()
        self.shard = TrainingShard(
            input_source=input_source,
            label_source=label_source,
            prognostic_var_names=prognostic_var_names,
            boundary_var_names=boundary_var_names,
            hist=hist,
            steps=steps,
            normalize_before_mask=normalize_before_mask,
            masked_fill_value=masked_fill_value,
            stride=stride,
            output_steps=output_steps,
            time_embedding_num_scales=time_embedding_num_scales,
            time_embedding_min_hours=time_embedding_min_hours,
            time_embedding_max_hours=time_embedding_max_hours,
            shard_id=shard_id,
        )
        self.preparer = TrainBatchPreparer(self.shard)
        self.id = self.shard.id
        self._concurrent_compute = concurrent_compute_

    def __len__(self) -> int:
        return len(self.shard)

    @property
    def hist(self) -> int:
        return self.shard.hist

    @property
    def steps(self) -> int:
        return self.shard.steps

    @property
    def stride(self) -> int:
        return self.shard.stride

    @property
    def prognostic_var_names(self) -> tuple[str, ...]:
        return self.shard.prognostic_var_names

    @property
    def boundary_var_names(self) -> tuple[str, ...]:
        return self.shard.boundary_var_names

    @property
    def sources(self) -> tuple[CanonicalSource, ...]:
        return self.shard.sources

    @property
    def batch_compatibility_key(self) -> str:
        return self.shard.batch_compatibility_key

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx: int):
        start_time = time.perf_counter()
        host_batch = HostBatch(self.id)
        plan = self.shard.window_plan([idx])
        for step in plan.steps:
            reads = (step.input, step.boundary, step.label)
            if self._concurrent_compute:
                executor = self._get_executor()
                futures = [
                    executor.submit(
                        use.source.read,
                        use.request.time_indices,
                        use.request.channels,
                    )
                    for use in reads
                ]
                loaded = [future.result()[0] for future in futures]
            else:
                loaded = [
                    use.source.read(
                        use.request.time_indices,
                        use.request.channels,
                    )[0]
                    for use in reads
                ]
            input_, boundary, label = map(torch.from_numpy, loaded)
            host_batch.append(input_, boundary, label, step.time_embedding[0])
        host_batch.load_stats = LoadStats(time.perf_counter() - start_time)

        return host_batch

    def to_model_batch(self, host_batch: HostBatch, device: torch.device) -> ModelBatch:
        """Convert HostBatch to ModelBatch, moving tensors to the specified device.

        Args:
            host_batch: CPU data from worker process
            device: Target device (typically GPU) to move tensors to

        Returns:
            ModelBatch with tensors on the target device
        """
        return self.preparer.prepare_host_batch(host_batch, device)


@final
class BatchLoader:
    """Wrapper around a torch DataLoader that handles GPU post-processing.

    This class wraps a DataLoader[HostBatch] and converts the raw data
    to ModelBatch by applying GPU-based normalization and masking. This allows
    the data loading process to handle I/O while the main process handles
    GPU operations.

    Since the data samples flow from one process to the other, we want to tie
    them back to the dataset they came from which knows how to do that second
    half once they're in the main process which has GPU access set up. To do that,
    each data sample (which could come from a different dataset) has a dataset ID
    -- `datasets` maps from those IDs to the original datasets.
    """

    def __init__(
        self,
        host_loader: torch.utils.data.DataLoader[HostBatch],
        datasets: list[TorchTrainDataset],
        device: torch.device,
    ):
        self._host_loader = host_loader
        self._datasets = {dataset.id: dataset for dataset in datasets}
        self._device = device

    def __iter__(self):
        """Iterate over the dataloader, converting HostBatch to ModelBatch."""
        for host_batch in self._host_loader:
            dataset = self._datasets[host_batch.dataset_id]
            model_batch = dataset.to_model_batch(host_batch, self._device)
            yield model_batch

    def __len__(self) -> int:
        return len(self._host_loader)

    def set_epoch(self, epoch: int) -> None:
        set_epoch = getattr(self._host_loader.batch_sampler, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)

    def close(self) -> None:
        """Release persistent PyTorch workers owned by this loader."""
        close_pytorch_dataloader(self._host_loader)

    def __getitem__(self, index: int) -> ModelBatch:
        """Access a single item by index, converting HostBatch to ModelBatch.

        Note: This bypasses the DataLoader's sampling/batching and directly accesses
        the underlying dataset for test purposes.
        """
        # Access the underlying dataset directly
        host_batch = self._host_loader.dataset[index]
        # Apply the collate function to add batch dimension (expects a list)
        collate_fn = self._host_loader.collate_fn
        if collate_fn is not None:
            host_batch = collate_fn([host_batch])
        # Get the dataset that created this raw data
        dataset = self._datasets[host_batch.dataset_id]
        # Convert to ModelBatch
        model_batch = dataset.to_model_batch(host_batch, self._device)
        return model_batch

    @property
    def dataset(self):
        return self._host_loader.dataset

    @property
    def sampler(self):
        return self._host_loader.sampler
