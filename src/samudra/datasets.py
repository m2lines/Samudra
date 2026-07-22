# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from collections.abc import Iterator
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, final

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from jaxtyping import Float
from torch.utils.data import Dataset

from samudra.constants import (
    Boundary,
    BoundaryVarNames,
    Example,
    GridMask,
    Input,
    LoaderVersion,
    Prognostic,
    PrognosticMask,
    PrognosticVarNames,
)
from samudra.utils.ctx import GridContext
from samudra.utils.data import (
    CanonicalDataset,
    CanonicalReadRequest,
    LoadStats,
    OceanData,
)
from samudra.utils.device import using_gpu
from samudra.utils.logging import elapsed

logger = logging.getLogger(__name__)

TargetTimeMode = Literal["forecast", "current"]


def season_decade_stratified_indices(
    time: xr.DataArray,
    valid_size: int,
    num_samples: int,
    seed: int,
    *,
    anchor_offset: int = 0,
) -> np.ndarray:
    """Select valid window indices evenly across decade/season strata.

    The returned indices refer to window starts. ``anchor_offset`` chooses the
    timestamp within each window used to assign its stratum, normally the first
    forecast target. Sampling windows rather than slicing the source preserves
    contiguous input/target timestamps around every selected example.
    """
    if valid_size < 1:
        raise ValueError("No valid training windows are available.")
    if num_samples > valid_size:
        raise ValueError(
            f"Requested {num_samples} samples from only {valid_size} valid windows."
        )
    if anchor_offset < 0 or anchor_offset + valid_size > time.size:
        raise ValueError("The stratum anchor lies outside the source time coordinate.")

    groups: dict[tuple[int, int], list[int]] = {}
    anchors = time.values[anchor_offset : anchor_offset + valid_size]
    for index, timestamp in enumerate(anchors):
        decade = int(timestamp.year) // 10 * 10
        season = (int(timestamp.month) - 1) // 3
        groups.setdefault((decade, season), []).append(index)

    generator = np.random.default_rng(seed)
    for indices in groups.values():
        generator.shuffle(indices)

    selected: list[int] = []
    ordered_groups = sorted(groups)
    offsets = {key: 0 for key in ordered_groups}
    while len(selected) < num_samples:
        made_progress = False
        for key in ordered_groups:
            offset = offsets[key]
            if offset >= len(groups[key]):
                continue
            selected.append(groups[key][offset])
            offsets[key] += 1
            made_progress = True
            if len(selected) == num_samples:
                break
        if not made_progress:
            raise AssertionError("Stratified sampling exhausted valid windows early.")

    result = np.asarray(sorted(selected), dtype=np.int64)
    result.setflags(write=False)
    return result


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
        src: CanonicalDataset,
        prognostic_var_names,
        boundary_var_names,
        hist,
        normalize_before_mask,
        masked_fill_value,
        long_rollout,
    ):
        super().__init__()
        # NOTE: Keep tensors on CPU during initialization. This allows the dataset
        # to be passed between DataLoader worker processes. Call to(device) before
        # using the dataset for inference.

        self.hist = hist

        self.num_prognostic_channels = (hist + 1) * len(prognostic_var_names)
        self.input_res = src.resolution
        self._prognostic_src = src.select_channels(
            prognostic_var_names, prefix="prognostic"
        )
        self._boundary_src = src.select_channels(boundary_var_names, prefix="boundary")
        self._times = src.time
        self.normalize_before_mask = normalize_before_mask
        self.masked_fill_value = masked_fill_value

        time_indices = np.arange(src.time.size)
        indices = xr.DataArray(
            time_indices,
            dims=["time"],
            coords={"time": time_indices},
        )
        total_steps = 2 * self.hist + 1
        rolling_indices = indices.rolling(
            time=len(time_indices) - total_steps, center=False
        ).construct("window_dim")
        rolling_indices = rolling_indices.transpose("window_dim", "time").isel(
            time=slice(len(time_indices) - total_steps - 1, None)
        )  # Remove first few null indices
        self.rolling_indices = rolling_indices.isel(
            window_dim=slice(0, None, self.hist + 1)
        )  # Skip indices based on history
        self.rolling_indices = self.rolling_indices.astype(int)

        if long_rollout:
            logger.info(
                f"Long rollout will use input at time {src.time.values[0]} and produce"
                f" output at {src.time.values[self.hist + 1]}"
            )

        self.wet: PrognosticMask = src.masks.prognostic
        self.wet_surface: GridMask = src.masks.boundary
        self.wet_label = src.masks.prognostic_with_hist(self.hist)
        self.size = len(self.rolling_indices)

        if using_gpu():
            self.wet = self.wet.pin_memory()
            self.wet_surface = self.wet_surface.pin_memory()
            self.wet_label = self.wet_label.pin_memory()

        # Inference only currently supports the same output resolution as the input
        # resolution.
        self.ctx = GridContext(self.wet_label, self.input_res, self.input_res)

    def __len__(self):
        return self.size

    def to(self, device: torch.device) -> "InferenceDataset":
        """Move the dataset's context tensors to the specified device.

        Call this before using the dataset for inference to ensure tensors
        are on the correct device (GPU).
        """
        self.ctx = self.ctx.to(device)
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
        steps_predicted = len(batch_index) // 2
        start_target_index = batch_index[steps_predicted]

        return self._times.isel(
            time=slice(
                start_target_index, start_target_index + num_steps * steps_predicted
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
            self._prognostic_src,
            x_index.values[:, : self.hist + 1],
            mask=self.wet,
        )

    def _get_boundary(self, x_index):
        """
        This function returns the boundary condition for the current time step.

        With hist > 0, the boundary condition considered is always the last step of
        the input.
        """
        return self._read_and_prepare(
            self._boundary_src,
            x_index.values[:, : self.hist + 1],
            mask=self.wet_surface,
        )

    def _get_label(self, x_index):
        return self._read_and_prepare(
            self._prognostic_src,
            x_index.values[:, self.hist + 1 :],
            mask=self.wet,
        )

    def _read_and_prepare(
        self,
        src: CanonicalDataset,
        time_indices: np.ndarray,
        *,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        data = torch.from_numpy(src.read(CanonicalReadRequest(time_indices)).values)
        prepared = OceanData.from_data_source(data, mask, src).normalize_and_mask(
            self.normalize_before_mask, self.masked_fill_value
        )
        return rearrange(
            prepared,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )

    def get_coords_dict(self):
        return self._prognostic_src.coordinates()


class InferenceDatasets(Dataset):
    def __init__(self, datasets: list[InferenceDataset], lengths: list[int]):
        self.datasets = datasets
        self.lengths = lengths

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return (self.datasets[idx], self.lengths[idx])


class RawTrainData:
    def __init__(self, dataset_id: "TorchTrainDataset.Id"):
        self.dataset_id: TorchTrainDataset.Id = dataset_id
        self.raw_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.load_stats: LoadStats | None = None

    def insert(
        self,
        input_: torch.Tensor,
        boundary: torch.Tensor,
        label: torch.Tensor,
    ):
        """Add a prognostic input, boundary, and prognostic label as the last step."""
        self.raw_data.append((input_, boundary, label))

    def to(self, device: torch.device):
        self.raw_data = [
            (
                input_.to(device, non_blocking=True),
                boundary.to(device, non_blocking=True),
                label.to(device, non_blocking=True),
            )
            for input_, boundary, label in self.raw_data
        ]

    def pin_memory(self):
        self.raw_data = [
            (
                input_.pin_memory(),
                boundary.pin_memory(),
                label.pin_memory(),
            )
            for input_, boundary, label in self.raw_data
        ]
        return self


class TrainData:
    """A single batch of training data.

    A single batch contains multiple steps worth of ``Example`` entries, each
    of which is a ``(prognostic_input, boundary_input, label)`` triple. The
    prognostic and boundary tensors are carried separately because the
    samudra-multi model encodes them separately (Samudra just concatenates them later).
    """

    def __init__(
        self, num_prognostic_channels: int, num_boundary_channels: int, ctx: GridContext
    ):
        self.num_prognostic_channels = num_prognostic_channels
        self.num_boundary_channels = num_boundary_channels
        self.ctx = ctx
        self.example_by_step: list[Example] = []
        self.load_stats: LoadStats | None = None

    def append(
        self, prognostic_input: Prognostic, boundary_input: Boundary, label: Prognostic
    ) -> None:
        """Add another Example as a new step."""
        self.example_by_step.append((prognostic_input, boundary_input, label))

    def get_initial_input(self) -> tuple[Prognostic, Boundary]:
        return self.get_input(0)

    def get_input(self, step: int) -> tuple[Prognostic, Boundary]:
        prog, boundary, _ = self.example_by_step[step]
        return prog, boundary

    def get_label(self, step: int) -> Prognostic:
        return self.example_by_step[step][2]

    def __getitem__(self, step: int) -> Example:
        """Converts index (step) into (prognostic, boundary, label) triple."""
        return self.example_by_step[step]

    def __len__(self) -> int:
        return len(self.example_by_step)

    def __iter__(self):
        return iter(range(len(self)))

    def to(self, device: torch.device) -> None:
        for step in self:
            prog, boundary, label = self.example_by_step[step]
            self.example_by_step[step] = (
                prog.to(device, non_blocking=True),
                boundary.to(device, non_blocking=True),
                label.to(device, non_blocking=True),
            )

    def pin_memory(self):
        for step in self:
            prog, boundary, label = self.example_by_step[step]
            self.example_by_step[step] = (
                prog.pin_memory(),
                boundary.pin_memory(),
                label.pin_memory(),
            )
        return self

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Keep batch storage alive until work queued on ``stream`` completes."""
        self.ctx.label_mask.record_stream(stream)
        for prognostic, boundary, label in self.example_by_step:
            prognostic.record_stream(stream)
            boundary.record_stream(stream)
            label.record_stream(stream)


class TrainBatchLoader(Protocol):
    """Training-loop contract shared by Torch and native batch loaders."""

    def __iter__(self) -> Iterator[TrainData]: ...

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

    source: CanonicalDataset
    request: CanonicalReadRequest
    mask: torch.Tensor
    cache_name: str


@dataclass(frozen=True)
class BatchReadStep:
    input: BatchReadUse
    boundary: BatchReadUse
    label: BatchReadUse


@dataclass(frozen=True)
class BatchReadPlan:
    """Storage-independent reads for a homogeneous batch and full rollout."""

    dataset_id: str
    steps: tuple[BatchReadStep, ...]


class TrainingShard:
    """Immutable-ish training semantics shared by sample and batch readers."""

    def __init__(
        self,
        src: CanonicalDataset,
        dst: CanonicalDataset | None,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        hist: int,
        steps: int,
        normalize_before_mask: bool,
        masked_fill_value: float,
        stride: int,
        shard_id: str | None = None,
        sample_num: int | None = None,
        sample_seed: int = 0,
        target_time_mode: TargetTimeMode = "forecast",
    ) -> None:
        self.id = shard_id or f"TrainingShard_{id(self)}"
        srcs = [src, dst] if dst else [src]
        self.hist = hist
        self.steps = steps
        self.stride = stride
        self.target_time_mode = target_time_mode
        self.normalize_before_mask = normalize_before_mask
        self.masked_fill_value = masked_fill_value
        self.prognostic_var_names = tuple(prognostic_var_names)
        self.boundary_var_names = tuple(boundary_var_names)
        self.num_prognostic_channels = (hist + 1) * len(prognostic_var_names)
        self.num_boundary_channels = (hist + 1) * len(boundary_var_names)
        if not np.array_equal(srcs[0].time, srcs[-1].time):
            raise ValueError("Source and destination have different time slices")

        self.prognostic_srcs = [
            source.select_channels(prognostic_var_names, prefix="prog")
            for source in srcs
        ]
        self.boundary_src = src.select_channels(boundary_var_names, prefix="boundary")
        # Forecast reads need one input-history block and one future-label block.
        # Current-time reconstruction reuses the input block as its label, so
        # reserving the unused future block would make ``rolling_indices`` shorter
        # than the valid size advertised below and fail on otherwise valid tail
        # samples.
        total_times = hist + 1 if target_time_mode == "current" else 2 * hist + 2
        num_windows = src.time.size - (total_times - 1) * stride
        indices = xr.DataArray(np.arange(num_windows), dims=["window"])
        offsets = xr.DataArray(np.arange(total_times), dims=["time"])
        self.rolling_indices: Float[xr.DataArray, "window time"] = (
            indices + stride * offsets
        )
        self.wet_prognostic = [source.masks.prognostic for source in srcs]
        self.wet_surface = src.masks.boundary
        self.ctx = GridContext(
            label_mask=self.prognostic_srcs[-1].masks.prognostic_with_hist(hist),
            input_resolution_cpu=self.prognostic_srcs[0].resolution,
            output_resolution_cpu=self.prognostic_srcs[-1].resolution,
        )
        if target_time_mode == "forecast":
            full_size = src.time.size - steps * (hist + 1) * stride - hist * stride
            sample_anchor_offset = (hist + 1) * stride
        else:
            full_size = src.time.size - ((steps - 1) * (hist + 1) + hist) * stride
            sample_anchor_offset = hist * stride
        self.sample_indices = (
            season_decade_stratified_indices(
                src.time,
                full_size,
                sample_num,
                sample_seed,
                anchor_offset=sample_anchor_offset,
            )
            if sample_num is not None
            else None
        )
        self.size = (
            len(self.sample_indices) if self.sample_indices is not None else full_size
        )

    def __len__(self) -> int:
        return self.size

    @property
    def batch_compatibility_key(self) -> str:
        """Preserve the current homogeneous dataset-ID batching contract."""
        return self.id

    def window_indices(self, index: int, step: int) -> np.ndarray:
        if index < 0:
            raise IndexError("Negative training-window indices are not supported")
        if index >= len(self):
            raise IndexError("Training-window index out of range")
        window_start = (
            int(self.sample_indices[index])
            if self.sample_indices is not None
            else index
        )
        window = window_start + step * (self.hist + 1) * self.stride
        return self.rolling_indices.isel(window=window, drop=True).to_numpy()

    def window_plan(self, indices: list[int]) -> BatchReadPlan:
        """Plan all reads for a batch, retaining batch/time index shape."""
        if not indices:
            raise ValueError("Cannot plan an empty training batch")
        planned_steps = []
        for step in range(self.steps):
            relative = np.stack(
                [self.window_indices(index, step) for index in indices]
            ).astype(np.int64, copy=False)
            current = relative[:, : self.hist + 1]
            forecast = relative[:, self.hist + 1 :]
            label = current if self.target_time_mode == "current" else forecast
            planned_steps.append(
                BatchReadStep(
                    input=BatchReadUse(
                        self.prognostic_srcs[0],
                        CanonicalReadRequest(current),
                        self.wet_prognostic[0],
                        "prognostic_input",
                    ),
                    boundary=BatchReadUse(
                        self.boundary_src,
                        CanonicalReadRequest(current),
                        self.wet_surface,
                        "boundary_input",
                    ),
                    label=BatchReadUse(
                        self.prognostic_srcs[-1],
                        CanonicalReadRequest(label),
                        self.wet_prognostic[-1],
                        "prognostic_label",
                    ),
                )
            )
        return BatchReadPlan(self.id, tuple(planned_steps))


class TrainBatchPreparer:
    """Rank-local normalization, masking, shaping, and device-static caches."""

    def __init__(self, shard: TrainingShard) -> None:
        self.shard = shard
        self._device_ocean_static: dict[
            tuple[torch.device, str],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._device_ctx: dict[torch.device, GridContext] = {}

    def prepare_raw(
        self, raw_train_data: RawTrainData, device: torch.device
    ) -> TrainData:
        train_data = self.new_train_data(device)
        policies = (
            (
                "prognostic_input",
                self.shard.wet_prognostic[0],
                self.shard.prognostic_srcs[0],
            ),
            ("boundary_input", self.shard.wet_surface, self.shard.boundary_src),
            (
                "prognostic_label",
                self.shard.wet_prognostic[-1],
                self.shard.prognostic_srcs[-1],
            ),
        )
        for raw_step in raw_train_data.raw_data:
            oceans = [
                self._ocean_data_to_device(name, data, mask, source, device)
                for data, (name, mask, source) in zip(raw_step, policies)
            ]
            train_data.append(*(self._prep_tensor_steps(ocean) for ocean in oceans))
        train_data.load_stats = raw_train_data.load_stats
        return train_data

    def new_train_data(self, device: torch.device) -> TrainData:
        device_ctx = self._device_ctx.get(device)
        if device_ctx is None:
            device_ctx = self.shard.ctx.to(device)
            self._device_ctx[device] = device_ctx
        return TrainData(
            self.shard.num_prognostic_channels,
            self.shard.num_boundary_channels,
            device_ctx,
        )

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
        ocean = self._ocean_data_from_device(
            use.cache_name, data.unsqueeze(0), use.mask, use.source, device
        )
        return ocean.normalize_and_mask(
            self.shard.normalize_before_mask, self.shard.masked_fill_value
        ).squeeze(0)

    def _ocean_data_to_device(
        self,
        cache_name: str,
        data: torch.Tensor,
        mask: torch.Tensor,
        source: CanonicalDataset,
        device: torch.device,
    ) -> OceanData:
        return self._ocean_data_from_device(
            cache_name, data.to(device=device, non_blocking=True), mask, source, device
        )

    def _ocean_data_from_device(
        self,
        cache_name: str,
        data: torch.Tensor,
        mask: torch.Tensor,
        source: CanonicalDataset,
        device: torch.device,
    ) -> OceanData:
        cache_key = (device, cache_name)
        static = self._device_ocean_static.get(cache_key)
        if static is None:
            template = OceanData.from_data_source(data[:0], mask, source).to(
                device=device, non_blocking=True
            )
            static = (template.means, template.stds, template.mask)
            self._device_ocean_static[cache_key] = static
        means, stds, device_mask = static
        return OceanData(data=data, means=means, stds=stds, mask=device_mask)

    def _prep_tensor_steps(self, ocean_data: OceanData) -> Input:
        steps = ocean_data.normalize_and_mask(
            self.shard.normalize_before_mask, self.shard.masked_fill_value
        )
        return rearrange(
            steps, "batch time variable lat lon -> batch (time variable) lat lon"
        )


@final
class TorchTrainDataset(Dataset[RawTrainData]):
    """
    This class is used for training and validation.

    It creates rolling indices to keep track of histories/past states. But different
    from InferenceDataset, as it creates rolling indices based on stride. By default,
    the sliding window / stride is 1.

    We make use of TrainData class to store a single sample.

    For example,
    Hist=0 ; TD: step=0->[0, 1]; step=1->[1, 2]; step=2->[2, 3]; step=3->[3, 4]
    Hist=1 ; TD: step=0->[[0, 1], [2, 3]]; step=1->[[2, 3], [4, 5]];
            step=2->[[4, 5], [6, 7]]; step=3->[[6, 7], [8, 9]]
    Hist=2 ; TD: step=0->[[0, 1, 2], [3, 4, 5]];
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
        src: CanonicalDataset,
        dst: CanonicalDataset | None,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        hist: int,
        steps: int,
        normalize_before_mask: bool,
        masked_fill_value: float,
        stride: int = 1,
        concurrent_compute_: bool = False,
        shard_id: str | None = None,
        sample_num: int | None = None,
        sample_seed: int = 0,
        target_time_mode: TargetTimeMode = "forecast",
    ):
        super().__init__()
        self.shard = TrainingShard(
            src=src,
            dst=dst,
            prognostic_var_names=prognostic_var_names,
            boundary_var_names=boundary_var_names,
            hist=hist,
            steps=steps,
            normalize_before_mask=normalize_before_mask,
            masked_fill_value=masked_fill_value,
            stride=stride,
            shard_id=shard_id,
            sample_num=sample_num,
            sample_seed=sample_seed,
            target_time_mode=target_time_mode,
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
    def prognostic_srcs(self) -> list[CanonicalDataset]:
        return self.shard.prognostic_srcs

    @property
    def boundary_src(self) -> CanonicalDataset:
        return self.shard.boundary_src

    @property
    def wet_prognostic(self) -> list[PrognosticMask]:
        return self.shard.wet_prognostic

    @property
    def wet_surface(self) -> GridMask:
        return self.shard.wet_surface

    @property
    def num_prognostic_channels(self) -> int:
        return self.shard.num_prognostic_channels

    @property
    def num_boundary_channels(self) -> int:
        return self.shard.num_boundary_channels

    @property
    def batch_compatibility_key(self) -> str:
        return self.shard.batch_compatibility_key

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx: int):
        start_time = time.perf_counter()
        raw = RawTrainData(self.id)
        plan = self.shard.window_plan([idx])
        for step in plan.steps:
            reads = (step.input, step.boundary, step.label)
            if self._concurrent_compute:
                executor = self._get_executor()
                futures = [
                    executor.submit(use.source.read, use.request) for use in reads
                ]
                loaded = [future.result().values[0] for future in futures]
            else:
                loaded = [use.source.read(use.request).values[0] for use in reads]
            input_, boundary, label = map(torch.from_numpy, loaded)
            raw.insert(input_, boundary, label)
        raw.load_stats = LoadStats(time.perf_counter() - start_time)

        return raw

    def to_train_data(
        self, raw_train_data: RawTrainData, device: torch.device
    ) -> TrainData:
        """Convert RawTrainData to TrainData, moving tensors to the specified device.

        Args:
            raw_train_data: CPU data from worker process
            device: Target device (typically GPU) to move tensors to

        Returns:
            TrainData with tensors on the target device
        """
        return self.preparer.prepare_raw(raw_train_data, device)


@final
class TrainDataLoader:
    """Wrapper around a torch DataLoader that handles GPU post-processing.

    This class wraps a DataLoader[RawTrainData] and converts the raw data
    to TrainData by applying GPU-based normalization and masking. This allows
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
        dataloader: torch.utils.data.DataLoader[RawTrainData],
        datasets: list[TorchTrainDataset],
        device: torch.device,
    ):
        self._dataloader = dataloader
        self._datasets = {dataset.id: dataset for dataset in datasets}
        self._device = device

    def __iter__(self):
        """Iterate over the dataloader, converting RawTrainData to TrainData."""
        for raw_train_data in self._dataloader:
            dataset = self._datasets[raw_train_data.dataset_id]
            train_data = dataset.to_train_data(raw_train_data, self._device)
            yield train_data

    def __len__(self) -> int:
        return len(self._dataloader)

    def set_epoch(self, epoch: int) -> None:
        batch_sampler = self._dataloader.batch_sampler
        set_epoch = getattr(batch_sampler, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)

    def close(self) -> None:
        """Release persistent PyTorch workers owned by this loader."""
        close_pytorch_dataloader(self._dataloader)

    def __getitem__(self, index: int) -> TrainData:
        """Access a single item by index, converting RawTrainData to TrainData.

        Note: This bypasses the DataLoader's sampling/batching and directly accesses
        the underlying dataset for test purposes.
        """
        # Access the underlying dataset directly
        raw_train_data = self._dataloader.dataset[index]
        # Apply the collate function to add batch dimension (expects a list)
        collate_fn = self._dataloader.collate_fn
        if collate_fn is not None:
            raw_train_data = collate_fn([raw_train_data])
        # Get the dataset that created this raw data
        dataset = self._datasets[raw_train_data.dataset_id]
        # Convert to TrainData
        train_data = dataset.to_train_data(raw_train_data, self._device)
        return train_data

    @property
    def dataset(self):
        return self._dataloader.dataset

    @property
    def sampler(self):
        return self._dataloader.sampler
