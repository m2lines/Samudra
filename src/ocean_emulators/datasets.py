import dataclasses
import logging
import os
import time
from concurrent.futures import wait
from concurrent.futures.thread import ThreadPoolExecutor
from typing import TypeAlias, final

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from jaxtyping import Float
from torch.utils.data import Dataset, IterableDataset
from xarray_einstats.einops import rearrange as xr_rearrange  # noqa: F401

from ocean_emulators.constants import (
    BoundaryVarNames,
    Example,
    GridMask,
    Input,
    LoaderVersion,
    Prognostic,
    PrognosticMask,
    PrognosticVarNames,
)
from ocean_emulators.utils.data import DataSource, LoadStats, conditional_rearrange
from ocean_emulators.utils.device import get_device, using_gpu
from ocean_emulators.utils.logging import elapsed


from numcodecs import blosc
blosc.use_threads = True
blosc.set_nthreads(
    max(1, int(os.environ.get("OCEAN_BLOSC_THREADS", min(4, os.cpu_count() or 1))))
)


logger = logging.getLogger(__name__)
_PIN_MEMORY_WARNING_EMITTED = False
_PIN_MEMORY_DISABLED = False

# Temporary speed-test source: prognostic fields remain in the configured LLC 003
# store, while every TorchTrainDataset boundary read is served by this pre-sliced
# cache instead.
_BOUNDARY_CACHE_LOCATION = (
    "/orcd/data/abodner/002/cody/LLC_patch/"
    "LLC4320_face1_i2880-3600_j720-1440_trainval_ready_"
    "20110913_20121014_t1-BOUNDARY-ONLY.zarr"
)


def _packed_channel_dim(data_array: xr.DataArray) -> str | None:
    for dim in data_array.dims:
        if dim.endswith("_channel"):
            return dim
    return None


def _dataset_to_numpy(selected: xr.Dataset, leading_dims: tuple[str, ...]) -> np.ndarray:
    arrays: list[np.ndarray] = []
    leading_dim_set = set(leading_dims)

    for data_array in selected.data_vars.values():
        if any(dim not in data_array.dims for dim in leading_dims):
            continue

        if (channel_dim := _packed_channel_dim(data_array)) is not None:
            spatial_dims = [
                dim
                for dim in data_array.dims
                if dim not in leading_dim_set and dim != channel_dim
            ]
            array = data_array.transpose(
                *leading_dims, channel_dim, *spatial_dims
            ).to_numpy()
        elif "lev" in data_array.dims:
            spatial_dims = [
                dim
                for dim in data_array.dims
                if dim not in leading_dim_set and dim != "lev"
            ]
            array = data_array.transpose(*leading_dims, "lev", *spatial_dims).to_numpy()
        else:
            spatial_dims = [dim for dim in data_array.dims if dim not in leading_dim_set]
            array = data_array.transpose(*leading_dims, *spatial_dims).to_numpy()
            array = np.expand_dims(array, axis=len(leading_dims))

        array = np.asarray(array)
        if not np.issubdtype(array.dtype, np.floating) or array.dtype.itemsize > 4:
            array = array.astype(np.float32, copy=False)
        arrays.append(array)

    if not arrays:
        raise ValueError("Dataset did not contain any compatible time-varying data variables.")

    return np.concatenate(arrays, axis=len(leading_dims))


class InferenceDataset(Dataset):
    """This class is used for inference rollouts.

    It creates rolling indices to keep track of histories/past states.
    When `inference_stride > 1`, the underlying time series is first subsampled
    before these windows are constructed.
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
        src: DataSource,
        prognostic_var_names,
        boundary_var_names,
        hist,
        normalize_before_mask,
        masked_fill_value,
        long_rollout,
        inference_stride: int = 1,
    ):
        super().__init__()
        self.device = get_device()

        self.hist = hist
        if inference_stride < 1:
            raise ValueError("inference_stride must be >= 1")
        self.inference_stride = inference_stride

        self.num_prognostic_channels = (hist + 1) * len(prognostic_var_names)
        if inference_stride > 1:
            src = src.map_data(
                lambda ds: ds.isel(time=slice(None, None, inference_stride)),
                suffix=f"inference_stride={inference_stride}",
            )
        data = src.data
        self._prognostic_src = src.filter(prognostic_var_names, prefix="prognostic")
        self._boundary_src = src.filter(boundary_var_names, prefix="boundary")
        self._times = data.time
        self.normalize_before_mask = normalize_before_mask
        self.masked_fill_value = masked_fill_value

        time_indices = np.arange(data.time.size)
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
                f"Long rollout will use input at time {data.time.values[0]} and produce"
                f" output at {data.time.values[self.hist + 1]}"
            )

        self.wet: PrognosticMask = src.masks.prognostic
        self.wet_surface: GridMask = src.masks.boundary
        self.size = len(self.rolling_indices)

        if using_gpu():
            self.wet = self.wet.pin_memory()
            self.wet_surface = self.wet_surface.pin_memory()

    def __len__(self):
        return self.size

    @property
    def initial_prognostic(self):
        x_index = self._get_x_index(0)
        data_in = self._get_prognostic(x_index)
        return data_in

    def inference_target(self, step: int | slice):
        x_index = self._get_x_index(step)
        label = self._get_label(x_index)
        return label

    def get_initial_input(self):
        data = self.__getitem__(0)[0]
        return data

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

    def merge_prognostic_and_boundary(self, prognostic: torch.Tensor, step: int):
        x_index = self._get_x_index(step)
        boundary = self._get_boundary(x_index).to(prognostic.device)
        data = torch.cat((prognostic, boundary), dim=1)
        return data

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx):
        x_index = self._get_x_index(idx)
        data_in = self._get_prognostic(x_index)
        data_in_boundary = self._get_boundary(x_index)
        data_in = torch.cat((data_in, data_in_boundary), dim=1)
        label = self._get_label(x_index)
        return (data_in, label)

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
        data_in_src = self._prognostic_src.map_data(
            lambda ds: ds.isel(time=x_index).isel(time=slice(None, self.hist + 1))
        )
        if self.normalize_before_mask:
            data_in_ds = data_in_src.normalize()
        else:
            data_in_ds = data_in_src.data

        data_in_np = _dataset_to_numpy(data_in_ds, ("window_dim", "time"))
        data_in: torch.Tensor = torch.from_numpy(data_in_np).float()
        data_in = torch.where(self.wet, data_in, self.masked_fill_value)
        if not self.normalize_before_mask:
            data_in = self._prognostic_src.normalize_with(data_in, variable_axis=2)
        data_in = rearrange(
            data_in,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )
        return data_in

    def _get_boundary(self, x_index):
        """
        This function returns the boundary condition for the current time step.

        With hist > 0, the boundary condition considered is always the last step of
        the input.
        """
        data_in_boundary_src = self._boundary_src.map_data(
            lambda ds: ds.isel(time=x_index).isel(time=slice(None, self.hist + 1))
        )
        if self.normalize_before_mask:
            data_in_boundary_ds = data_in_boundary_src.normalize()
        else:
            data_in_boundary_ds = data_in_boundary_src.data
        data_in_boundary_np = _dataset_to_numpy(
            data_in_boundary_ds,
            ("window_dim", "time"),
        )
        data_in_boundary: torch.Tensor = torch.from_numpy(data_in_boundary_np).float()
        data_in_boundary = torch.where(
            self.wet_surface, data_in_boundary, self.masked_fill_value
        )
        if not self.normalize_before_mask:
            data_in_boundary = self._boundary_src.normalize_with(
                data_in_boundary, variable_axis=2
            )
        data_in_boundary = rearrange(
            data_in_boundary,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )
        return data_in_boundary

    def _get_label(self, x_index):
        label_src = self._prognostic_src.map_data(
            lambda ds: ds.isel(time=x_index).isel(time=slice(self.hist + 1, None))
        )
        if self.normalize_before_mask:
            label_ds = label_src.normalize()
        else:
            label_ds = label_src.data
        label_np = _dataset_to_numpy(label_ds, ("window_dim", "time"))
        label: torch.Tensor = torch.from_numpy(label_np).float()
        label = torch.where(self.wet, label, self.masked_fill_value)
        if not self.normalize_before_mask:
            label = self._prognostic_src.normalize_with(label, variable_axis=2)
        label = rearrange(
            label,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )
        return label

    def get_coords_dict(self):
        return {
            co: self._prognostic_src.data[co] for co in self._prognostic_src.data.coords
        }


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
        self.raw_data: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.load_stats: LoadStats | None = None
        # Source dataset indices for debugging pathological slow samples.
        self.source_indices: list[int] = []

    def insert(self, all_prognostic: torch.Tensor, all_boundary: torch.Tensor):
        self.raw_data.append((all_prognostic, all_boundary))

    def to(self, device: torch.device):
        self.raw_data = [
            (
                all_prognostic.to(device, non_blocking=True),
                all_boundary.to(device, non_blocking=True),
            )
            for all_prognostic, all_boundary in self.raw_data
        ]

    def pin_memory(self):
        global _PIN_MEMORY_WARNING_EMITTED
        global _PIN_MEMORY_DISABLED
        if _PIN_MEMORY_DISABLED:
            return self
        try:
            self.raw_data = [
                (all_prognostic.pin_memory(), all_boundary.pin_memory())
                for all_prognostic, all_boundary in self.raw_data
            ]
        except RuntimeError as e:
            # Large multi-worker batches can intermittently fail host pinning.
            # Fall back to pageable memory so training can continue.
            if not _PIN_MEMORY_WARNING_EMITTED:
                logger.warning(
                    "Pin-memory failed for a batch; continuing without pinning. "
                    f"Error: {e}"
                )
                _PIN_MEMORY_WARNING_EMITTED = True
            _PIN_MEMORY_DISABLED = True
        return self


class TrainData:
    """A single batch of training data.

    A single batch contains multiple steps worth of `Example`s (i.e., input/output pairs). These steps are used during
    autoregressive rollout in the training and inference process.

    Constraint: The `Input` tensor is a combination of (flattened) prognostic variables (at all depth levels) and
    boundary forcings. The top `num_prognostic_channels` number of channels must be prognostic variables whereas the
    remaining bottom channels are boundary forcings.
    """

    def __init__(self, num_prognostic_channels: int):
        self.num_prognostic_channels = num_prognostic_channels
        self.example_by_step: list[Example] = []
        self.load_stats: LoadStats | None = None
        self.source_indices: list[int] = []

    def append(self, input_: Input, label: Prognostic):
        """Add another Example as a new step."""
        self.example_by_step.append((input_, label))

    def get_initial_input(self) -> Input:
        return self.get_input(0)

    def get_input(self, step: int) -> Input:
        return self[step][0]

    def get_label(self, step: int) -> Prognostic:
        return self[step][1]

    def merge_prognostic_and_boundary(self, prognostic: torch.Tensor, step: int):
        input_ = self.get_input(step)
        # Concatenation is equivalent to clone + slice assignment for ordinary
        # tensors and remains dispatchable when H/W are ShardTensor dimensions.
        boundary = input_[:, self.num_prognostic_channels :]
        return torch.cat((prognostic, boundary), dim=1)

    def values(self):
        return self.example_by_step

    def __getitem__(self, step: int) -> Example:
        """Converts index (step) into (data, label) tuple."""
        return self.example_by_step[step]

    def __len__(self) -> int:
        return len(self.example_by_step)

    def __iter__(self):
        return iter(range(len(self)))

    def to(self, device: torch.device) -> None:
        for step in self:
            self.example_by_step[step] = (
                self[step][0].to(device, non_blocking=True),
                self[step][1].to(device, non_blocking=True),
            )

    def pin_memory(self):
        for step in self:
            self.example_by_step[step] = (
                self[step][0].pin_memory(),
                self[step][1].pin_memory(),
            )
        return self


@dataclasses.dataclass(frozen=True)
class ReplayCursor:
    dataset_index: int
    source_index: int
    lead_step: int
    stride: int
    temporal_stride: int

    def advance(self) -> "ReplayCursor":
        return dataclasses.replace(self, lead_step=self.lead_step + 1)


@dataclasses.dataclass(frozen=True)
class ReplayBatchSlot:
    replay_index: int
    cursor: ReplayCursor


@dataclasses.dataclass(frozen=True)
class ReplaySeedSlot:
    replay_index: int
    cursor: ReplayCursor
    reason: str


@dataclasses.dataclass(frozen=True)
class ReplayBatchRequest:
    request_id: int
    train_slots: tuple[ReplayBatchSlot, ...]
    seed_slots: tuple[ReplaySeedSlot, ...]
    temporal_bundle_size: int = 1

    @property
    def reserved_indices(self) -> set[int]:
        return {
            slot.replay_index
            for slot in (*self.train_slots, *self.seed_slots)
        }


@dataclasses.dataclass
class RawReplayTransition:
    dataset_id: "TorchTrainDataset.Id"
    dataset_index: int
    source_index: int
    lead_step: int
    current_time_index: int
    target_time_index: int
    target_prognostic: torch.Tensor | None = None
    seed_prognostic: torch.Tensor | None = None
    boundary: torch.Tensor | None = None

    def pin_memory(self):
        if self.target_prognostic is not None:
            self.target_prognostic = self.target_prognostic.pin_memory()
        if self.seed_prognostic is not None:
            self.seed_prognostic = self.seed_prognostic.pin_memory()
        if self.boundary is not None:
            self.boundary = self.boundary.pin_memory()
        return self


@dataclasses.dataclass
class RawReplayBatch:
    request: ReplayBatchRequest
    train_transitions: list[RawReplayTransition]
    seed_transitions: list[RawReplayTransition]
    load_stats: LoadStats | None = None

    def pin_memory(self):
        global _PIN_MEMORY_WARNING_EMITTED
        global _PIN_MEMORY_DISABLED
        if _PIN_MEMORY_DISABLED:
            return self
        try:
            for transition in (*self.train_transitions, *self.seed_transitions):
                transition.pin_memory()
        except RuntimeError as e:
            if not _PIN_MEMORY_WARNING_EMITTED:
                logger.warning(
                    "Pin-memory failed for a replay batch; continuing without "
                    f"pinning. Error: {e}"
                )
                _PIN_MEMORY_WARNING_EMITTED = True
            _PIN_MEMORY_DISABLED = True
        return self


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

    Id: TypeAlias = str

    FLAG = LoaderVersion.OM4_TORCH

    @staticmethod
    def _with_boundary_cache(boundary_src: DataSource) -> DataSource:
        """Replace a 003 boundary view with the pre-sliced boundary test cache."""
        cache = xr.open_zarr(_BOUNDARY_CACHE_LOCATION, chunks=None)
        channel_coords = boundary_src.data["boundary_channel"]
        source_times = boundary_src.data.time.values
        cache_times = (
            source_times
            if np.issubdtype(source_times.dtype, np.datetime64)
            else np.asarray([np.datetime64(str(time)) for time in source_times])
        )
        cache_boundary = cache.sel(
            boundary_channel=channel_coords,
            time=cache_times,
        )

        if "boundary" not in cache_boundary:
            raise ValueError(
                f"Boundary cache {_BOUNDARY_CACHE_LOCATION} has no boundary field"
            )

        boundary_mask = torch.from_numpy(
            cache_boundary["boundary_mask"].to_numpy().astype(bool, copy=False)
        )
        return dataclasses.replace(
            boundary_src,
            name=f"boundary-cache[{boundary_src.name}]",
            data=cache_boundary[["boundary"]],
            means=cache_boundary[["boundary_mean"]],
            stds=cache_boundary[["boundary_std"]],
            masks=dataclasses.replace(boundary_src.masks, boundary=boundary_mask),
        )

    @elapsed
    def __init__(
        self,
        src: DataSource,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        hist: int,
        steps: int,
        normalize_before_mask: bool,
        masked_fill_value: float,
        stride: int = 1,
        temporal_stride: int = 1,
        executor: ThreadPoolExecutor | None = None,
    ):
        super().__init__()
        self.id = f"{self.__class__.__name__}_{str(id(self))}"
        self.device = get_device()

        self.hist: int = hist
        self.steps: int = steps
        self.stride: int = stride
        if temporal_stride < 1:
            raise ValueError("temporal_stride must be >= 1")
        self.temporal_stride: int = temporal_stride
        self.normalize_before_mask: bool = normalize_before_mask
        self.masked_fill_value: float = masked_fill_value
        self._executor = executor

        self.num_prognostic_channels: int = (hist + 1) * len(prognostic_var_names)
        data = src.data
        self._prognostic_src = src.filter(prognostic_var_names, prefix="prognostic")
        self._boundary_src = self._with_boundary_cache(
            src.filter(boundary_var_names, prefix="boundary")
        )

        # This class will be used only for training and validation
        total_steps: int = 2 * self.hist + 2

        # Calculate the number of windows
        num_windows = data.time.size - (total_steps - 1) * self.stride

        # Create base indices
        indices = np.arange(num_windows)
        indices_da = xr.DataArray(indices, dims=["window"])

        # Create window dimension
        window_dim = xr.DataArray(np.arange(total_steps), dims=["time"])

        # Construct rolling indices
        self.rolling_indices: Float[xr.DataArray, "window time"] = (
            indices_da + stride * window_dim
        )

        self.wet: PrognosticMask = src.masks.prognostic.to(self.device)
        self.wet_surface: GridMask = src.masks.boundary.to(self.device)

        def flatten_to_device(means_or_stds: xr.Dataset) -> torch.Tensor:
            if "lev" in means_or_stds.dims:
                array = conditional_rearrange(
                    means_or_stds,
                    "(variable lev)=var",
                    concat_dim="var",
                ).rename({"var": "variable"})
            else:
                array = means_or_stds.to_dataarray()
            return torch.from_numpy(array.to_numpy().flatten()).to(self.device)

        self.prognostic_means = flatten_to_device(self._prognostic_src.means)
        self.prognostic_stds = flatten_to_device(self._prognostic_src.stds)

        self.boundary_means = flatten_to_device(self._boundary_src.means)
        self.boundary_stds = flatten_to_device(self._boundary_src.stds)

        base_size = (
            data.time.size
            - self.steps * (self.hist + 1) * self.stride
            - self.hist * self.stride
        )
        self.size: int = max(0, (base_size + self.temporal_stride - 1) // self.temporal_stride)

    def __len__(self) -> int:
        return self.size

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx: int):
        start_time = time.perf_counter()
        TD = RawTrainData(self.id)
        TD.source_indices.append(idx)

        for step in range(self.steps):
            x_index = self._get_x_index(idx, step)
            current_x_index = x_index.isel(time=slice(0, self.hist + 1))
            prognostic_selected = self._prognostic_src.data.isel(time=x_index)
            boundary_selected = self._boundary_src.data.isel(time=current_x_index)

            if self._executor is not None:
                concurrent_compute(
                    prognostic_selected, boundary_selected, executor=self._executor
                )

            prognostic_all = self._dataset_to_tensor(prognostic_selected)
            boundary = self._dataset_to_tensor(boundary_selected)

            TD.insert(prognostic_all, boundary)
        TD.load_stats = LoadStats(time.perf_counter() - start_time)

        return TD

    def _dataset_to_tensor(self, selected: xr.Dataset) -> torch.Tensor:
        return torch.from_numpy(_dataset_to_numpy(selected, ("time",)))

    def get_replay_time_indices(
        self,
        *,
        source_index: int,
        lead_step: int,
    ) -> tuple[int, int]:
        x_index = self._get_x_index(source_index, lead_step)
        values = x_index.values
        return int(values[self.hist]), int(values[self.hist + 1])

    def get_raw_replay_transition(
        self,
        *,
        dataset_index: int,
        source_index: int,
        lead_step: int,
    ) -> RawReplayTransition:
        return self.get_raw_replay_train_transition(
            dataset_index=dataset_index,
            source_index=source_index,
            lead_step=lead_step,
        )

    def get_raw_replay_train_transition(
        self,
        *,
        dataset_index: int,
        source_index: int,
        lead_step: int,
    ) -> RawReplayTransition:
        x_index = self._get_x_index(source_index, lead_step)
        values = x_index.values
        t_cur = int(values[self.hist])
        t_tgt = int(values[self.hist + 1])

        # Basic indexing keeps a length-1 time dim and avoids vectorized/fancy isel.
        target_selected = self._prognostic_src.data.isel(time=slice(t_tgt, t_tgt + 1))
        boundary_selected = self._boundary_src.data.isel(time=slice(t_cur, t_cur + 1))

        target_prognostic = self._dataset_to_tensor(target_selected)
        boundary = self._dataset_to_tensor(boundary_selected)

        return RawReplayTransition(
            dataset_id=self.id,
            dataset_index=dataset_index,
            source_index=source_index,
            lead_step=lead_step,
            current_time_index=t_cur,
            target_time_index=t_tgt,
            target_prognostic=target_prognostic,
            boundary=boundary,
        )

    def get_raw_replay_seed_transition(
        self,
        *,
        dataset_index: int,
        source_index: int,
        lead_step: int,
    ) -> RawReplayTransition:
        x_index = self._get_x_index(source_index, lead_step)
        values = x_index.values
        t_cur = int(values[self.hist])
        t_tgt = int(values[self.hist + 1])
        # Basic indexing keeps a length-1 time dim and avoids vectorized/fancy isel.
        seed_selected = self._prognostic_src.data.isel(time=slice(t_cur, t_cur + 1))

        seed_prognostic = self._dataset_to_tensor(seed_selected)
        return RawReplayTransition(
            dataset_id=self.id,
            dataset_index=dataset_index,
            source_index=source_index,
            lead_step=lead_step,
            current_time_index=t_cur,
            target_time_index=t_tgt,
            seed_prognostic=seed_prognostic,
        )

    def to_train_data(self, raw_train_data: RawTrainData) -> TrainData:
        train_data = TrainData(self.num_prognostic_channels)
        for prognostic_all, boundary_all in raw_train_data.raw_data:
            input, label = self._get_input_and_label(
                prognostic_all.to(device=self.device, non_blocking=True),
                boundary_all.to(device=self.device, non_blocking=True),
            )
            train_data.append(input, label)
        train_data.load_stats = raw_train_data.load_stats
        train_data.source_indices = list(raw_train_data.source_indices)
        return train_data

    def get_replay_transition(
        self,
        source_index: int,
        lead_step: int,
        prognostic_state: torch.Tensor | None = None,
    ) -> Example:
        x_index = self._get_x_index(source_index, lead_step)
        current_x_index = x_index.isel(time=slice(0, self.hist + 1))
        prognostic_selected = self._prognostic_src.data.isel(time=x_index)
        boundary_selected = self._boundary_src.data.isel(time=current_x_index)

        if self._executor is not None:
            concurrent_compute(
                prognostic_selected, boundary_selected, executor=self._executor
            )

        prognostic_all = self._dataset_to_tensor(prognostic_selected).unsqueeze(0)
        boundary_all = self._dataset_to_tensor(boundary_selected).unsqueeze(0)
        input, label = self._get_input_and_label(
            prognostic_all.to(device=self.device, non_blocking=True),
            boundary_all.to(device=self.device, non_blocking=True),
        )
        if prognostic_state is not None:
            if prognostic_state.ndim == 3:
                prognostic_state = prognostic_state.unsqueeze(0)
            if prognostic_state.shape != input[:, : self.num_prognostic_channels].shape:
                raise ValueError(
                    "Replay prognostic state shape does not match model input: "
                    f"{prognostic_state.shape} vs "
                    f"{input[:, : self.num_prognostic_channels].shape}"
                )
            input = input.clone()
            input[:, : self.num_prognostic_channels] = prognostic_state.to(
                device=input.device,
                dtype=input.dtype,
                non_blocking=True,
            )
        return input, label

    def get_replay_initial_state(self, source_index: int) -> Prognostic:
        input, _ = self.get_replay_transition(source_index, lead_step=0)
        return input[:, : self.num_prognostic_channels]

    def remask_prognostic_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 3:
            squeeze_batch = True
            state = state.unsqueeze(0)
        else:
            squeeze_batch = False

        mask = torch.cat([self.wet] * (self.hist + 1), dim=0).to(
            device=state.device,
            dtype=torch.bool,
        )
        if self.normalize_before_mask:
            fill = torch.full_like(state, self.masked_fill_value)
        else:
            fill_value = (
                (self.masked_fill_value - self.prognostic_means)
                / self.prognostic_stds
            )
            fill_value = torch.cat([fill_value] * (self.hist + 1), dim=0)
            fill = fill_value.to(device=state.device, dtype=state.dtype).view(
                1, -1, 1, 1
            )
            fill = fill.expand_as(state)

        masked = torch.where(mask.unsqueeze(0), state, fill)
        return masked.squeeze(0) if squeeze_batch else masked

    def _get_input_and_label(
        self,
        # time includes (self.hist + 1) past steps and the (label) future steps
        prognostic_all: Float[torch.Tensor, "batch time variable lat lon"],
        boundary_all: Float[torch.Tensor, "batch time variable lat lon"],
    ) -> tuple[Input, Prognostic]:
        # grab past steps and prep for model
        total_input = self._prep_tensor_steps(
            prognostic_all[:, : self.hist + 1, :, :, :],
            boundary_all[:, : self.hist + 1, :, :, :],
        )
        # grab future steps, repeat as we do for input
        label = self._prep_tensor_steps(
            prognostic_all[:, self.hist + 1 :, :, :, :]
        ).to(dtype=torch.float32)
        return total_input, label

    def _prep_tensor_steps(
        self,
        prognostic_steps: Float[torch.Tensor, "batch time variable lat lon"],
        boundary_steps: Float[torch.Tensor, "batch time variable lat lon"]
        | None = None,
    ) -> Input:
        """Prepare tensor steps by normalizing, masking and flattening dimensions."""

        prognostic_steps = self._normalize_and_mask_steps(
            prognostic_steps,
            self.prognostic_means,
            self.prognostic_stds,
            self.wet,
        )
        if boundary_steps is not None:
            boundary_steps = self._normalize_and_mask_steps(
                boundary_steps,
                self.boundary_means,
                self.boundary_stds,
                self.wet_surface,
            )

        prognostic_steps = self._flatten_steps(prognostic_steps)
        if boundary_steps is not None:
            boundary_steps = self._flatten_steps(boundary_steps)
            return torch.cat((prognostic_steps, boundary_steps), dim=1)

        return prognostic_steps

    def _prep_boundary_steps(
        self,
        boundary_steps: Float[torch.Tensor, "batch time variable lat lon"],
    ) -> Input:
        boundary_steps = self._normalize_and_mask_steps(
            boundary_steps,
            self.boundary_means,
            self.boundary_stds,
            self.wet_surface,
        )
        return self._flatten_steps(boundary_steps)

    def _normalize_and_mask_steps(
        self,
        tensor: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.normalize_before_mask:
            tensor = self._normalize_steps(tensor, means, stds)
        tensor = torch.where(mask, tensor, self.masked_fill_value)
        if not self.normalize_before_mask:
            tensor = self._normalize_steps(tensor, means, stds)
        return tensor

    @staticmethod
    def _normalize_steps(
        data: Float[torch.Tensor, "batch time var lat lon"],
        means: Float[torch.Tensor, " var"],
        stds: Float[torch.Tensor, " var"],
        fill_nan: bool = True,
        fill_value: float = 0.0,
    ) -> Float[torch.Tensor, "batch time var lat lon"]:
        compute_dtype = (
            torch.float32
            if data.dtype in (torch.float16, torch.bfloat16)
            else data.dtype
        )
        data_for_norm = data.to(dtype=compute_dtype)
        means = means.to(device=data.device, dtype=compute_dtype)
        stds = stds.to(device=data.device, dtype=compute_dtype)
        norm = (data_for_norm - means.view(1, 1, -1, 1, 1)) / stds.view(
            1, 1, -1, 1, 1
        )
        if fill_nan:
            norm = norm.nan_to_num(nan=fill_value)
        return norm.to(data.dtype)

    @staticmethod
    def _flatten_steps(tensor: torch.Tensor) -> torch.Tensor:
        return rearrange(
            tensor, "batch time variable lat lon -> batch (time variable) lat lon"
        )

    def _get_x_index(self, idx: int, step: int) -> xr.DataArray:
        assert isinstance(idx, int)
        if idx < 0:
            raise IndexError("Sorry, negative indexing is not supported!")
        if idx >= len(self):
            raise IndexError("Index out of range")

        # Subsample training window start positions in time.
        window_index = (
            idx * self.temporal_stride + step * (self.hist + 1) * self.stride
        )
        return self.rolling_indices.isel(window=window_index, drop=True)


@final
class ReplayRequestDataset(IterableDataset[RawReplayBatch]):
    """Worker-side replay loader fed by deterministic trainer requests."""

    def __init__(
        self,
        datasets: list[TorchTrainDataset],
        request_queue,
    ) -> None:
        super().__init__()
        self._datasets = datasets
        self._request_queue = request_queue

    def __iter__(self):
        while True:
            request = self._request_queue.get()
            if request is None:
                return

            start_time = time.perf_counter()
            train_transitions = [
                self._load_train_transition(slot.cursor)
                for slot in request.train_slots
            ]
            seed_transitions = [
                self._load_seed_transition(slot.cursor)
                for slot in request.seed_slots
            ]
            yield RawReplayBatch(
                request=request,
                train_transitions=train_transitions,
                seed_transitions=seed_transitions,
                load_stats=LoadStats(time.perf_counter() - start_time),
            )

    def _load_train_transition(self, cursor: ReplayCursor) -> RawReplayTransition:
        dataset = self._datasets[cursor.dataset_index]
        return dataset.get_raw_replay_train_transition(
            dataset_index=cursor.dataset_index,
            source_index=cursor.source_index,
            lead_step=cursor.lead_step,
        )

    def _load_seed_transition(self, cursor: ReplayCursor) -> RawReplayTransition:
        dataset = self._datasets[cursor.dataset_index]
        return dataset.get_raw_replay_seed_transition(
            dataset_index=cursor.dataset_index,
            source_index=cursor.source_index,
            lead_step=cursor.lead_step,
        )


def concurrent_compute(
    *datasets: xr.Dataset,
    executor: ThreadPoolExecutor,
) -> None:
    def load_variable_data(var: xr.Variable) -> None:
        var.load()

    futures = []
    for ds in datasets:
        for var in ds.variables.values():
            futures.append(executor.submit(load_variable_data, var))

    wait(futures)


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
            train_data = dataset.to_train_data(raw_train_data)
            yield train_data

    def __len__(self) -> int:
        return len(self._dataloader)

    def __getitem__(self, index: int) -> TrainData:
        """Access a single item by index, converting RawTrainData to TrainData.

        Note: This bypasses the DataLoader's sampling/batching and directly accesses
        the underlying dataset for test purposes.
        """
        # Access the underlying dataset directly
        raw_train_data = self._dataloader.dataset[index]
        # Apply collate function to add batch dimension (expects a list)
        collate_fn = self._dataloader.collate_fn
        if collate_fn is not None:
            raw_train_data = collate_fn([raw_train_data])
        # Get the dataset that created this raw data
        dataset = self._datasets[raw_train_data.dataset_id]
        # Convert to TrainData
        train_data = dataset.to_train_data(raw_train_data)
        return train_data

    @property
    def dataset(self):
        return self._dataloader.dataset

    @property
    def sampler(self):
        return self._dataloader.sampler
