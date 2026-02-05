import dataclasses
import logging
import time
from concurrent.futures import wait
from concurrent.futures.thread import ThreadPoolExecutor
from typing import TypeAlias, final

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from jaxtyping import Float
from torch.utils.data import Dataset
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
from ocean_emulators.utils.data import (
    DataSource,
    LoadStats,
    OceanData,
    _flatten,
    conditional_rearrange,
)
from ocean_emulators.utils.device import get_device, using_gpu
from ocean_emulators.utils.logging import elapsed

logger = logging.getLogger(__name__)


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
        src: DataSource,
        prognostic_var_names,
        boundary_var_names,
        hist,
        normalize_before_mask,
        masked_fill_value,
        long_rollout,
    ):
        super().__init__()
        self.device = get_device()

        self.hist = hist

        self.num_prognostic_channels = (hist + 1) * len(prognostic_var_names)
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

        if "lev" in data_in_ds.dims:
            data_in_np: np.ndarray = (
                conditional_rearrange(
                    data_in_ds,
                    "window_dim time (variable lev)=var lat lon",
                    concat_dim="var",
                )
                .rename({"var": "variable"})
                .to_numpy()
            )
        else:
            data_in_np = (
                data_in_ds.to_array()
                .transpose("window_dim", "time", "variable", "lat", "lon")
                .to_numpy()
            )
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
        data_in_boundary_np: np.ndarray = (
            data_in_boundary_ds.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
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
        if "lev" in label_ds.dims:
            label_np: np.ndarray = (
                conditional_rearrange(
                    label_ds,
                    "window_dim time (variable lev)=var lat lon",
                    concat_dim="var",
                )
                .rename({"var": "variable"})
                .to_numpy()
            )
        else:
            label_np = (
                label_ds.to_array()
                .transpose("window_dim", "time", "variable", "lat", "lon")
                .to_numpy()
            )
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
    def __init__(self, dataset_id: "TorchTrainDataset.Id", label_mask: PrognosticMask):
        self.dataset_id: TorchTrainDataset.Id = dataset_id
        self.label_mask = label_mask
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

    A single batch contains multiple steps worth of `Example`s (i.e., input/output pairs). These steps are used during
    autoregressive rollout in the training and inference process.

    Constraint: The `Input` tensor is a combination of (flattened) prognostic variables (at all depth levels) and
    boundary forcings. The top `num_prognostic_channels` number of channels must be prognostic variables whereas the
    remaining bottom channels are boundary forcings.
    """

    def __init__(self, num_prognostic_channels: int, label_mask: PrognosticMask):
        self.num_prognostic_channels = num_prognostic_channels
        self.label_mask = label_mask
        self.example_by_step: list[Example] = []
        self.load_stats: LoadStats | None = None

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
        merged = input_.clone()
        merged[:, : self.num_prognostic_channels] = prognostic
        return merged

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


@dataclasses.dataclass
class _NormCache:
    prognostic_means: list[torch.Tensor]
    prognostic_stds: list[torch.Tensor]
    boundary_means: torch.Tensor
    boundary_stds: torch.Tensor
    wet_prognostic: list[PrognosticMask]
    wet_surface: GridMask


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

    @elapsed
    def __init__(
        self,
        src: DataSource,
        dst: DataSource,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        hist: int,
        steps: int,
        normalize_before_mask: bool,
        masked_fill_value: float,
        stride: int = 1,
        executor: ThreadPoolExecutor | None = None,
    ):
        super().__init__()
        self.id = f"{self.__class__.__name__}_{str(id(self))}"
        self.device = get_device()
        # If the src and dst DataSource are the same, we can do a lot less work.
        srcs = [src] if src.name == dst.name else [src, dst]

        self.hist: int = hist
        self.steps: int = steps
        self.stride: int = stride
        self.normalize_before_mask: bool = normalize_before_mask
        self.masked_fill_value: float = masked_fill_value
        self._executor = executor

        self.num_prognostic_channels: int = (hist + 1) * len(prognostic_var_names)
        assert np.array_equal(src.data.time, dst.data.time), (
            "src and dst DataSource have different time slices!"
        )
        time_ = src.data.time
        self._prognostic_srcs = [
            src.filter(prognostic_var_names, prefix="prog") for src in srcs
        ]
        self._boundary_src = src.filter(boundary_var_names, prefix="boundary")

        # This class will be used only for training and validation
        total_steps: int = 2 * self.hist + 2

        # Calculate the number of windows
        num_windows = time_.size - (total_steps - 1) * self.stride

        # Create base indices
        indices = np.arange(num_windows)
        indices_da = xr.DataArray(indices, dims=["window"])

        # Create window dimension
        window_dim = xr.DataArray(np.arange(total_steps), dims=["time"])

        # Construct rolling indices
        self.rolling_indices: Float[xr.DataArray, "window time"] = (
            indices_da + stride * window_dim
        )

        # Cache masks and normalization stats on CPU (device transfer is lazy).
        self._cpu_cache = _NormCache(
            prognostic_means=[_flatten(src.means) for src in self._prognostic_srcs],
            prognostic_stds=[_flatten(src.stds) for src in self._prognostic_srcs],
            boundary_means=_flatten(self._boundary_src.means),
            boundary_stds=_flatten(self._boundary_src.stds),
            wet_prognostic=[src.masks.prognostic for src in srcs],
            wet_surface=src.masks.boundary,
        )
        self._device_cache: _NormCache | None = None
        self._device_cache_device: torch.device | None = None

        self.size: int = (
            time_.size
            - self.steps * (self.hist + 1) * self.stride
            - self.hist * self.stride
        )

    def __len__(self) -> int:
        return self.size

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx: int):
        start_time = time.perf_counter()
        TD = RawTrainData(
            self.id, self._prognostic_srcs[-1].masks.prognostic_with_hist(self.hist)
        )

        for step in range(self.steps):
            x_index = self._get_x_index(idx, step)
            prognostic_selected = [
                src.data.isel(time=x_index) for src in self._prognostic_srcs
            ]
            boundary_selected = self._boundary_src.data.isel(time=x_index)

            if self._executor is not None:
                datasets = prognostic_selected + [boundary_selected]
                concurrent_compute(
                    *datasets,
                    executor=self._executor,
                )

            if "lev" in prognostic_selected[0].dims:
                prognostics = [
                    torch.from_numpy(
                        conditional_rearrange(
                            selected,
                            "time (variable lev)=var lat lon",
                            concat_dim="var",
                        )
                        .rename({"var": "variable"})
                        .to_numpy()
                        .astype(np.float32, copy=False)
                    )
                    for selected in prognostic_selected
                ]
            else:
                prognostics = [
                    torch.from_numpy(
                        selected.to_array()
                        .transpose("time", "variable", "lat", "lon")
                        .to_numpy()
                        .astype(np.float32, copy=False)
                    )
                    for selected in prognostic_selected
                ]
            boundary = torch.from_numpy(
                boundary_selected.to_array()
                .transpose("time", "variable", "lat", "lon")
                .to_numpy()
                .astype(np.float32, copy=False)
            )
            input_, label = prognostics[0], prognostics[-1]
            TD.insert(input_, boundary, label)
        TD.load_stats = LoadStats(time.perf_counter() - start_time)

        return TD

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
        train_data = TrainData(self.num_prognostic_channels, raw_train_data.label_mask)

        cache = self._get_norm_cache(device)
        for input_, boundary, label in raw_train_data.raw_data:
            input_, label = self._to_example(
                OceanData(
                    input_,
                    cache.prognostic_means[0],
                    cache.prognostic_stds[0],
                    cache.wet_prognostic[0],
                ).to(device=device, non_blocking=True),
                OceanData(
                    boundary,
                    cache.boundary_means,
                    cache.boundary_stds,
                    cache.wet_surface,
                ).to(device=device, non_blocking=True),
                OceanData(
                    label,
                    cache.prognostic_means[-1],
                    cache.prognostic_stds[-1],
                    cache.wet_prognostic[-1],
                ).to(device=device, non_blocking=True),
            )
            train_data.append(input_, label)
        train_data.load_stats = raw_train_data.load_stats
        return train_data

    def _get_norm_cache(self, device: torch.device) -> _NormCache:
        if self._device_cache is not None and self._device_cache_device == device:
            return self._device_cache

        if device.type == "cpu":
            self._device_cache = self._cpu_cache
            self._device_cache_device = device
            return self._device_cache

        self._device_cache = _NormCache(
            prognostic_means=[
                t.to(device, non_blocking=True)
                for t in self._cpu_cache.prognostic_means
            ],
            prognostic_stds=[
                t.to(device, non_blocking=True) for t in self._cpu_cache.prognostic_stds
            ],
            boundary_means=self._cpu_cache.boundary_means.to(device, non_blocking=True),
            boundary_stds=self._cpu_cache.boundary_stds.to(device, non_blocking=True),
            wet_prognostic=[
                t.to(device, non_blocking=True) for t in self._cpu_cache.wet_prognostic
            ],
            wet_surface=self._cpu_cache.wet_surface.to(device, non_blocking=True),
        )
        self._device_cache_device = device
        return self._device_cache

    def _to_example(
        self,
        # time includes (self.hist + 1) past steps and the (label) future steps
        input_: OceanData,
        boundary: OceanData,
        label: OceanData,
    ) -> tuple[Input, Prognostic]:
        # Move normalization parameters to the same device as input data
        # grab past steps and prep for model
        total_input = self._prep_tensor_steps(
            input_.with_time(slice(0, self.hist + 1)),
            boundary.with_time(slice(0, self.hist + 1)),
        )
        # grab future steps, repeat as we do for input
        label_tensor = self._prep_tensor_steps(
            label.with_time(slice(self.hist + 1, None))
        )
        return total_input, label_tensor

    def _prep_tensor_steps(
        self,
        prognostic: OceanData,
        boundary: OceanData | None = None,
    ) -> Input:
        """Prepare tensor steps by normalizing, masking and flattening dimensions."""
        prognostic_steps = prognostic.normalize_and_mask(
            self.normalize_before_mask, self.masked_fill_value
        )

        # Flatten time and variable dimensions
        def flatten_dims(tensor: torch.Tensor) -> torch.Tensor:
            return rearrange(
                tensor, "batch time variable lat lon -> batch (time variable) lat lon"
            )

        prognostic_steps = flatten_dims(prognostic_steps)
        if boundary is not None:
            boundary_steps = boundary.normalize_and_mask(
                self.normalize_before_mask, self.masked_fill_value
            )
            boundary_steps = flatten_dims(boundary_steps)
            return torch.cat((prognostic_steps, boundary_steps), dim=1)

        return prognostic_steps

    def _get_x_index(self, idx: int, step: int) -> xr.DataArray:
        assert isinstance(idx, int)
        if idx < 0:
            raise IndexError("Sorry, negative indexing is not supported!")
        if idx >= len(self):
            raise IndexError("Index out of range")

        window_index = idx + step * (self.hist + 1) * self.stride
        return self.rolling_indices.isel(window=window_index, drop=True)


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
            train_data = dataset.to_train_data(raw_train_data, self._device)
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
