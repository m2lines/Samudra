import itertools
import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import wait
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Self, TypeAlias, final

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from jaxtyping import Float
from torch.utils.data import BatchSampler, Dataset, Sampler, SubsetRandomSampler
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
        srcs = [src] if src is dst else [src, dst]

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

        self.wet_prognostic: list[PrognosticMask] = [
            src.masks.prognostic.to(self.device) for src in srcs
        ]
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

        self.prognostic_means = [
            flatten_to_device(src.means) for src in self._prognostic_srcs
        ]
        self.prognostic_stds = [
            flatten_to_device(dst.stds) for dst in self._prognostic_srcs
        ]

        self.boundary_means = flatten_to_device(self._boundary_src.means)
        self.boundary_stds = flatten_to_device(self._boundary_src.stds)

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
        TD = RawTrainData(self.id)

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

    def to_train_data(self, raw_train_data: RawTrainData) -> TrainData:
        train_data = TrainData(self.num_prognostic_channels)
        for input_, boundary, label in raw_train_data.raw_data:
            input_, label = self._to_example(
                input_.to(device=self.device, non_blocking=True),
                boundary.to(device=self.device, non_blocking=True),
                # If this is the same as input_, `to` should return without copying.
                label.to(device=self.device, non_blocking=True),
            )
            train_data.append(input_, label)
        train_data.load_stats = raw_train_data.load_stats
        return train_data

    def _to_example(
        self,
        # time includes (self.hist + 1) past steps and the (label) future steps
        input_: Float[torch.Tensor, "batch time variable lat lon"],
        boundary: Float[torch.Tensor, "batch time variable lat lon"],
        label: Float[torch.Tensor, "batch time variable lat lon"],
    ) -> tuple[Input, Prognostic]:
        # grab past steps and prep for model
        total_input = self._prep_tensor_steps(
            input_[:, : self.hist + 1, :, :, :],
            self.prognostic_means[0],
            self.prognostic_stds[0],
            self.wet_prognostic[0],
            boundary[:, : self.hist + 1, :, :, :],
        )
        # grab future steps, repeat as we do for input
        label = self._prep_tensor_steps(
            label[:, self.hist + 1 :, :, :, :],
            self.prognostic_means[-1],
            self.prognostic_stds[-1],
            self.wet_prognostic[-1],
        )
        return total_input, label

    def _prep_tensor_steps(
        self,
        prognostic_steps: Float[torch.Tensor, "batch time variable lat lon"],
        prognostic_means: Float[torch.Tensor, " variable"],
        prognostic_stds: Float[torch.Tensor, " variable"],
        prognostic_mask: Float[torch.Tensor, " variable"],
        boundary_steps: Float[torch.Tensor, "batch time variable lat lon"]
        | None = None,
    ) -> Input:
        """Prepare tensor steps by normalizing, masking and flattening dimensions."""

        def normalize(
            data: Float[torch.Tensor, "batch time var lat lon"],
            means: Float[torch.Tensor, " var"],
            stds: Float[torch.Tensor, " var"],
            fill_nan: bool = True,
            fill_value: float = 0.0,
        ) -> Float[torch.Tensor, "batch time var lat lon"]:
            """Normalize input data treated as torch Tensors."""
            norm = (data - means.view(1, 1, -1, 1, 1)) / stds.view(1, 1, -1, 1, 1)
            if fill_nan:
                norm = norm.nan_to_num(nan=fill_value)
            norm = norm.to(data.dtype)
            return norm

        # Normalize and mask tensors
        def normalize_and_mask(
            tensor: torch.Tensor,
            means: torch.Tensor,
            stds: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            if self.normalize_before_mask:
                tensor = normalize(tensor, means, stds)
            tensor = torch.where(mask, tensor, self.masked_fill_value)
            if not self.normalize_before_mask:
                tensor = normalize(tensor, means, stds)
            return tensor

        prognostic_steps = normalize_and_mask(
            prognostic_steps,
            prognostic_means,
            prognostic_stds,
            prognostic_mask,
        )
        if boundary_steps is not None:
            boundary_steps = normalize_and_mask(
                boundary_steps,
                self.boundary_means,
                self.boundary_stds,
                self.wet_surface,
            )

        # Flatten time and variable dimensions
        def flatten_dims(tensor: torch.Tensor) -> torch.Tensor:
            return rearrange(
                tensor, "batch time variable lat lon -> batch (time variable) lat lon"
            )

        prognostic_steps = flatten_dims(prognostic_steps)
        if boundary_steps is not None:
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


class _SimpleSubsetSampler(Sampler):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class EquivalenceGroupBatchSampler(Sampler[list[int]]):
    """Groups indices by dataset membership in ConcatDataset, batches within groups, and optionally shuffles.

    This sampler partitions dataset indices into groups based on their source dataset when using
    ConcatDataset. It creates batches within each group, then chains them together. When shuffle=True,
    batches are globally shuffled each epoch to avoid sequential group processing.

    Args:
        dataset_sizes: List of individual dataset sizes. Groups are created based on dataset boundaries,
            where each dataset forms its own equivalence group.
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle indices within groups and shuffle batches globally
        drop_last: Whether to drop incomplete batches at the end of each group
    """

    def __init__(
        self,
        dataset_sizes: list[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        super().__init__()
        self.group_size = len(dataset_sizes)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Create groups based on cumulative dataset boundaries
        cumsum = 0
        self.groups = []
        for size in dataset_sizes:
            self.groups.append(list(range(cumsum, cumsum + size)))
            cumsum += size

    @classmethod
    def from_datasets(
        cls,
        datasets: list["TorchTrainDataset"],
        group_key: Callable[["TorchTrainDataset"], Any],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ) -> Self:
        """Create sampler by grouping datasets using a key function.

        This factory method allows grouping datasets by arbitrary criteria (e.g., resolution,
        regardless of other parameters like stride). Datasets with the same key are batched together.

        Args:
            datasets: List of TorchTrainDataset instances to group
            group_key: Callable that extracts grouping key from a dataset.
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices within groups and shuffle batches globally
            drop_last: Whether to drop incomplete batches at the end of each group

        Examples:
                - lambda ds: (ds._input_src.data.sizes['lat'], ds._input_src.data.sizes['lon'])  # group by resolution
                - lambda ds: ds._input_src.data.sizes['lat']  # group by latitude size only
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle indices within groups and shuffle batches globally
            drop_last: Whether to drop incomplete batches at the end of each group

        Returns:
            EquivalenceGroupBatchSampler configured to group by the provided key

        Example:
            >>> # Group datasets by resolution, allowing different strides to be batched together
            >>> sampler = EquivalenceGroupBatchSampler.from_datasets(
            ...     datasets=dataset_list,
            ...     group_key=lambda ds: (ds._input_src.data.sizes['lat'], ds._input_src.data.sizes['lon']),
            ...     batch_size=32,
            ...     shuffle=True,
            ...     drop_last=True,
            ... )
        """
        from collections import defaultdict

        # Group indices by their key
        groups: dict[tuple, list[int]] = defaultdict(list)

        cumsum = 0
        for ds in datasets:
            key = group_key(ds)
            # Make key hashable if it isn't already
            if not isinstance(key, (int, str, tuple)):
                key = tuple(key) if hasattr(key, "__iter__") else (key,)
            groups[key].extend(range(cumsum, cumsum + len(ds)))
            cumsum += len(ds)

        # Convert groups to dataset_sizes format
        # Sort by key for deterministic ordering across runs
        sorted_groups = sorted(groups.items(), key=lambda x: str(x[0]))

        # Create instance with computed dataset_sizes
        instance = cls.__new__(cls)
        instance.batch_size = batch_size
        instance.shuffle = shuffle
        instance.drop_last = drop_last
        instance.group_size = len(sorted_groups)

        # Store the actual grouped indices (not just sizes)
        instance.groups = [indices for _, indices in sorted_groups]

        return instance

    def __iter__(self):
        # Choose sampler based on shuffle setting
        SubsetSampler = SubsetRandomSampler if self.shuffle else _SimpleSubsetSampler

        # Create batch samplers for each group
        batch_sampler = itertools.chain(
            *[
                BatchSampler(
                    SubsetSampler(group),
                    batch_size=self.batch_size,
                    drop_last=self.drop_last,
                )
                for group in self.groups
            ]
        )

        if not self.shuffle:
            # No global shuffle: return batches in sequential group order
            yield from batch_sampler
        else:
            # Shuffle batches globally to avoid sequential group processing
            # This is regenerated each epoch, giving different orderings
            all_batches = list(batch_sampler)
            random.shuffle(all_batches)
            yield from all_batches

    def __len__(self):
        """Calculate total number of batches across all groups."""
        total_batches = 0
        for group in self.groups:
            if self.drop_last:
                total_batches += len(group) // self.batch_size
            else:
                total_batches += (len(group) + self.batch_size - 1) // self.batch_size


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
