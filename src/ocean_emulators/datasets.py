import itertools
import logging
import time
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import wait
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

import numpy as np
import spdl
import torch
import xarray as xr
from einops import rearrange
from jaxtyping import Float
from spdl.pipeline import PipelineBuilder
from spdl.pipeline._pipeline import PipelineIterator
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
)
from xarray_einstats.einops import rearrange as xr_rearrange  # noqa: F401

from ocean_emulators.constants import (
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
        wet,
        wet_surface,
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

        self.wet: torch.Tensor = wet.bool()
        self.wet_surface: torch.Tensor = wet_surface.bool()
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


class TrainData:
    def __init__(self, num_prognostic_channels: int):
        self.td_dict: dict[int, Example] = {}
        self.load_stats: LoadStats | None = None
        self.num_prognostic_channels = num_prognostic_channels
        self.steps = 0

    def insert(self, input_: Input, label: Prognostic):
        self.td_dict[self.steps] = (input_, label)
        self.steps += 1

    def get_initial_input(self) -> Input:
        return self.td_dict[0][0]

    def get_input(self, step: int) -> Input:
        return self.td_dict[step][0]

    def get_label(self, step: int) -> Prognostic:
        return self.td_dict[step][1]

    def merge_prognostic_and_boundary(self, prognostic: torch.Tensor, step: int):
        input, _ = self.td_dict[step]
        merged = input.clone()
        merged[:, : self.num_prognostic_channels] = prognostic
        return merged

    def __getitem__(self, step: int) -> Example:
        """Converts index (step) into (data, label) tuple."""
        return self.td_dict[step]

    def __len__(self) -> int:
        return self.steps

    def to(self, device: torch.device) -> None:
        for step in self.td_dict:
            self.td_dict[step] = (
                self.td_dict[step][0].to(device),
                self.td_dict[step][1].to(device),
            )


class TrainDataset(Dataset):
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

    FLAG = LoaderVersion.OM4_EAGER

    @elapsed
    def __init__(
        self,
        src: DataSource,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        wet: PrognosticMask,
        wet_surface: GridMask,
        hist: int,
        steps: int,
        normalize_before_mask: bool,
        masked_fill_value: float,
        stride: int = 1,
    ):
        super().__init__()
        self.device = get_device()

        self.hist: int = hist
        self.steps: int = steps
        self.stride: int = stride
        self.normalize_before_mask: bool = normalize_before_mask
        self.masked_fill_value: float = masked_fill_value
        data = src.data
        self._prognostic_src = src.filter(prognostic_var_names, prefix="prognostic")
        self._boundary_src = src.filter(boundary_var_names, prefix="boundary")

        self.num_prognostic_channels: int = (hist + 1) * len(prognostic_var_names)

        # This class will be used only for training
        total_steps: int = 2 * self.hist + 2

        # Calculate the number of windows
        num_windows = data.time.size - (total_steps - 1) * self.stride

        # Create base indices
        indices = np.arange(num_windows)
        indices_da = xr.DataArray(indices, dims=["window_dim"])

        # Create window dimension
        window_dim = xr.DataArray(np.arange(total_steps), dims=["time"])

        # Construct rolling indices
        self.rolling_indices: Float[xr.DataArray, "window_dim time"] = (
            indices_da + stride * window_dim
        )

        self.wet = wet.bool()
        self.wet_surface = wet_surface.bool()

        self.size: int = (
            data.time.size
            - self.steps * (self.hist + 1) * self.stride
            - self.hist * self.stride
        )

        if using_gpu():
            self.wet = self.wet.pin_memory()
            self.wet_surface = self.wet_surface.pin_memory()

    def __len__(self) -> int:
        return self.size

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx: int):
        start_time = time.perf_counter()
        TD = TrainData(self.num_prognostic_channels)
        prev_rolling_idx = None
        for step in range(self.steps):
            x_index = self._get_x_index(idx, step, prev_rolling_idx)

            data_in: Prognostic = self._get_input(x_index)
            data_in_boundary: Boundary = self._get_boundary(x_index)

            data_combined: Input = torch.cat(
                (data_in, data_in_boundary), dim=1
            ).squeeze()

            label: Prognostic = self._get_label(x_index)

            TD.insert(
                input_=data_combined,
                label=label,
            )

        TD.load_stats = LoadStats(time.perf_counter() - start_time)

        return TD

    def _get_x_index(
        self, idx: int, step: int, prev_rolling_idx: int | None
    ) -> xr.Variable:
        assert isinstance(idx, int)
        if idx < 0:
            raise IndexError("Sorry, negative indexing is not supported!")
        if idx >= len(self):
            raise IndexError("Index out of range")

        start = idx + step * (self.hist + 1) * self.stride
        end = start + 1
        # Create a slice for similar indexing as in InferenceDataset
        idx_slice = slice(start, end)
        rolling_idx = self.rolling_indices.isel(window_dim=idx_slice)

        x_index = xr.Variable(["window_dim", "time"], rolling_idx)
        return x_index

    def _get_input(self, x_index) -> Prognostic:
        # TODO(jder): nicer typing
        data_in_src: Any = self._prognostic_src.map_data(
            lambda ds: ds.isel(time=x_index).isel(time=slice(None, self.hist + 1))
        )
        if self.normalize_before_mask:
            data_in = data_in_src.normalize()
        else:
            data_in = data_in_src.data

        data_in = (
            data_in.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in = torch.from_numpy(data_in).float()
        data_in = torch.where(self.wet, data_in, self.masked_fill_value)
        if not self.normalize_before_mask:
            data_in = self._prognostic_src.normalize_with(data_in, variable_axis=2)

        data_in = rearrange(
            data_in,
            "window_dim time variable lat lon -> \
                window_dim (time variable) lat lon",
        )
        return data_in

    def _get_boundary(self, x_index) -> Boundary:
        """
        This function returns the boundary condition for the current time step.

        With hist > 0, the boundary condition considered is always the last step of
        the input.
        """
        # TODO(jder): nicer typing
        data_in_boundary_src: Any = self._boundary_src.map_data(
            lambda ds: ds.isel(time=x_index).isel(time=slice(None, self.hist + 1))
        )
        if self.normalize_before_mask:
            data_in_boundary = data_in_boundary_src.normalize()
        else:
            data_in_boundary = data_in_boundary_src.data
        data_in_boundary = (
            data_in_boundary.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in_boundary = torch.from_numpy(data_in_boundary).float()
        data_in_boundary = torch.where(
            self.wet_surface, data_in_boundary, self.masked_fill_value
        )
        if not self.normalize_before_mask:
            data_in_boundary = self._boundary_src.normalize_with(
                data_in_boundary, variable_axis=2
            )
        data_in_boundary = rearrange(
            data_in_boundary,
            "window_dim time variable lat lon -> \
                window_dim (time variable) lat lon",
        )
        return data_in_boundary

    def _get_label(self, x_index) -> Prognostic:
        # TODO(jder): nicer typing
        label_src: Any = self._prognostic_src.map_data(
            lambda ds: ds.isel(time=x_index).isel(time=slice(self.hist + 1, None))
        )
        if self.normalize_before_mask:
            label = label_src.normalize()
        else:
            label = label_src.data
        label = (
            label.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        label = torch.from_numpy(label).float()
        label = torch.where(self.wet, label, self.masked_fill_value)
        if not self.normalize_before_mask:
            label = label_src.normalize_with(label, variable_axis=2)
        label = rearrange(
            label,
            "window_dim time variable lat lon ->\
                window_dim (time variable) lat lon",
        ).squeeze()
        return label


class TorchTrainDataset(Dataset):
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

    FLAG = LoaderVersion.OM4_TORCH

    @elapsed
    def __init__(
        self,
        src: DataSource,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        wet: PrognosticMask,
        wet_surface: GridMask,
        hist: int,
        steps: int,
        normalize_before_mask: bool,
        masked_fill_value: float,
        stride: int = 1,
        executor: ThreadPoolExecutor | None = None,
    ):
        super().__init__()
        self.device = get_device()

        self.hist: int = hist
        self.steps: int = steps
        self.stride: int = stride
        self.normalize_before_mask: bool = normalize_before_mask
        self.masked_fill_value: float = masked_fill_value
        self.executor = executor

        self.num_prognostic_channels: int = (hist + 1) * len(prognostic_var_names)
        data = src.data
        self._prognostic_src = src.filter(prognostic_var_names, prefix="prognostic")
        self._boundary_src = src.filter(boundary_var_names, prefix="boundary")

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

        self.wet = wet.bool()
        self.wet_surface = wet_surface.bool()

        self.size: int = (
            data.time.size
            - self.steps * (self.hist + 1) * self.stride
            - self.hist * self.stride
        )

        if using_gpu():
            self.wet = self.wet.pin_memory()
            self.wet_surface = self.wet_surface.pin_memory()

    def __len__(self) -> int:
        return self.size

    def perform_io(self, x_index, *, executor=None) -> tuple[xr.Dataset, xr.Dataset]:
        prognostic_selected = self._prognostic_src.data.isel(time=x_index)
        boundary_selected = self._boundary_src.data.isel(time=x_index)

        if executor is not None:
            concurrent_compute(
                prognostic_selected, boundary_selected, executor=self._executor
            )

        return prognostic_selected, boundary_selected

    def prognostic_to_tensor(self, prognostic_selected):
        if "lev" in prognostic_selected.dims:
            prognostic_all = torch.from_numpy(
                conditional_rearrange(
                    prognostic_selected,
                    "time (variable lev)=var lat lon",
                    concat_dim="var",
                )
                .rename({"var": "variable"})
                .to_numpy()
            )
        else:
            prognostic_all = torch.from_numpy(
                prognostic_selected.to_array()
                .transpose("time", "variable", "lat", "lon")
                .to_numpy()
            )

        return prognostic_all

    def boundary_to_tensor(self, boundary_selected):
        boundary = torch.from_numpy(
            boundary_selected.to_array()
            .transpose("time", "variable", "lat", "lon")
            .to_numpy()
        )
        return boundary

    @elapsed(level=logging.DEBUG)
    def __getitem__(self, idx: int):
        start_time = time.perf_counter()
        TD = TrainData(self.num_prognostic_channels)

        for step in range(self.steps):
            x_index = self.index_by_step(idx, step)
            prognostic_selected, boundary_selected = self.perform_io(x_index)

            prognostic = self.prognostic_to_tensor(prognostic_selected)
            boundary = self.boundary_to_tensor(boundary_selected)

            input_, label = self.join_and_partition_by_time(prognostic, boundary)
            TD.insert(input_=input_, label=label)

        TD.load_stats = LoadStats(time.perf_counter() - start_time)
        return TD

    def join_and_partition_by_time(
        self,
        # time includes (self.hist + 1) past steps and the (label) future steps
        prognostic_all: Float[torch.Tensor, "time variable lat lon"],
        boundary_all: Float[torch.Tensor, "time variable lat lon"],
    ) -> tuple[Input, Prognostic]:
        # grab past steps and prep for model
        total_input = self._prep_tensor_steps(
            prognostic_all[: self.hist + 1, :, :, :],
            boundary_all[: self.hist + 1, :, :, :],
        )
        # grab future steps, repeat as we do for input
        label = self._prep_tensor_steps(prognostic_all[self.hist + 1 :, :, :, :])
        return total_input, label

    def _prep_tensor_steps(
        self,
        prognostic_steps: Float[torch.Tensor, "time variable lat lon"],
        boundary_steps: Float[torch.Tensor, "time variable lat lon"] | None = None,
    ) -> Input:
        """Prepare tensor steps by normalizing, masking and flattening dimensions."""

        # Normalize and mask tensors
        def normalize_and_mask(
            tensor: torch.Tensor,
            src: DataSource,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            if self.normalize_before_mask:
                tensor = src.normalize_with(tensor, variable_axis=1).float()
            tensor = torch.where(mask, tensor, self.masked_fill_value)
            if not self.normalize_before_mask:
                tensor = src.normalize_with(tensor, variable_axis=1).float()
            return tensor

        prognostic_steps = normalize_and_mask(
            prognostic_steps, self._prognostic_src, self.wet
        )
        if boundary_steps is not None:
            boundary_steps = normalize_and_mask(
                boundary_steps, self._boundary_src, self.wet_surface
            )

        # Flatten time and variable dimensions
        def flatten_dims(tensor: torch.Tensor) -> torch.Tensor:
            return rearrange(tensor, "time variable lat lon -> (time variable) lat lon")

        prognostic_steps = flatten_dims(prognostic_steps)
        if boundary_steps is not None:
            boundary_steps = flatten_dims(boundary_steps)
            return torch.cat((prognostic_steps, boundary_steps), dim=0)

        return prognostic_steps

    def index_by_step(self, idx: int, step: int) -> xr.DataArray:
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


T = TypeVar("T")
WithSteps = tuple[int, T]


class SpdlTorchDataLoader(DataLoader):
    def __init__(
        self,
        dataset: ConcatDataset,
        collate_fn: Callable,
        io_workers: int,
        cpu_workers: int,
        batch_size: int,
        drop_last: bool = False,
        sampler: type[Sampler] | partial[DistributedSampler] | None = None,
        pin_memory: bool = False,
        timeout: float = 0,
    ) -> None:
        super().__init__(
            dataset,
            batch_size,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        first_dataset = dataset.datasets[0]
        assert isinstance(first_dataset, TorchTrainDataset), (
            "This DataLoader only works with TorchTrainDataset."
        )
        self.first_dataset: TorchTrainDataset = first_dataset
        self.steps = self.first_dataset.steps
        self.executor = self.first_dataset.executor
        self._sampler_klass = sampler
        self.io_workers = io_workers
        self.cpu_workers = cpu_workers
        self.timeout = timeout
        if pin_memory:
            warnings.warn(f"Pinning memory is currently not supported in {self!r}.")

    def _source(self) -> Iterable[int]:
        """Generated indexes to load, using the input sampler."""
        items = list(range(len(self)))
        samples: Iterable = items
        if self._sampler_klass is not None:
            samples = self._sampler_klass(items)
        if self.drop_last:
            samples = itertools.islice(samples, len(items) - 1)
        yield from samples

    def _with_steps(self, index: int) -> Iterable[WithSteps[xr.DataArray]]:
        yield from (
            (step, self.first_dataset.index_by_step(index, step))
            for step in range(self.steps)
        )

    def _download(
        self, x_index_pair: WithSteps[xr.DataArray]
    ) -> WithSteps[tuple[xr.Dataset, xr.Dataset]]:
        step, x_index = x_index_pair
        return step, self.first_dataset.perform_io(x_index, executor=self.executor)

    def _process_traindata(
        self, downloaded: list[WithSteps[tuple[xr.Dataset, xr.Dataset]]]
    ) -> TrainData:
        train_data = TrainData(self.first_dataset.num_prognostic_channels)

        # Ensure that the pipeline-level aggregation respects the steps of train data!
        steps = [step for step, _ in downloaded]
        assert steps == list(range(self.steps))

        for _, (prognostic_selected, boundary_selected) in downloaded:
            prognostic = self.first_dataset.prognostic_to_tensor(prognostic_selected)
            boundary = self.first_dataset.boundary_to_tensor(boundary_selected)
            input_, label = self.first_dataset.join_and_partition_by_time(
                prognostic, boundary
            )
            train_data.insert(input_, label)

        return train_data

    def _build_pipeline(self) -> spdl.pipeline.Pipeline[TrainData]:
        total_workers = self.io_workers + self.cpu_workers + 2
        return (
            PipelineBuilder()
            .add_source(self._source())  # type: ignore
            .pipe(self._with_steps, output_order="input")  # type: ignore
            .pipe(
                self._download,  # type: ignore
                concurrency=self.io_workers,
                executor=self.executor,
                output_order="input",
            )
            .aggregate(self.steps)
            .pipe(self._process_traindata, concurrency=self.cpu_workers)
            .aggregate(self.batch_size or 1)
            .pipe(self.collate_fn, concurrency=self.cpu_workers)
            # Later, we can add a step to efficiently dispatch data to GPU.
            .add_sink(6)
            .build(num_threads=total_workers)
        )

    # This doesn't return the same type as the base data loader type, but I don't think
    # this should be a problem, so we ignore type checking.
    def __iter__(self) -> PipelineIterator[TrainData]:  # type: ignore
        pipeline = self._build_pipeline()

        # TIL that the __exit__ condition will still be called even though we use
        # a `return` instead of a `yield`.
        with pipeline.auto_stop():
            # Need to adapt the `DataLoader.timeout` type with the
            # `spdl.pipeline.Pipeline.timeout` type. The former uses `0` as "no timeout"
            # whereas the latter uses `None`.
            timeout = None if self.timeout == 0 else self.timeout
            iterator = pipeline.get_iterator(timeout=timeout)
            assert isinstance(iterator, PipelineIterator)
            return iterator
