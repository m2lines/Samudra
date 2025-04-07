import logging
from typing import Any

import numpy as np
import torch
import xarray as xr
from einops import rearrange
from jaxtyping import Float, Integer
from torch.utils.data import Dataset
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
from ocean_emulators.utils.data import (
    Normalize,
    filter_compact_prognostic,
    is_compact,
    mask,
    unflatten_masks,
)
from ocean_emulators.utils.device import get_device, using_gpu


class OM4Dataset(Dataset):
    """A `torch.Dataset` for Zarr-backed OM4 data."""

    FLAG = LoaderVersion.OM4_LAZY

    def __init__(
        self,
        data: xr.Dataset,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        hist: int,
        steps: int,
        stride: int = 1,
        is_inference: bool = False,
    ) -> None:
        self.hist: int = hist
        self.steps: int = steps
        self.stride: int = stride
        self.is_inference: bool = is_inference
        self.is_compact: bool = is_compact(data)

        self._size: int = (
            data.time.size
            - self.steps * (self.hist + 1) * self.stride
            - self.hist * self.stride
        )

        # TODO(alxmrs): When we want to support inference later on, we will need to
        #  calculate different steps.
        if self.is_inference:
            raise NotImplementedError("does not (yet) support inference.")

        self.prognostic, self.boundary = self.prepare_data(
            data, prognostic_var_names, boundary_var_names
        )

        total_steps: int = 2 * self.hist + 2
        # Calculate the number of windows
        num_windows = data.time.size - (total_steps - 1) * self.stride
        # Create base indices
        indices = np.arange(num_windows)
        indices_da = xr.DataArray(indices, dims=["step"])
        # Create window dimension
        window_dim = xr.DataArray(np.arange(total_steps), dims=["time"])
        # Construct rolling indices
        self.rolling_indices: Integer[xr.DataArray, "step time"] = (
            indices_da + stride * window_dim
        )

    # TODO(alxmrs): Put back into init!
    def prepare_data(
        self, data, prognostic_var_names, boundary_var_names
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Initialize loader datasets."""
        # Ensure that a `wetmask` DataArray exists along a `lev` dimension.
        data_ = unflatten_masks(data)
        if self.is_compact:
            prognostic = filter_compact_prognostic(data_, prognostic_var_names)
        else:
            prognostic = data_[prognostic_var_names]
        boundary = data_[boundary_var_names]

        # Normalize data. E.g. mean=zero, std=1., NaN --> 0.0
        norm = Normalize.get_instance()
        norm_prognostic = norm.normalize_prognostic(prognostic)
        norm_boundary = norm.normalize_boundary(boundary)

        # Set non-ocean areas to zero.
        prognostic = mask(norm_prognostic, data_.wetmask)
        boundary = mask(norm_boundary, data_.wetmask)

        return prognostic, boundary

    def __len__(self) -> int:
        return self._size

    def window_from(
        self, idx: int | slice, step: int
    ) -> Integer[xr.DataArray, "step time"]:
        """Coalesce index values to a window of the input data."""
        # First, parse int inputs as a slice.
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self):
                raise IndexError(
                    f"index {idx!r} out of bounds. Must be between 0 and {len(self)}."
                )

            if not self.is_inference:
                idx = idx + step * (self.hist + 1) * self.stride
            idx = slice(idx, idx + 1, 1)

        # Validate and normalize all slices.
        if self.is_inference:
            if idx.start < 0 or idx.stop < 0 or idx.step < 0:
                raise IndexError(
                    f"index {idx!r} invalid: negative values not supported."
                )
            if idx.start > self._size or idx.stop > self._size:
                raise IndexError(
                    f"index {idx!r} out of bounds. "
                    f"All bounds must be less than or equal to {len(self)}."
                )

        if idx.start is None:
            idx = slice(0, idx.stop, idx.step)
        if idx.stop is None:
            idx = slice(idx.start, len(self), idx.step)

        return self.rolling_indices.isel(step=idx)

    def __getitem__(self, idx: int) -> Example:
        windows = [self.window_from(idx, step) for step in range(self.steps)]
        window = xr.concat(windows, dim="step")

        # This point in time splits the training data and the label data!
        time_split = self.hist + 1

        # TODO(alxmrs): Tune dask parallelization
        # https://tutorial.xarray.dev/advanced/apply_ufunc/dask_apply_ufunc.html

        if self.is_compact:
            with_lev = [v for v in self.prognostic if "lev" in self.prognostic[v].dims]
            without_lev = [
                v for v in self.prognostic if "lev" not in self.prognostic[v].dims
            ]

            label_without_lev = (
                self.prognostic.isel(time=window)
                .isel(time=slice(time_split, None))[without_lev]
                .to_array()
                .einops.rearrange("step (time variable)=var lat lon", dask="allowed")
                .rename({"var": "variable"})
            )
            label_with_lev = (
                self.prognostic.isel(time=window)
                .isel(time=slice(time_split, None))[with_lev]
                .to_array()
                .einops.rearrange(
                    "step (time variable lev)=var lat lon", dask="allowed"
                )
                .rename({"var": "variable"})
            )
            label = xr.concat([label_without_lev, label_with_lev], dim="variable")
            prognostic_withot_lev = (
                self.prognostic.isel(time=window)
                .isel(time=slice(None, time_split))[without_lev]
                .to_array(name="prognostic")
                .einops.rearrange("step (time variable)=var lat lon", dask="allowed")
                .drop_vars("var", errors="ignore")
            )
            prognostic_with_lev = (
                self.prognostic.isel(time=window)
                .isel(time=slice(None, time_split))[with_lev]
                .to_array(name="prognostic")
                .einops.rearrange(
                    "step (time variable lev)=var lat lon", dask="allowed"
                )
                .drop_vars("var", errors="ignore")
            )
            prognostic = xr.concat(
                [prognostic_withot_lev, prognostic_with_lev], dim="var"
            )

        else:
            label = (
                self.prognostic.isel(time=window)
                .isel(time=slice(time_split, None))
                .to_array()
                .einops.rearrange("step (time variable)=var lat lon", dask="allowed")
                .rename({"var": "variable"})
            )

            prognostic = (
                self.prognostic.isel(time=window)
                .isel(time=slice(None, time_split))
                .to_array(name="prognostic")
                .einops.rearrange("step (time variable)=var lat lon", dask="allowed")
                .drop_vars("var", errors="ignore")
            )
        boundary = (
            self.boundary.isel(time=window)
            .isel(time=self.hist)
            .to_array("var", "boundary")
            .einops.rearrange("step var lat lon", dask="allowed")
            .drop_vars("var", errors="ignore")
        )
        # Combine prognostic and boundary data
        input_ = xr.concat([prognostic, boundary], dim="var").rename(
            {"var": "variable"}
        )

        return input_, label


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

    def __init__(
        self,
        data,
        prognostic_var_names,
        boundary_var_names,
        wet,
        wet_surface,
        hist,
        long_rollout,
        is_compact=False,
    ):
        super().__init__()
        self.device = get_device()

        self.hist = hist

        self.num_prognostic_channels = (hist + 1) * len(prognostic_var_names)

        if is_compact:
            self._prognostic_vars = filter_compact_prognostic(
                data, prognostic_var_names
            )
        else:
            self._prognostic_vars = data[prognostic_var_names]
        self._boundary_vars = data[boundary_var_names]

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
            logging.info(
                "Long rollout will use input at time {0} and produce"
                " output at {1}".format(
                    data.time.values[0],
                    data.time.values[self.hist + 1],
                )
            )

        self.wet = wet.bool()
        self.wet_surface = wet_surface.bool()
        self.size = len(self.rolling_indices)

        if using_gpu():
            self.wet = self.wet.pin_memory()
            self.wet_surface = self.wet_surface.pin_memory()

    def __len__(self):
        return self.size

    @property
    def initial_prognostic(self):
        data = self.__getitem__(0)[0]
        return data[:, : self.num_prognostic_channels]

    def inference_target(self, step: int):
        return self.__getitem__(step)[1]

    def get_initial_input(self):
        data = self.__getitem__(0)[0]
        return data

    # TODO: This is a placeholder for now since time returned is incorrect
    def get_input_time(self, step: int):
        return self._prognostic_vars.time[step]

    def merge_prognostic_and_boundary(self, prognostic: torch.Tensor, step: int):
        x_index = self._get_x_index(step)
        boundary = self._get_boundary(x_index).to(prognostic.device)
        data = torch.cat((prognostic, boundary), dim=1)
        return data

    def __getitem__(self, idx):
        x_index = self._get_x_index(idx)
        data_in = self._get_prognostic(x_index)
        data_in_boundary = self._get_boundary(x_index)
        data_in = torch.cat((data_in, data_in_boundary), dim=1)
        label = self._get_label(x_index)
        return (data_in, label)

    def _get_x_index(self, idx):
        if isinstance(idx, slice):
            if idx.start < 0 or idx.stop < 0 or idx.step < 0:
                raise IndexError("Sorry, negative indexing is not supported!")
            elif idx.start >= self.size or idx.stop >= self.size:
                raise IndexError(f"Index {idx} out of range with size {self.size}")
            elif idx.start is None and idx.stop is None:
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
        data_in = self._prognostic_vars.isel(time=x_index).isel(
            time=slice(None, self.hist + 1)
        )
        data_in = Normalize.get_instance().normalize_prognostic(
            data_in
        )  # TODO: Weird error when I get_instance in init
        data_in = (
            data_in.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in = rearrange(
            data_in,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )
        data_in = torch.from_numpy(data_in).float()
        data_in = torch.where(self.wet, data_in, 0.0)
        return data_in

    def _get_boundary(self, x_index):
        """
        This function returns the boundary condition for the current time step.

        With hist > 0, the boundary condition considered is always the last step of
        the input.
        """
        data_in_boundary = self._boundary_vars.isel(time=x_index).isel(time=self.hist)
        data_in_boundary = Normalize.get_instance().normalize_boundary(data_in_boundary)
        data_in_boundary = (
            data_in_boundary.to_array()
            .transpose("window_dim", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in_boundary = torch.from_numpy(data_in_boundary).float()
        data_in_boundary = torch.where(self.wet_surface, data_in_boundary, 0.0)
        return data_in_boundary

    def _get_label(self, x_index):
        label = self._prognostic_vars.isel(time=x_index).isel(
            time=slice(self.hist + 1, None)
        )
        label = Normalize.get_instance().normalize_prognostic(label)
        label = (
            label.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        label = rearrange(
            label,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )
        label = torch.from_numpy(label).float()
        # label = label * self.wet
        label = torch.where(self.wet, label, 0.0)
        return label

    def get_coords_dict(self):
        return {co: self._prognostic_vars[co] for co in self._prognostic_vars.coords}


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

    def __init__(
        self,
        data: xr.Dataset,
        prognostic_var_names: PrognosticVarNames,
        boundary_var_names: BoundaryVarNames,
        wet: PrognosticMask,
        wet_surface: GridMask,
        hist: int,
        steps: int,
        stride: int = 1,
    ):
        super().__init__()
        self.device = get_device()

        self.hist: int = hist
        self.steps: int = steps
        self.stride: int = stride

        self.num_prognostic_channels: int = (hist + 1) * len(prognostic_var_names)
        self._prognostic_vars: xr.Dataset = data[prognostic_var_names]
        self._boundary_vars: xr.Dataset = data[boundary_var_names]

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

        # Normalize
        logging.info("Normalizing inputs")
        self.normalize = Normalize.get_instance()
        self._prognostic_vars = self.normalize.normalize_prognostic(
            self._prognostic_vars
        )
        self._boundary_vars = self.normalize.normalize_boundary(self._boundary_vars)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
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
        # Convert to tests, tests are outdated since changing time definition
        # if prev_rolling_idx is not None:
        #     assert (
        #         prev_rolling_idx.isel(time=slice(self.hist + 1, None))
        #         - rolling_idx.isel(time=slice(0, self.hist + 1))
        #     ).sum() == 0  # Prev output = Cur Input
        #     assert (
        #         rolling_idx.diff("time") == self.stride
        #     ).all()  # Stride is maintained
        #     assert rolling_idx.isel(time=-1) < self.size  # Last index check

        x_index = xr.Variable(["window_dim", "time"], rolling_idx)
        return x_index

    def _get_input(self, x_index) -> Prognostic:
        # TODO(jder): nicer typing
        data_in: Any = self._prognostic_vars.isel(time=x_index).isel(
            time=slice(None, self.hist + 1)
        )
        data_in = (
            data_in.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in = rearrange(
            data_in,
            "window_dim time variable lat lon -> \
                window_dim (time variable) lat lon",
        )
        data_in = torch.from_numpy(data_in).float()
        data_in = torch.where(self.wet, data_in, 0.0)
        return data_in

    def _get_boundary(self, x_index) -> Boundary:
        """
        This function returns the boundary condition for the current time step.

        With hist > 0, the boundary condition considered is always the last step of
        the input.
        """
        # TODO(jder): nicer typing
        data_in_boundary: Any = self._boundary_vars.isel(time=x_index).isel(
            time=self.hist
        )
        data_in_boundary = (
            data_in_boundary.to_array()
            .transpose("window_dim", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in_boundary = torch.from_numpy(data_in_boundary).float()
        data_in_boundary = torch.where(self.wet_surface, data_in_boundary, 0.0)
        return data_in_boundary

    def _get_label(self, x_index) -> Prognostic:
        # TODO(jder): nicer typing
        label: Any = self._prognostic_vars.isel(time=x_index).isel(
            time=slice(self.hist + 1, None)
        )
        label = (
            label.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        label = rearrange(
            label,
            "window_dim time variable lat lon ->\
                window_dim (time variable) lat lon",
        ).squeeze()
        label = torch.from_numpy(label).float()
        label = torch.where(self.wet, label, 0.0)
        return label
