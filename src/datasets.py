import numpy as np
import torch
import xarray as xr
from einops import rearrange

from utils.data import Normalize
from utils.device import get_device


class data_CNN_Disk(torch.utils.data.Dataset):
    """This class is used for validation and rollouts.

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
        inputs_str,
        extra_in_str,
        outputs_str,
        wet,
        hist,
        long_rollout,
    ):
        super().__init__()
        self.device = get_device()

        self.size = data.time.size
        self.hist = hist

        self.inputs = data[inputs_str + extra_in_str]
        self.outputs = data[outputs_str]
        self.inputs_no_extra = data[inputs_str]
        self.extras = data[extra_in_str]

        self.normalize = Normalize.get_instance()

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
            print(
                "Long rollout will use input at time {0} and produce"
                " output at {1}".format(
                    data.time.values[0],
                    data.time.values[self.hist + 1],
                )
            )

        self.wet = wet

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if idx.start is None and idx.stop is None:
                idx = slice(0, self.size, idx.step)
            elif idx.start is None:
                idx = slice(0, idx.stop, idx.step)
            elif idx.stop is None:
                idx = slice(idx.start, self.size, idx.step)
        elif isinstance(idx, int):
            idx = slice(idx, idx + 1, 1)

        rolling_idx = self.rolling_indices.isel(window_dim=idx)
        x_index = xr.Variable(["window_dim", "time"], rolling_idx)
        data_in = self.inputs_no_extra.isel(time=x_index).isel(
            time=slice(None, self.hist + 1)
        )
        data_in = self.normalize.normalize_inputs(data_in)
        data_in = (
            data_in.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in = rearrange(
            data_in,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )
        data_in_boundary = self.extras.isel(time=x_index).isel(time=self.hist)
        data_in_boundary = self.normalize.normalize_boundary(data_in_boundary)
        data_in_boundary = (
            data_in_boundary.to_array()
            .transpose("window_dim", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in = np.concatenate((data_in, data_in_boundary), axis=1)

        label = self.outputs.isel(time=x_index).isel(time=slice(self.hist + 1, None))
        label = self.normalize.normalize_outputs(label)
        label = (
            label.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        label = rearrange(
            label,
            "window_dim time variable lat lon -> window_dim (time variable) lat lon",
        )

        items = (torch.from_numpy(data_in).float(), torch.from_numpy(label).float())

        return items


class data_CNN_Disk_steps(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        inputs_str,
        extra_in_str,
        outputs_str,
        wet,
        hist,
        steps,
        stride=1,
    ):
        super().__init__()
        self.device = get_device()

        self.size = data.time.size
        self.hist = hist
        self.steps = steps
        self.stride = stride

        self.inputs = data[inputs_str + extra_in_str]
        self.outputs = data[outputs_str]
        self.inputs_no_extra = data[inputs_str]
        self.extras = data[extra_in_str]

        self.normalize = Normalize.get_instance()

        # This class will be used only for training
        total_steps = 2 * self.hist + 2

        # Calculate the number of windows
        num_windows = data.time.size - (total_steps - 1) * self.stride

        # Create base indices
        indices = np.arange(num_windows)
        indices_da = xr.DataArray(indices, dims=["window_dim"])

        # Create window dimension
        window_dim = xr.DataArray(np.arange(total_steps), dims=["time"])

        # Construct rolling indices
        self.rolling_indices = indices_da + stride * window_dim

        self.wet = wet

    def __len__(self):
        return (
            self.size
            - self.steps * (self.hist + 1) * self.stride
            - self.hist * self.stride
        )

    def __getitem__(self, idx):
        outputs = []

        if idx >= len(self):
            raise IndexError("Index out of range")

        assert isinstance(idx, int)
        prev_rolling_idx = None
        for step in range(self.steps):
            start = idx + step * (self.hist + 1) * self.stride
            end = start + 1
            idx_slice = slice(
                start, end
            )  # Create a slice for similar indexing as in data_CNN_Disk
            rolling_idx = self.rolling_indices.isel(window_dim=idx_slice)
            if prev_rolling_idx is not None:
                assert (
                    prev_rolling_idx.isel(time=slice(self.hist + 1, None))
                    - rolling_idx.isel(time=slice(0, self.hist + 1))
                ).sum() == 0  # Prev output = Cur Input
                assert (
                    rolling_idx.diff("time") == self.stride
                ).all()  # Stride is maintained
                assert rolling_idx.isel(time=-1) < self.size  # Last index check
            x_index = xr.Variable(["window_dim", "time"], rolling_idx)
            data_in = self.inputs_no_extra.isel(time=x_index).isel(
                time=slice(None, self.hist + 1)
            )
            data_in = self.normalize.normalize_inputs(data_in)
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
            data_in_boundary = self.extras.isel(time=x_index).isel(time=self.hist)
            data_in_boundary = self.normalize.normalize_boundary(data_in_boundary)
            data_in_boundary = (
                data_in_boundary.to_array()
                .transpose("window_dim", "variable", "lat", "lon")
                .to_numpy()
            )
            data_in = np.concatenate((data_in, data_in_boundary), axis=1).squeeze()

            label = self.outputs.isel(time=x_index).isel(
                time=slice(self.hist + 1, None)
            )
            label = self.normalize.normalize_outputs(label)
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

            outputs.append(torch.from_numpy(data_in).float())
            outputs.append(torch.from_numpy(label).float())

        return outputs
