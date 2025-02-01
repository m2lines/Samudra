import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.ndimage import gaussian_filter
from einops import rearrange
import os
import cftime
from datetime import timedelta



def get_time_slice(time_config, initial_cond=False, time_delta=5, hist=1):
    start_time_str = time_config.start_time
    start_year, start_month, start_day = start_time_str.split("-")
    start_time = cftime.DatetimeNoLeap(int(start_year), int(start_month), int(start_day), 0, 0, 0)
    
    end_time_str = time_config.end_time
    end_year, end_month, end_day = end_time_str.split("-")
    end_time = cftime.DatetimeNoLeap(int(end_year), int(end_month), int(end_day), 0, 0, 0)
    num_steps = (end_time - start_time).days // time_delta + 1

    if initial_cond:
        start_time = start_time - timedelta(days=time_delta * (hist + 1)) # Prepending initial condition

    return slice(start_time, end_time), num_steps


def get_wet_mask(inputs, device="cpu"):
    wet = xr.zeros_like(inputs[0][0])
    # inputs[0][0,12,12] = np.nan
    for data in inputs:
        wet += np.isnan(data[0])

    wet_nan = xr.where(wet != 0, np.nan, 1).to_numpy()
    wet = np.isnan(xr.where(wet == 0, np.nan, 0))
    wet = np.nan_to_num(wet.to_numpy())
    wet = torch.from_numpy(wet).type(torch.float32).to(device=device)
    return wet, wet_nan


class data_CNN_Disk(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        inputs_str,
        extra_in_str,
        outputs_str,
        wet,
        data_mean,
        data_std,
        hist,
        long_rollout,
        device="cuda",
    ):
        super().__init__()
        self.device = device

        self.size = data.time.size
        self.hist = hist

        self.inputs = data[inputs_str + extra_in_str]
        self.outputs = data[outputs_str]
        self.inputs_no_extra = data[inputs_str]
        self.extras = data[extra_in_str]

        # This class will be used only for validation and rollouts
        # Rolling indices to keep track of histories/past states:
        # HIST=0 ; 0->[0, 1]; 1->[1, 2]; 2->[2, 3]; 3->[3, 4]
        # HIST=1 ; 0->[[0, 1], [2, 3]]; 1->[[2, 3], [4, 5]]; 2->[[4, 5], [6, 7]]; 3->[[6, 7], [8, 9]]
        # HIST=2 ; 0->[[0, 1, 2], [3, 4, 5]]; 1->[[3, 4, 5], [6, 7, 8]]; 2->[[6, 7, 8], [9, 10, 11]]; 3->[[9, 10, 11], [12, 13, 14]]
        time_indices = np.arange(data.time.size)
        indices = xr.DataArray(
            time_indices,
            dims=["time"],
            coords={"time": time_indices},
        )
        total_steps = 2 * self.hist + 1
        rolling_indices = (
            indices.rolling(time=len(time_indices) - total_steps, center=False)
            .construct("window_dim")
        )
        rolling_indices = rolling_indices.transpose("window_dim", "time").isel(
            time=slice(len(time_indices) - total_steps - 1, None)
        )  # Remove first few null indices
        self.rolling_indices = rolling_indices.isel(
            window_dim=slice(0, None, self.hist + 1)
        )  # Skip indices based on history
        self.rolling_indices = self.rolling_indices.astype(int)

        if long_rollout:
            window0 = self.rolling_indices.isel(window_dim=0)
            print(
                "Long rollout will use input at time {0} and produce output at {1}".format(
                    data.time.values[0],
                    data.time.values[self.hist + 1],
                )
            )

        self.in_mean = data_mean[inputs_str + extra_in_str]
        self.in_std = data_std[inputs_str + extra_in_str]
        self.out_mean = data_mean[outputs_str]
        self.out_std = data_std[outputs_str]
        self.inputs_no_extra_mean = data_mean[inputs_str]
        self.inputs_no_extra_std = data_std[inputs_str]
        self.extras_mean = data_mean[extra_in_str]
        self.extras_std = data_std[extra_in_str]

        self.wet = wet

    def set_device(self, device):
        self.device = device

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if type(idx) == slice:
            if idx.start == None and idx.stop == None:
                idx = slice(0, self.size, idx.step)
            elif idx.start == None:
                idx = slice(0, idx.stop, idx.step)
            elif idx.stop == None:
                idx = slice(idx.start, self.size, idx.step)
        elif type(idx) == int:
            idx = slice(idx, idx + 1, 1)

        rolling_idx = self.rolling_indices.isel(window_dim=idx)
        x_index = xr.Variable(["window_dim", "time"], rolling_idx)
        data_in = self.inputs_no_extra.isel(time=x_index).isel(
            time=slice(None, self.hist + 1)
        )
        data_in = (
            (data_in - self.inputs_no_extra_mean) / self.inputs_no_extra_std
        ).fillna(0)
        data_in = (
            data_in.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in = rearrange(
            data_in, "window_dim time variable lat lon -> window_dim (time variable) lat lon"
        )
        data_in_boundary = self.extras.isel(time=x_index).isel(time=self.hist)
        data_in_boundary = (
            (data_in_boundary - self.extras_mean) / self.extras_std
        ).fillna(0)
        data_in_boundary = (
            data_in_boundary.to_array()
            .transpose("window_dim", "variable", "lat", "lon")
            .to_numpy()
        )
        data_in = np.concatenate((data_in, data_in_boundary), axis=1)

        label = self.outputs.isel(time=x_index).isel(time=slice(self.hist + 1, None))
        label = ((label - self.out_mean) / self.out_std).fillna(0)
        label = (
            label.to_array()
            .transpose("window_dim", "time", "variable", "lat", "lon")
            .to_numpy()
        )
        label = rearrange(
            label, "window_dim time variable lat lon -> window_dim (time variable) lat lon"
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
        data_mean,
        data_std,
        hist,
        steps,
        stride=1,
        device="cuda",
    ):
        super().__init__()
        self.device = device

        self.size = data.time.size
        self.hist = hist
        self.steps = steps
        self.stride = stride

        self.inputs = data[inputs_str + extra_in_str]
        self.outputs = data[outputs_str]
        self.inputs_no_extra = data[inputs_str]
        self.extras = data[extra_in_str]

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

        self.inputs_no_extra_mean = data_mean[inputs_str]
        self.inputs_no_extra_std = data_std[inputs_str]
        self.extras_mean = data_mean[extra_in_str]
        self.extras_std = data_std[extra_in_str]
        self.in_mean = data_mean[inputs_str + extra_in_str]
        self.in_std = data_std[inputs_str + extra_in_str]

        self.out_mean = data_mean[outputs_str]
        self.out_std = data_std[outputs_str]

        self.wet = wet

    def set_device(self, device):
        self.device = device

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

        assert type(idx) == int
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
            data_in = (
                (data_in - self.inputs_no_extra_mean) / self.inputs_no_extra_std
            ).fillna(0)
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
            data_in_boundary = (
                (data_in_boundary - self.extras_mean) / self.extras_std
            ).fillna(0)
            data_in_boundary = (
                data_in_boundary.to_array()
                .transpose("window_dim", "variable", "lat", "lon")
                .to_numpy()
            )
            data_in = np.concatenate((data_in, data_in_boundary), axis=1).squeeze()

            label = self.outputs.isel(time=x_index).isel(
                time=slice(self.hist + 1, None)
            )
            label = ((label - self.out_mean) / self.out_std).fillna(0)
            label = (
                label.to_array()
                .transpose("window_dim", "time", "variable", "lat", "lon")
                .to_numpy()
            )
            label = rearrange(
                label, "window_dim time variable lat lon -> window_dim (time variable) lat lon"
            ).squeeze()

            outputs.append(torch.from_numpy(data_in).float())
            outputs.append(torch.from_numpy(label).float())

        return outputs
