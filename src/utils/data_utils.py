import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.ndimage import gaussian_filter
from einops import rearrange
import os

# Class defined to store information about the grid and corresponding graph of data. Importantly produces the adjacency matrices
# and keeps track of what is land vs ocean


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
        n_samples,
        lag,
        interval,
        ind_start,
        device="cuda",
    ):
        super().__init__()
        self.device = device

        self.size = n_samples
        self.lag = lag
        self.interval = interval
        self.ind_start = ind_start

        self.inputs = data[inputs_str + extra_in_str]
        self.outputs = data[outputs_str]

        self.in_mean = data_mean[inputs_str + extra_in_str]
        self.in_std = data_std[inputs_str + extra_in_str]

        self.out_mean = data_mean[outputs_str]
        self.out_std = data_std[outputs_str]

        self.wet = wet

    def set_device(self, device):
        self.device = device

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        if type(idx) == list:
            ind_in = [self.ind_start + i * self.interval for i in idx]
            ind_out = [self.ind_start + i * self.interval + self.lag for i in idx]
        elif type(idx) == slice:
            if idx.start == None and idx.stop == None:
                idx = slice(0, self.size, idx.step)
            elif idx.start == None:
                idx = slice(0, idx.stop, idx.step)
            elif idx.stop == None:
                idx = slice(idx.start, self.size, idx.step)

            ind_in = slice(
                self.ind_start + idx.start, idx.stop * self.interval, self.interval
            )
            ind_out = slice(
                self.ind_start + idx.start + self.lag,
                self.ind_start + idx.stop * self.interval + self.lag,
                self.interval,
            )
        if type(idx) == int:
            ind_in = self.ind_start + idx * self.interval
            ind_out = self.ind_start + idx * self.interval + self.lag

        data_in = self.inputs.isel(time=ind_in)
        data_in = ((data_in - self.in_mean) / self.in_std).fillna(0)
        label = self.outputs.isel(time=ind_out)
        label = ((label - self.out_mean) / self.out_std).fillna(0)

        if type(idx) == int:
            data_in = data_in.to_array().transpose("variable", "y", "x").to_numpy()
            label = label.to_array().transpose("variable", "y", "x").to_numpy()
        else:
            data_in = (
                data_in.to_array().transpose("time", "variable", "y", "x").to_numpy()
            )
            label = label.to_array().transpose("time", "variable", "y", "x").to_numpy()

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
        n_samples,
        lag,
        interval,
        steps,
        device="cuda",
    ):
        super().__init__()
        self.device = device

        self.size = n_samples
        self.lag = lag
        self.interval = interval
        self.steps = steps
        self.inputs = data[inputs_str + extra_in_str]
        self.outputs = data[outputs_str]

        self.in_mean = data_mean[inputs_str + extra_in_str]
        self.in_std = data_std[inputs_str + extra_in_str]

        self.out_mean = data_mean[outputs_str]
        self.out_std = data_std[outputs_str]

        self.wet = wet

    def set_device(self, device):
        self.device = device

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        outputs = []
        for step in range(self.steps):
            if type(idx) == list:
                ind_in = [i * self.interval + self.lag * step for i in idx]
                ind_out = [i * self.interval + self.lag * (step + 1) for i in idx]

            elif type(idx) == slice:
                if idx.start == None and idx.stop == None:
                    idx = slice(0, self.size, idx.step)
                elif idx.start == None:
                    idx = slice(0, idx.stop, idx.step)
                elif idx.stop == None:
                    idx = slice(idx.start, self.size, idx.step)

                ind_in = slice(
                    idx.start, idx.stop * self.interval + self.lag * step, self.interval
                )
                ind_out = slice(
                    idx.start + self.lag,
                    idx.stop * self.interval + self.lag * (step + 1),
                    self.interval,
                )

            if type(idx) == int:
                ind_in = idx * self.interval + self.lag * step
                ind_out = idx * self.interval + self.lag * (step + 1)

            data_in = self.inputs.isel(time=ind_in)
            data_in = ((data_in - self.in_mean) / self.in_std).fillna(0)
            label = self.outputs.isel(time=ind_out)
            label = ((label - self.out_mean) / self.out_std).fillna(0)

            if type(idx) == int:
                data_in = data_in.to_array().transpose("variable", "y", "x").to_numpy()
                label = label.to_array().transpose("variable", "y", "x").to_numpy()
            else:
                data_in = (
                    data_in.to_array()
                    .transpose("time", "variable", "y", "x")
                    .to_numpy()
                )
                label = (
                    label.to_array().transpose("time", "variable", "y", "x").to_numpy()
                )

            outputs.append(torch.from_numpy(data_in).float())
            outputs.append(torch.from_numpy(label).float())

        return outputs


class data_CNN_Dynamic(torch.utils.data.Dataset):

    def __init__(self, data_in, data_out, wet, std_dict=None, device="cuda"):
        super().__init__()
        self.device = device
        num_inputs = data_in.shape[3]
        num_outputs = data_out.shape[3]
        self.size = data_in.shape[0]

        data_in = np.nan_to_num(data_in)
        data_out = np.nan_to_num(data_out)

        if std_dict is None:
            std_data = np.nanstd(data_in, axis=(0, 1, 2))
            mean_data = np.nanmean(data_in, axis=(0, 1, 2))
            std_label = np.nanstd(data_out, axis=(0, 1, 2))
            mean_label = np.nanmean(data_out, axis=(0, 1, 2))
        else:
            std_data = std_dict["s_in"]
            mean_data = std_dict["m_in"]
            std_label = std_dict["s_out"]
            mean_label = std_dict["m_out"]

        self.wet = wet

        # data_in[:, :, :, -1] = (data_in[:, :, :, -1] + 1)

        for i in range(num_inputs):
            data_in[:, :, :, i] = (data_in[:, :, :, i] - mean_data[i]) / std_data[i]

        for i in range(num_outputs):
            data_out[:, :, :, i] = (data_out[:, :, :, i] - mean_label[i]) / std_label[i]

        data_in = torch.from_numpy(data_in).type(torch.float32).to(device="cpu")
        data_out = torch.from_numpy(data_out).type(torch.float32).to(device="cpu")

        std_dict = {
            "s_in": std_data,
            "s_out": std_label,
            "m_in": mean_data,
            "m_out": mean_label,
        }

        if wet == None:
            self.input = torch.swapaxes(torch.swapaxes(data_in, 1, 3), 2, 3)
            self.output = torch.swapaxes(torch.swapaxes(data_out, 1, 3), 2, 3)

        else:
            self.input = torch.mul(
                torch.swapaxes(torch.swapaxes(data_in, 1, 3), 2, 3), wet
            )
            self.output = torch.mul(
                torch.swapaxes(torch.swapaxes(data_out, 1, 3), 2, 3), wet
            )

        self.norm_vals = std_dict

    def set_device(self, device):
        self.device = device

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_in = self.input[idx]
        label = self.output[idx]
        return data_in.to(device=self.device), label.to(device=self.device)


class data_CNN_steps_Dynamic(torch.utils.data.Dataset):

    def __init__(
        self, data_in, data_out, steps, wet=None, std_dict=None, device="cuda"
    ):
        super().__init__()
        self.device = device
        steps = len(data_out)
        self.steps = steps
        num_inputs = data_in[0].shape[3]
        num_outputs = data_out[0].shape[3]
        self.size = data_in[0].shape[0]
        self.wet = wet

        for i in range(steps):
            data_out[i] = np.nan_to_num(data_out[i])
            data_in[i] = np.nan_to_num(data_in[i])

        if std_dict is None:
            std_data = np.nanstd(data_in[0], axis=(0, 1, 2))
            mean_data = np.nanmean(data_in[0], axis=(0, 1, 2))
            std_label = np.nanstd(data_out[0], axis=(0, 1, 2))
            mean_label = np.nanmean(data_out[0], axis=(0, 1, 2))
        else:
            std_data = std_dict["s_in"]
            mean_data = std_dict["m_in"]
            std_label = std_dict["s_out"]
            mean_label = std_dict["m_out"]

        for j in range(steps):
            for i in range(num_outputs):
                data_out[j][:, :, :, i] = (
                    data_out[j][:, :, :, i] - mean_label[i]
                ) / std_label[i]
            for i in range(num_inputs):
                data_in[j][:, :, :, i] = (
                    data_in[j][:, :, :, i] - mean_data[i]
                ) / std_data[i]

        for j in range(steps):
            data_out[j] = (
                torch.from_numpy(data_out[j]).type(torch.float32).to(device="cpu")
            )
            data_in[j] = (
                torch.from_numpy(data_in[j]).type(torch.float32).to(device="cpu")
            )

        std_dict = {
            "s_in": std_data,
            "s_out": std_label,
            "m_in": mean_data,
            "m_out": mean_label,
        }

        if wet == None:
            for j in range(steps):
                data_out[j] = torch.swapaxes(torch.swapaxes(data_out[j], 1, 3), 2, 3)
                data_in[j] = torch.swapaxes(torch.swapaxes(data_in[j], 1, 3), 2, 3)
        else:
            for j in range(steps):
                data_out[j] = torch.mul(
                    torch.swapaxes(torch.swapaxes(data_out[j], 1, 3), 2, 3), wet
                )
                data_in[j] = torch.mul(
                    torch.swapaxes(torch.swapaxes(data_in[j], 1, 3), 2, 3), wet
                )

        self.input = data_in

        self.output = data_out
        self.norm_vals = std_dict

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data = [
            self.input[0][idx].to(device=self.device),
            self.output[0][idx].to(device=self.device),
        ]
        for k in range(1, self.steps):
            data.append(self.input[k][idx].to(device=self.device))
            data.append(self.output[k][idx].to(device=self.device))

        return tuple(data)


class data_CNN_steps(torch.utils.data.Dataset):

    def __init__(self, data_in, data_out, wet=None, device="cuda"):
        steps = len(data_out)
        self.steps = steps
        super().__init__()
        num_inputs = data_in[0].shape[3]
        num_outputs = data_out[0].shape[3]
        self.size = data_in[0].shape[0]
        self.wet = wet

        for i in range(steps):
            data_out[i] = np.nan_to_num(data_out[i])
            data_in[i] = np.nan_to_num(data_in[i])

        std_data = np.nanstd(data_in[0], axis=(0, 1, 2))
        mean_data = np.nanmean(data_in[0], axis=(0, 1, 2))
        std_label = np.nanstd(data_out[0], axis=(0, 1, 2))
        mean_label = np.nanmean(data_out[0], axis=(0, 1, 2))

        for j in range(steps):
            for i in range(num_outputs):
                data_out[j][:, :, :, i] = (
                    data_out[j][:, :, :, i] - mean_label[i]
                ) / std_label[i]
            for i in range(num_inputs):
                data_in[j][:, :, :, i] = (
                    data_in[j][:, :, :, i] - mean_data[i]
                ) / std_data[i]

        for j in range(steps):
            data_out[j] = (
                torch.from_numpy(data_out[j]).type(torch.float32).to(device=device)
            )
            data_in[j] = (
                torch.from_numpy(data_in[j]).type(torch.float32).to(device=device)
            )

        std_dict = {
            "s_in": std_data,
            "s_out": std_label,
            "m_in": mean_data,
            "m_out": mean_label,
        }

        if wet == None:
            for j in range(steps):
                data_out[j] = torch.swapaxes(torch.swapaxes(data_out[j], 1, 3), 2, 3)
                data_in[j] = torch.swapaxes(torch.swapaxes(data_in[j], 1, 3), 2, 3)
        else:
            for j in range(steps):
                data_out[j] = torch.mul(
                    torch.swapaxes(torch.swapaxes(data_out[j], 1, 3), 2, 3), wet
                )
                data_in[j] = torch.mul(
                    torch.swapaxes(torch.swapaxes(data_in[j], 1, 3), 2, 3), wet
                )

        self.input = data_in

        self.output = data_out
        self.norm_vals = std_dict

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data = [self.input[0][idx], self.output[0][idx]]
        for k in range(1, self.steps):
            data.append(self.input[k][idx])
            data.append(self.output[k][idx])

        return tuple(data)


class data_CNN(torch.utils.data.Dataset):

    def __init__(self, data_in, data_out, wet, device="cuda"):

        super().__init__()
        num_inputs = data_in.shape[3]
        num_outputs = data_out.shape[3]
        self.size = data_in.shape[0]

        data_in = np.nan_to_num(data_in)
        data_out = np.nan_to_num(data_out)

        std_data = np.nanstd(data_in, axis=(0, 1, 2))
        mean_data = np.nanmean(data_in, axis=(0, 1, 2))
        std_label = np.nanstd(data_out, axis=(0, 1, 2))
        mean_label = np.nanmean(data_out, axis=(0, 1, 2))

        self.wet = wet

        for i in range(num_inputs):
            data_in[:, :, :, i] = (data_in[:, :, :, i] - mean_data[i]) / std_data[i]

        for i in range(num_outputs):
            data_out[:, :, :, i] = (data_out[:, :, :, i] - mean_label[i]) / std_label[i]

        data_in = torch.from_numpy(data_in).type(torch.float32).to(device=device)
        data_out = torch.from_numpy(data_out).type(torch.float32).to(device=device)

        std_dict = {
            "s_in": std_data,
            "s_out": std_label,
            "m_in": mean_data,
            "m_out": mean_label,
        }

        if wet == None:
            self.input = torch.swapaxes(torch.swapaxes(data_in, 1, 3), 2, 3)
            self.output = torch.swapaxes(torch.swapaxes(data_out, 1, 3), 2, 3)

        else:
            self.input = torch.mul(
                torch.swapaxes(torch.swapaxes(data_in, 1, 3), 2, 3), wet
            )
            self.output = torch.mul(
                torch.swapaxes(torch.swapaxes(data_out, 1, 3), 2, 3), wet
            )

        self.norm_vals = std_dict

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_in = self.input[idx]
        label = self.output[idx]
        return data_in, label


class data_CNN_Lateral(torch.utils.data.Dataset):

    def __init__(
        self,
        data_in,
        data_out,
        wet,
        N_atm,
        Nb,
        device="cuda",
        wet_atm=False,
        norms="None",
        N_vars=None,
    ):
        super().__init__()
        self.device = device
        num_inputs = data_in.shape[3]
        num_outputs = data_out.shape[3]
        self.size = data_in.shape[0]

        data_in = np.nan_to_num(data_in)
        data_out = np.nan_to_num(data_out)

        if norms != "None":
            std_data = norms["s_in"]
            mean_data = norms["m_in"]
            std_label = norms["s_out"]
            mean_label = norms["m_out"]
        else:

            std_data = np.nanstd(data_in, axis=(0, 1, 2))
            mean_data = np.nanmean(data_in, axis=(0, 1, 2))
            std_label = np.nanstd(data_out, axis=(0, 1, 2))
            mean_label = np.nanmean(data_out, axis=(0, 1, 2))

            std_data[int(num_outputs + N_atm) : int(2 * num_outputs + N_atm)] = (
                std_data[:num_outputs]
            )
            mean_data[int(num_outputs + N_atm) : int(2 * num_outputs + N_atm)] = (
                mean_data[:num_outputs]
            )

        self.wet = wet

        for i in range(num_inputs):
            data_in[:, :, :, i] = (data_in[:, :, :, i] - mean_data[i]) / std_data[i]

        for i in range(int(num_outputs + N_atm), int(2 * num_outputs + N_atm)):
            data_in[:, Nb:-Nb, Nb:-Nb, i] = 0.0

        for i in range(num_outputs):
            data_out[:, :, :, i] = (data_out[:, :, :, i] - mean_label[i]) / std_label[i]

        data_in = torch.from_numpy(data_in).type(torch.float32).to(device="cpu")
        data_out = torch.from_numpy(data_out).type(torch.float32).to(device="cpu")

        std_dict = {
            "s_in": std_data,
            "s_out": std_label,
            "m_in": mean_data,
            "m_out": mean_label,
        }

        if type(wet) == list:
            temp_in = torch.swapaxes(torch.swapaxes(data_in, 1, 3), 2, 3)
            temp_out = torch.swapaxes(torch.swapaxes(data_out, 1, 3), 2, 3)
            for i in range(len(wet)):
                temp_in[:, i * N_vars : i * N_vars] = torch.mul(
                    temp_in[:, i * N_vars : i * N_vars], wet[i]
                )
                start = num_outputs + N_atm
                temp_in[:, start + i * N_vars : start + i * N_vars] = torch.mul(
                    temp_in[:, start + i * N_vars : start + i * N_vars], wet[i]
                )
                temp_out[:, i * N_vars : i * N_vars] = torch.mul(
                    temp_out[:, i * N_vars : i * N_vars], wet[i]
                )
            temp_in[:, num_outputs : num_outputs + N_atm] = torch.mul(
                temp_in[:, num_outputs : num_outputs + N_atm], wet[0]
            )
            self.input = temp_in
            self.output = temp_out

        elif wet == None:
            self.input = torch.swapaxes(torch.swapaxes(data_in, 1, 3), 2, 3)
            self.output = torch.swapaxes(torch.swapaxes(data_out, 1, 3), 2, 3)

        else:

            if wet_atm:
                self.input = torch.swapaxes(torch.swapaxes(data_in, 1, 3), 2, 3)
                self.output = torch.swapaxes(torch.swapaxes(data_out, 1, 3), 2, 3)
                for i in range(num_outputs):
                    self.input[:, i] = torch.mul(self.input[:, i], wet)
                    self.output[:, i] = torch.mul(self.input[:, i], wet)
                for i in range(int(num_outputs + N_atm), int(2 * num_outputs + N_atm)):
                    self.input[:, i] = torch.mul(self.input[:, i], wet)
            else:
                self.input = torch.mul(
                    torch.swapaxes(torch.swapaxes(data_in, 1, 3), 2, 3), wet
                )
                self.output = torch.mul(
                    torch.swapaxes(torch.swapaxes(data_out, 1, 3), 2, 3), wet
                )
        self.norm_vals = std_dict

    def set_device(self, device):
        self.device = device

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_in = self.input[idx]
        label = self.output[idx]
        return data_in.to(device=self.device), label.to(device=self.device)


class data_CNN_steps_Lateral(torch.utils.data.Dataset):

    def __init__(
        self,
        data_in,
        data_out,
        steps,
        wet,
        N_atm,
        Nb,
        device="cuda",
        wet_atm=False,
        norms="None",
        N_vars=None,
    ):
        super().__init__()
        self.device = device
        steps = len(data_out)
        self.steps = steps
        num_inputs = data_in[0].shape[3]
        num_outputs = data_out[0].shape[3]
        self.size = data_in[0].shape[0]
        self.wet = wet

        for i in range(steps):
            data_out[i] = np.nan_to_num(data_out[i])
            data_in[i] = np.nan_to_num(data_in[i])

        if norms != "None":
            std_data = norms["s_in"]
            mean_data = norms["m_in"]
            std_label = norms["s_out"]
            mean_label = norms["m_out"]
        else:
            std_data = np.nanstd(data_in[0], axis=(0, 1, 2))
            mean_data = np.nanmean(data_in[0], axis=(0, 1, 2))
            std_label = np.nanstd(data_out[0], axis=(0, 1, 2))
            mean_label = np.nanmean(data_out[0], axis=(0, 1, 2))

            std_data[int(num_outputs + N_atm) : int(2 * num_outputs + N_atm)] = (
                std_data[:num_outputs]
            )
            mean_data[int(num_outputs + N_atm) : int(2 * num_outputs + N_atm)] = (
                mean_data[:num_outputs]
            )

        for j in range(steps):
            for i in range(num_outputs):
                data_out[j][:, :, :, i] = (
                    data_out[j][:, :, :, i] - mean_label[i]
                ) / std_label[i]

            for i in range(num_inputs):
                data_in[j][:, :, :, i] = (
                    data_in[j][:, :, :, i] - mean_data[i]
                ) / std_data[i]

            for i in range(int(num_outputs + N_atm), int(2 * num_outputs + N_atm)):
                data_in[j][:, Nb:-Nb, Nb:-Nb, i] = 0.0

        for j in range(steps):
            data_out[j] = (
                torch.from_numpy(data_out[j]).type(torch.float32).to(device="cpu")
            )
            data_in[j] = (
                torch.from_numpy(data_in[j]).type(torch.float32).to(device="cpu")
            )

        std_dict = {
            "s_in": std_data,
            "s_out": std_label,
            "m_in": mean_data,
            "m_out": mean_label,
        }

        if type(wet) == list:
            for j in range(steps):
                temp_in = torch.swapaxes(torch.swapaxes(data_in[j], 1, 3), 2, 3)
                temp_out = torch.swapaxes(torch.swapaxes(data_out[j], 1, 3), 2, 3)
                for i in range(len(wet)):
                    temp_in[:, i * N_vars : i * N_vars] = torch.mul(
                        temp_in[:, i * N_vars : i * N_vars], wet[i]
                    )
                    start = num_outputs + N_atm
                    temp_in[:, start + i * N_vars : start + i * N_vars] = torch.mul(
                        temp_in[:, start + i * N_vars : start + i * N_vars], wet[i]
                    )
                    temp_out[:, i * N_vars : i * N_vars] = torch.mul(
                        temp_out[:, i * N_vars : i * N_vars], wet[i]
                    )
                print(num_outputs, start)
                temp_in[:, num_outputs:start] = torch.mul(
                    temp_in[:, num_outputs:start], wet[0]
                )
                data_out[j] = temp_out.clone()
                data_in[j] = temp_in.clone()
        elif wet == None:
            for j in range(steps):
                data_out[j] = torch.swapaxes(torch.swapaxes(data_out[j], 1, 3), 2, 3)
                data_in[j] = torch.swapaxes(torch.swapaxes(data_in[j], 1, 3), 2, 3)
        else:
            for j in range(steps):
                data_out[j] = torch.mul(
                    torch.swapaxes(torch.swapaxes(data_out[j], 1, 3), 2, 3), wet
                )
                data_in[j] = torch.mul(
                    torch.swapaxes(torch.swapaxes(data_in[j], 1, 3), 2, 3), wet
                )

        self.input = data_in

        self.output = data_out
        self.norm_vals = std_dict

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data = [
            self.input[0][idx].to(device=self.device),
            self.output[0][idx].to(device=self.device),
        ]
        for k in range(1, self.steps):
            data.append(self.input[k][idx].to(device=self.device))
            data.append(self.output[k][idx].to(device=self.device))

        return tuple(data)


def get_oceanGPT_data(s, e, steps, inputs, extra_in, wet):
    # inputs, extra_in and outputs are xarrays
    num_input_vars = len(inputs)

    inputs = torch.stack(
        [torch.tensor(data_input.to_numpy()) for data_input in inputs], dim=0
    )
    C, N, H, W = inputs.shape
    inputs = rearrange(inputs, "C N H W -> N C H W")
    inputs = inputs[s:e]
    inputs = torch.nan_to_num(inputs)
    inputs = torch.mul(inputs, wet)
    inputs = inputs.reshape(-1, steps, C, H, W)
    inputs = rearrange(inputs, "N T C H W -> N C T H W")

    # Do not use lateral boundary conditions here
    extra_in = extra_in[:-num_input_vars]
    extra_in = torch.stack(
        [torch.tensor(data_input.to_numpy()) for data_input in extra_in], dim=0
    )
    C, N, H, W = extra_in.shape
    extra_in = rearrange(extra_in, "C N H W -> N C H W")
    extra_in = extra_in[s:e]
    extra_in = torch.nan_to_num(extra_in)
    extra_in = torch.mul(extra_in, wet)
    extra_in = extra_in.reshape(-1, steps, C, H, W)
    extra_in = rearrange(extra_in, "N T C H W -> N C T H W")

    return inputs, extra_in


def get_recunet_data(s, e, inputs, extra_in):
    # Returns data of shape - N, C, H, W
    num_input_vars = len(inputs)

    inputs = torch.stack(
        [torch.tensor(data_input.to_numpy()) for data_input in inputs], dim=0
    )
    extra_in = torch.stack(
        [torch.tensor(data_input.to_numpy()) for data_input in extra_in], dim=0
    )
    assert (
        torch.nan_to_num(inputs[0, :, :4, :4])
        == torch.nan_to_num(extra_in[3, :, :4, :4])
    ).all()
    assert (
        torch.nan_to_num(inputs[1, :, :4, :4])
        == torch.nan_to_num(extra_in[4, :, :4, :4])
    ).all()
    assert (
        torch.nan_to_num(inputs[2, :, :4, :4])
        == torch.nan_to_num(extra_in[5, :, :4, :4])
    ).all()

    inputs = torch.cat([inputs, extra_in], dim=0)
    C, N, H, W = inputs.shape
    inputs = rearrange(inputs, "C N H W -> N C H W")
    inputs = inputs[s:e]
    # print("Nanmean: ", torch.nanmean(inputs, dim=[0,2,3]))
    # print("wet mask mean",inputs[:,:,wet==1].mean(dim=[0,2]))
    # print("wet mask std",inputs[:,:,wet==1].std(dim=[0,2]))
    inputs = torch.nan_to_num(inputs)
    # inputs = torch.mul(inputs, wet)
    # print("mean post multiplication", inputs.mean(dim=[0,2,3]))
    # print("std post multiplication", inputs.std(dim=[0,2,3]))

    return inputs


class RecUnetDataset(torch.utils.data.Dataset):
    # N C H W
    def __init__(
        self,
        data_path,
        steps,
        input_time_dim,
        presteps,
        output_channels,
        Nb,
        wet_path,
        device="cuda",
    ):
        super().__init__()
        self.data_path = data_path
        self.input = torch.load(data_path, map_location=torch.device("cpu"))
        self.wet = torch.load(wet_path, map_location=torch.device("cpu"))
        self.input_steps = steps + presteps * input_time_dim
        self.output_steps = steps
        self.output_offset = (1 + presteps) * input_time_dim

        self.Nb = Nb
        self.output_channels = output_channels
        self.presteps = presteps
        self.time_dim = input_time_dim
        self.device = device
        self.preprocess()

    def preprocess(self):
        N, C, H, W = self.input.shape
        inputs = rearrange(self.input, "N C H W -> N H W C")

        std_data = torch.std(inputs, dim=[0, 1, 2])
        mean_data = torch.mean(inputs, dim=[0, 1, 2])

        std_data[-self.output_channels :] = std_data[: self.output_channels]
        mean_data[-self.output_channels :] = mean_data[: self.output_channels]

        inputs = (inputs - mean_data) / (std_data)

        inputs[:, self.Nb : -self.Nb, self.Nb : -self.Nb, -self.output_channels :] = 0.0

        std_dict = {
            "s_in": std_data,
            "m_in": mean_data,
            "s_out": std_data[: self.output_channels],  # same
            "m_out": mean_data[: self.output_channels],  # same
        }

        print("Current norms: ", std_dict)

        self.norm_vals = std_dict
        inputs = rearrange(inputs, "N H W C -> N C H W")
        self.input = torch.mul(inputs, self.wet)

    def __len__(self):
        return len(self.input) - (self.output_steps + self.output_offset)

    def __getitem__(self, idx):
        # print(f"Input indices- {idx}:{idx+self.input_steps}\nTarget indices- {idx+self.output_offset}:{idx+self.output_offset+self.output_steps}")
        inputs = self.input[idx : idx + self.input_steps]
        targets = self.input[
            idx + self.output_offset : idx + self.output_offset + self.output_steps,
            : self.output_channels,
        ]
        assert inputs.shape[0] != 0
        assert targets.shape[0] != 0
        return inputs.to(self.device), targets.to(self.device)


class RecUnetEvalDataset(torch.utils.data.Dataset):
    # N C H W
    def __init__(
        self,
        data_path,
        steps,
        input_time_dim,
        presteps,
        output_channels,
        Nb,
        wet_path,
        device="cuda",
    ):
        super().__init__()
        self.data_path = data_path
        self.input = torch.load(data_path, map_location=torch.device("cpu"))
        self.wet = torch.load(wet_path, map_location=torch.device("cpu"))
        self.input_steps = steps + presteps * input_time_dim
        self.steps = steps
        self.output_steps = steps
        self.output_offset = (1 + presteps) * input_time_dim
        self.input_offset = presteps * input_time_dim

        self.Nb = Nb
        self.output_channels = output_channels
        self.presteps = presteps
        self.time_dim = input_time_dim
        self.device = device
        self.converted_ = False
        self.preprocess()

    def preprocess(self):
        N, C, H, W = self.input.shape
        inputs = rearrange(self.input, "N C H W -> N H W C")

        std_data = torch.std(inputs, dim=[0, 1, 2])
        mean_data = torch.mean(inputs, dim=[0, 1, 2])

        std_data[-self.output_channels :] = std_data[: self.output_channels]
        mean_data[-self.output_channels :] = mean_data[: self.output_channels]

        inputs = (inputs - mean_data) / (std_data)

        inputs[:, self.Nb : -self.Nb, self.Nb : -self.Nb, -self.output_channels :] = 0.0

        std_dict = {
            "s_in": std_data,
            "m_in": mean_data,
            "s_out": std_data[: self.output_channels],  # same
            "m_out": mean_data[: self.output_channels],  # same
        }

        print("Current norms: ", std_dict)

        self.norm_vals = std_dict
        inputs = rearrange(inputs, "N H W C -> N C H W")
        self.input = torch.mul(inputs, self.wet)

    def __len__(self):
        return (len(self.input) - self.input_offset) // (self.output_steps)

    def __getitem__(self, idx):
        if not self.converted_:
            # print(f"Input indices- {idx}:{idx+self.input_steps}\nTarget indices- {idx+self.output_offset}:{idx+self.output_offset+self.output_steps}")
            if idx == 0:
                inputs = self.input[idx : idx + self.input_steps]  # 0-10
                targets = self.input[
                    idx
                    + self.output_offset : idx
                    + self.output_offset
                    + self.output_steps,
                    : self.output_channels,
                ]  # 4-12
            else:
                inputs = self.input[
                    idx * (self.steps) : self.input_offset + (idx + 1) * (self.steps)
                ]  # i*8-2+(i+1)*8
                targets = self.input[
                    self.output_offset
                    + idx * (self.steps) : self.output_offset
                    + (idx + 1) * (self.steps),
                    : self.output_channels,
                ]  # 4+i*8-4+(i+1)*8
            assert inputs.shape[0] != 0
            assert targets.shape[0] != 0
            return inputs.to(self.device), targets.to(self.device)
        else:
            return self.input[idx].to(self.device), self.output[idx].to(self.device)

    def set_input(self, start, interval):
        self.input = self.input[start : start + interval + self.output_offset]

    def set_device_and_convert_returns(self, N_test, device="cpu"):
        self.device = device
        self.input.to(device)

        self.norm_vals["s_in"].to(device)
        self.norm_vals["m_in"].to(device)

        self.norm_vals["s_in"] = self.norm_vals["s_in"].numpy()
        self.norm_vals["m_in"] = self.norm_vals["m_in"].numpy()

        self.norm_vals["s_out"].to(device)
        self.norm_vals["m_out"].to(device)

        self.norm_vals["s_out"] = self.norm_vals["s_out"].numpy()
        self.norm_vals["m_out"] = self.norm_vals["m_out"].numpy()

        self.converted_ = True
        # hack
        self.output = self.input[
            self.output_offset : self.output_offset + N_test, : self.output_channels
        ]
        self.input = self.input[
            self.output_offset - 1 : self.output_offset - 1 + N_test
        ]


def gen_data_in(step, s, e, interval, lag, hist, inputs, extra_in):
    s = s + lag * step
    e = e + lag * step
    num_outs = len(inputs)
    num_extra = len(extra_in)
    temp_inputs = []
    for j in range(num_outs):
        print(f"Inputs: Getting {s}:{e}:{interval} vals for channel {j}")
        temp_inputs.append(inputs[j][s:e:interval].to_numpy())
    temp_extra = []
    for j in range(num_extra):
        print(f"Extra: Getting {s}:{e}:{interval} vals for channel {j}")
        temp_extra.append(extra_in[j][s:e:interval].to_numpy())

    data_in = np.stack((*temp_inputs, *temp_extra), -1)

    for i in range(hist):
        temp_inputs = []
        for j in range(num_outs):
            print(
                f"Hist: Getting {s - (hist - i) * lag}:{e - (hist - i) * lag}:{interval} vals for channel {j}"
            )
            temp_inputs.append(
                np.expand_dims(
                    inputs[j][
                        s - (hist - i) * lag : e - (hist - i) * lag : (interval)
                    ].to_numpy(),
                    -1,
                )
            )
        data_in = np.concatenate((data_in, *temp_inputs), axis=3)
    return data_in


def gen_data_out(step, s, e, lag, interval, outputs):
    s = s + lag * step
    e = e + lag * step

    num_outs = len(outputs)
    temp_outputs = []
    for j in range(num_outs):
        print(f"Outputs: Getting {s}:{e}:{interval} vals for channel {j}")
        temp_outputs.append(outputs[j][s:e:interval].to_numpy())

    data_out = np.stack(temp_outputs, -1)
    return data_out


def gen_data_in_test(step, s, N_test, lag, hist, inputs, extra_in):
    if lag == 0:
        lag = 1
    s = s + lag * step
    e = s + lag * N_test + lag * step

    num_outs = len(inputs)
    num_extra = len(extra_in)
    temp_inputs = []
    for j in range(num_outs):
        temp_inputs.append(inputs[j][s:e:lag].to_numpy())
    temp_extra = []
    for j in range(num_extra):
        temp_extra.append(extra_in[j][s:e:lag].to_numpy())

    data_in = np.stack((*temp_inputs, *temp_extra), -1)

    for i in range(hist):
        temp_inputs = []
        for j in range(num_outs):
            temp_inputs.append(
                np.expand_dims(
                    inputs[j][
                        s - (hist - i) * lag : e - (hist - i) * lag : (lag)
                    ].to_numpy(),
                    -1,
                )
            )
        data_in = np.concatenate((data_in, *temp_inputs), axis=3)
    return data_in


def gen_data_out_test(step, s, N_test, lag, hist, outputs):
    if lag == 0:
        lag = 1
    s = s + lag * step
    e = s + lag * N_test + lag * step

    num_outs = len(outputs)
    temp_outputs = []
    for j in range(num_outs):
        temp_outputs.append(outputs[j][s:e:lag].to_numpy())

    data_out = np.stack(temp_outputs, -1)
    return data_out


def gen_data(input_vars, extra_vars, output_vars, lag, factor, region="Kuroshio"):
    var_dict = {
        "um": "u_mean",
        "vm": "v_mean",
        "Tm": "T_mean",
        "ur": "u_res",
        "vr": "v_res",
        "Tr": "T_res",
        "u": "u",
        "v": "v",
        "T": "T",
        "tau_u": "tau_u",
        "tau_v": "tau_v",
        "tau": "tau",
        "t_ref": "t_ref",
    }

    data = xr.open_zarr(
        "/scratch/as15415/Data/Emulation_Data/Velocities_"
        + region
        + "_Factor_"
        + str(factor)
        + ".zarr/"
    )
    data_temp = xr.open_zarr(
        "/scratch/as15415/Data/Emulation_Data/Temperature_"
        + region
        + "_Factor_"
        + str(factor)
        + ".zarr/"
    )
    data_atmos = xr.open_zarr(
        "/scratch/as15415/Data/Emulation_Data/Data_Atmos_" + region + ".zarr/"
    )
    data_atmos = data_atmos.rename_dims({"lat": "yu_ocean", "lon": "xu_ocean"})
    data_atmos = data_atmos.rename({"lat": "yu_ocean", "lon": "xu_ocean"})

    data_temp["xu_ocean"] = data.xu_ocean.data
    data_temp["yu_ocean"] = data.yu_ocean.data

    data = xr.merge([data, data_temp, data_atmos])

    inputs = []
    extra_in = []
    outputs = []

    for var in input_vars:
        inputs.append(data[var_dict[var]])

    for var in extra_vars:
        extra_in.append(data[var_dict[var]])

    for var in output_vars:
        outputs.append(data[var_dict[var]][lag:])

    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs


def gen_3D_data(raw_data_path, input_vars, extra_vars, output_vars, lag=1, depth_mode="all"):
    data = xr.open_zarr(raw_data_path)

    inputs = []
    extra_in = []
    outputs = []

    for var in input_vars:
        if var == "zos":
            inputs.append(data[var])
        elif depth_mode == "surface":
            inputs.append(data[var][:, 0])
        elif depth_mode == "all":
            assert data[var].shape[1] == 19
            for i in range(data[var].shape[1]):
                inputs.append(data[var][:, i])

    for var in extra_vars:
        extra_in.append(data[var])

    for var in output_vars:
        if var == "zos":
            outputs.append(data[var][lag:])
        elif depth_mode == "surface":
            outputs.append(data[var][lag:, 0])
        elif depth_mode == "all":
            assert data[var].shape[1] == 19
            for i in range(data[var].shape[1]):
                outputs.append(data[var][lag:, i])

    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs


def gen_data_global(input_vars, extra_vars, output_vars, lag, res="1"):
    var_dict = {
        "um": "u_mean",
        "vm": "v_mean",
        "Tm": "T_mean",
        "ur": "u_res",
        "vr": "v_res",
        "Tr": "T_res",
        "u": "u",
        "v": "v",
        "T": "T",
        "tau_u": "tau_u",
        "tau_v": "tau_v",
        "tau": "tau",
        "t_ref": "t_ref",
    }

    data = xr.open_zarr(
        "/scratch/as15415/Data/Emulation_Data/Global_Ocean_" + res + "deg.zarr"
    )
    if res != "1":
        data_atmos = (
            xr.open_zarr(
                "/scratch/as15415/Data/Emulation_Data/Data_Atmos_" + res + "_deg.zarr"
            )
            .drop(["xu_ocean", "T_mean"])
            .assign_coords({"lon": data.xu_ocean.data})
        )
    else:
        data_atmos = xr.open_zarr(
            "/scratch/as15415/Data/Emulation_Data/Data_Atmos_" + res + "_deg.zarr"
        )
    data_atmos = data_atmos.rename_dims({"lat": "yu_ocean", "lon": "xu_ocean"})
    data_atmos = data_atmos.rename({"lat": "yu_ocean", "lon": "xu_ocean"})

    data_atmos["xu_ocean"] = data.xu_ocean.data
    data_atmos["yu_ocean"] = data.yu_ocean.data

    data = xr.merge([data, data_atmos])

    inputs = []
    extra_in = []
    outputs = []

    for var in input_vars:
        inputs.append(data[var_dict[var]])

    for var in extra_vars:
        extra_in.append(data[var_dict[var]])

    for var in output_vars:
        outputs.append(data[var_dict[var]][lag:])

    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs


def gen_data_global_new(input_vars, extra_vars, output_vars, lag, run_type=""):
    var_dict = {
        "u": "u",
        "v": "v",
        "T": "T",
        "tau_u": "tau_u",
        "tau_v": "tau_v",
        "t_ref": "t_ref",
    }
    if run_type != "":
        run_type = "_" + run_type
    data = xr.open_zarr(
        "/scratch/as15415/Data/Emulation_Data/Global_Ocean_1deg"
        + run_type
        + "_New.zarr"
    )

    data_atmos = xr.open_zarr(
        "/scratch/as15415/Data/Emulation_Data/Data_Atmos_1deg" + run_type + "_New.zarr"
    )
    data_atmos = data_atmos.rename_dims({"lat": "y", "lon": "x"})
    data_atmos = data_atmos.rename({"lat": "y", "lon": "x"})

    data = data.sel(time=slice(data_atmos.time[0], data_atmos.time[-1]))
    data_atmos = data_atmos.sel(time=slice(data.time[0], data.time[-1]))

    data_atmos["xu_ocean"] = data.x.data
    data_atmos["yu_ocean"] = data.y.data

    data = xr.merge([data, data_atmos])

    inputs = []
    extra_in = []
    outputs = []

    for var in input_vars:
        inputs.append(data[var_dict[var]])

    for var in extra_vars:
        extra_in.append(data[var_dict[var]])

    for var in output_vars:
        outputs.append(data[var_dict[var]][lag:])

    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs


def gen_data_lateral(input_vars, extra_vars, output_vars, lag, factor, region, Nb=2):
    var_dict = {
        "um": "u_mean",
        "vm": "v_mean",
        "Tm": "T_mean",
        "ur": "u_res",
        "vr": "v_res",
        "Tr": "T_res",
        "u": "u",
        "v": "v",
        "T": "T",
        "tau_u": "tau_u",
        "tau_v": "tau_v",
        "tau": "tau",
        "t_ref": "t_ref",
    }

    data = xr.open_zarr(
        "/scratch/as15415/Data/Emulation_Data/Velocities_"
        + region
        + "_Factor_"
        + str(factor)
        + ".zarr/"
    )
    data_temp = xr.open_zarr(
        "/scratch/as15415/Data/Emulation_Data/Temperature_"
        + region
        + "_Factor_"
        + str(factor)
        + ".zarr/"
    )
    data_atmos = xr.open_zarr(
        "/scratch/as15415/Data/Emulation_Data/Data_Atmos_" + region + ".zarr/"
    )
    data_atmos = data_atmos.rename_dims({"lat": "yu_ocean", "lon": "xu_ocean"})
    data_atmos = data_atmos.rename({"lat": "yu_ocean", "lon": "xu_ocean"})

    data_temp["xu_ocean"] = data.xu_ocean.data
    data_temp["yu_ocean"] = data.yu_ocean.data

    data = xr.merge([data, data_temp, data_atmos])
    inputs = []
    extra_in = []
    outputs = []

    for var in input_vars:
        temp = data[var_dict[var]].copy(deep=True)
        #         temp[:,:Nb,:] = 0.*temp[0,:Nb,:]
        #         temp[:,-Nb:,:] = 0.*temp[:,-Nb:,:]
        #         temp[:,:,:Nb] = 0.*temp[:,:,:Nb]
        #         temp[:,:,-Nb:] = 0.*temp[:,:,-Nb:]
        inputs.append(temp)

    for var in extra_vars:
        extra_in.append(data[var_dict[var]])

    for var in input_vars:
        temp = data[var_dict[var]].copy(deep=True)
        temp[:, Nb:-Nb, Nb:-Nb] = 0.0 * temp[0, Nb:-Nb, Nb:-Nb]
        extra_in.append(temp)

    for var in output_vars:
        temp = data[var_dict[var]].copy(deep=True)
        #         temp[:,:Nb,:] = 0.*temp[0,:Nb,:]
        #         temp[:,-Nb:,:] = 0.*temp[:,-Nb:,:]
        #         temp[:,:,:Nb] = 0.*temp[:,:,:Nb]
        #         temp[:,:,-Nb:] = 0.*temp[:,:,-Nb:]
        outputs.append(data[var_dict[var]][lag:])

    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs


def gen_data_025(input_vars, extra_vars, output_vars, lag, lat, lon):
    var_dict = {
        "um": "u_mean",
        "vm": "v_mean",
        "Tm": "T_mean",
        "ur": "u_res",
        "vr": "v_res",
        "Tr": "T_res",
        "u": "u",
        "v": "v",
        "T": "T",
        "tau_u": "tau_u",
        "tau_v": "tau_v",
        "tau": "tau",
        "t_ref": "t_ref",
    }

    data = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Global_Ocean_025deg.zarr")
    data_atmos = (
        xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Data_Atmos_025_deg.zarr")
        .drop(["xu_ocean", "T_mean"])
        .assign_coords({"lon": data.xu_ocean.data})
    )
    data_atmos = data_atmos.rename_dims({"lat": "yu_ocean", "lon": "xu_ocean"})
    data_atmos = data_atmos.rename({"lat": "yu_ocean", "lon": "xu_ocean"})

    data_atmos["xu_ocean"] = data.xu_ocean.data
    data_atmos["yu_ocean"] = data.yu_ocean.data

    data = xr.merge([data, data_atmos])

    data = data.sel(yu_ocean=slice(lat[0], lat[1]), xu_ocean=slice(lon[0], lon[1]))

    inputs = []
    extra_in = []
    outputs = []

    for var in input_vars:
        inputs.append(data[var_dict[var]])

    for var in extra_vars:
        extra_in.append(data[var_dict[var]])

    for var in output_vars:
        outputs.append(data[var_dict[var]][lag:])

    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs


def gen_data_025_lateral(
    input_vars,
    extra_vars,
    output_vars,
    lag,
    lat,
    lon,
    Nb=2,
    filter_T=False,
    filter_width=20,
    area=None,
):
    var_dict = {
        "um": "u_mean",
        "vm": "v_mean",
        "Tm": "T_mean",
        "ur": "u_res",
        "vr": "v_res",
        "Tr": "T_res",
        "u": "u",
        "v": "v",
        "T": "T",
        "tau_u": "tau_u",
        "tau_v": "tau_v",
        "tau": "tau",
        "t_ref": "t_ref",
    }

    data = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Global_Ocean_025deg.zarr")
    data_atmos = (
        xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Data_Atmos_025_deg.zarr")
        .drop(["xu_ocean", "T_mean"])
        .assign_coords({"lon": data.xu_ocean.data})
    )

    #     data_atmos = xr.open_zarr("/scratch/as15415/Data/Emulation_Data/Data_Atmos_025_deg_filtered.zarr").assign_coords({"lon":data.xu_ocean.data})
    data_atmos = data_atmos.rename_dims({"lat": "yu_ocean", "lon": "xu_ocean"})
    data_atmos = data_atmos.rename({"lat": "yu_ocean", "lon": "xu_ocean"})

    #     data_atmos["xu_ocean"] = data.xu_ocean.data
    #     data_atmos["yu_ocean"] = data.yu_ocean.data
    #     data_atmos["time"] = data.time.data

    data = xr.merge([data, data_atmos])

    data = data.sel(yu_ocean=slice(lat[0], lat[1]), xu_ocean=slice(lon[0], lon[1]))

    inputs = []
    extra_in = []
    outputs = []

    for i, var in enumerate(input_vars):
        print(f"Extracting {var_dict[var]} and appending to input at index {i}")
        inputs.append(data[var_dict[var]])

    for i, var in enumerate(extra_vars):
        print(f"Extracting {var_dict[var]} and appending to extra at index {i}")
        if var == "t_ref" and filter_T:
            if filter_width == "mean":
                data[var_dict[var]] = (
                    data[var_dict[var]] * 0
                    + (
                        (data[var_dict[var]] * area).sum(dim=["xu_ocean", "yu_ocean"])
                        / area.sum()
                    ).compute()
                )
            else:
                data[var_dict[var]].data = gaussian_filter(
                    data[var_dict[var]], filter_width
                )

        extra_in.append(data[var_dict[var]])

    cur_extra_len = len(extra_vars)
    for i, var in enumerate(input_vars):
        print(
            f"Extracting {var_dict[var]} and appending to extra at index {i+cur_extra_len}"
        )
        temp = data[var_dict[var]].copy(deep=True)
        temp[:, Nb:-Nb, Nb:-Nb] = 0.0 * temp[0, Nb:-Nb, Nb:-Nb]
        extra_in.append(temp)

    for var in output_vars:
        outputs.append(data[var_dict[var]][lag:])

    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs


def gen_data_025_lateral_subsurf(
    input_vars, extra_vars, output_vars, depth, lag, lat, lon, Nb=2
):
    var_dict = {
        "um": "u_mean",
        "vm": "v_mean",
        "Tm": "T_mean",
        "ur": "u_res",
        "vr": "v_res",
        "Tr": "T_res",
        "u": "u",
        "v": "v",
        "T": "T",
        "tau_u": "tau_u",
        "tau_v": "tau_v",
        "tau": "tau",
        "t_ref": "t_ref",
    }

    data = xr.open_zarr(
        "/scratch/as15415/Data/Emulation_Data/Global_Ocean_025deg_depth_"
        + depth
        + ".zarr"
    )
    data_atmos = xr.open_zarr(
        "/scratch/as15415/Data/Emulation_Data/Global_Ocean_025deg_5_day_Avg.zarr"
    )
    data_atmos = data_atmos.rename({"u": "tau_u", "v": "tau_v", "T": "t_ref"})
    data_atmos = data_atmos.drop(
        ["u_mean", "v_mean", "T_mean", "u_res", "v_res", "T_res"]
    )

    data = xr.merge([data, data_atmos])

    data = data.sel(yu_ocean=slice(lat[0], lat[1]), xu_ocean=slice(lon[0], lon[1]))

    inputs = []
    extra_in = []
    outputs = []

    for var in input_vars:
        inputs.append(data[var_dict[var]])

    for var in extra_vars:
        extra_in.append(data[var_dict[var]])

    for var in input_vars:
        temp = data[var_dict[var]].copy(deep=True)
        temp[:, Nb:-Nb, Nb:-Nb] = 0.0 * temp[0, Nb:-Nb, Nb:-Nb]
        extra_in.append(temp)

    for var in output_vars:
        outputs.append(data[var_dict[var]][lag:])

    inputs = tuple(inputs)
    extra_in = tuple(extra_in)
    outputs = tuple(outputs)

    return inputs, extra_in, outputs


def get_norms(region, inputs, extra_in, outputs, Lateral=True):

    if region == "Africa_Ext":
        mean_dict = {
            "um": 4.82752338e-2,
            "vm": 1.1804e-2,
            "Tm": 10.8398,
            "tau_u": 6.83281880e-5,
            "tau_v": -1.04916221e-5,
            "t_ref": 2.85715049e2,
            "um_0": 2.80994716e-02,
            "vm_0": 1.1804e-2,
            "Tm_0": 10.8398,
            "um_99": 2.93019213e-02,
            "vm_99": -4.84687595e-03,
            "Tm_99": 8.91647284,
            "um_253": 2.49547245e-02,
            "vm_253": -4.48418542e-03,
            "Tm_253": 6.77900864,
        }

        std_dict = {
            "um": 1.78659424e-1,
            "vm": 1.62300227e-01,
            "Tm": 8.35850152,
            "tau_u": 1.32713360e-4,
            "tau_v": 9.77289618e-5,
            "t_ref": 7.38032223e0,
            "um_0": 1.78659424e-1,
            "vm_0": 1.62300227e-01,
            "Tm_0": 8.35850152,
            "um_99": 0.14836239,
            "vm_99": 0.14007329,
            "Tm_99": 7.49790942,
            "um_253": 0.13086694,
            "vm_253": 0.12258011,
            "Tm_253": 5.91231414,
        }

    elif region == "Gulf_Stream_Ext":
        mean_dict = {
            "um": 4.48291145e-02,
            "vm": -1.05211059e-02,
            "Tm": 1.21081783e01,
            "tau_u": 3.82348062e-05,
            "tau_v": 8.21162517e-06,
            "t_ref": 2.85715049e2,
            "um_0": 4.48291145e-02,
            "vm_0": -1.05211059e-02,
            "Tm_0": 1.21081783e01,
            "um_99": 2.93019213e-02,
            "vm_99": 4.04511051e-03,
            "Tm_99": 9.98328467e00,
            "um_253": 2.39374764e-02,
            "vm_253": 4.68866993e-03,
            "Tm_253": 8.55389633e00,
        }

        std_dict = {
            "um": 1.51401489e-01,
            "vm": 1.43377056e-01,
            "Tm": 9.03735246e00,
            "tau_u": 9.60773988e-05,
            "tau_v": 8.42097158e-05,
            "t_ref": 9.21912876e00,
            "um_0": 1.51401489e-01,
            "vm_0": 1.43377056e-01,
            "Tm_0": 9.03735246e00,
            "um_99": 0.13803744,
            "vm_99": 0.12650987,
            "Tm_99": 8.29932087,
            "um_253": 0.12300715,
            "vm_253": 0.11049998,
            "Tm_253": 7.68735387,
        }

    elif region == "Tropics_Ext":
        mean_dict = {
            "um": -6.56001477e-02,
            "vm": 3.03974905e-02,
            "Tm": 1.85474497e01,
            "tau_u": -3.79044117e-05,
            "tau_v": -4.91180848e-06,
            "t_ref": 2.96640798e02,
            "um_0": -6.56001477e-02,
            "vm_0": 3.03974905e-02,
            "Tm_0": 1.85474497e01,
            "um_99": 4.71830777e-03,
            "vm_99": 4.80014641e-04,
            "Tm_99": 1.22588992e01,
            "um_253": -6.63711393e-03,
            "vm_253": 1.01151822e-03,
            "Tm_253": 8.36092632e00,
        }

        std_dict = {
            "um": 1.95064212e-01,
            "vm": 1.42985598e-01,
            "Tm": 1.13369541e01,
            "tau_u": 4.90698542e-05,
            "tau_v": 3.49303944e-05,
            "t_ref": 2.97622406e00,
            "um_0": 1.95064212e-01,
            "vm_0": 1.42985598e-01,
            "Tm_0": 1.13369541e01,
            "um_99": 0.14478501,
            "vm_99": 0.0861209,
            "Tm_99": 9.17627311,
            "um_253": 0.0638834,
            "vm_253": 0.05032483,
            "Tm_253": 6.74403382,
        }

    elif region == "Quiescent_Ext":
        mean_dict = {
            "um": 3.18046221e-02,
            "vm": 1.31442399e-03,
            "Tm": 1.61681938e01,
            "tau_u": 2.58048575e-05,
            "tau_v": -3.11680868e-06,
            "t_ref": 2.87996087e02,
            "um_0": 3.18046221e-02,
            "vm_0": 1.31442399e-03,
            "Tm_0": 1.61681938e01,
            "um_99": 1.87243772e-02,
            "vm_99": 2.10032583e-03,
            "Tm_99": 1.41782428e01,
            "um_253": 1.18352078e-02,
            "vm_253": 1.63565767e-03,
            "Tm_253": 1.10941377e01,
        }
        std_dict = {
            "um": 1.04966172e-01,
            "vm": 8.70280337e-02,
            "Tm": 7.18070321e00,
            "tau_u": 1.25641199e-04,
            "tau_v": 9.48015232e-05,
            "t_ref": 7.01619519e00,
            "um_0": 1.04966172e-01,
            "vm_0": 8.70280337e-02,
            "Tm_0": 7.18070321e00,
            "um_99": 0.07786532,
            "vm_99": 0.05987989,
            "Tm_99": 6.86105509,
            "um_253": 0.07023565,
            "vm_253": 0.05248215,
            "Tm_253": 5.19177608,
        }

    if Lateral:
        mean_in = np.zeros(len(inputs + extra_in + inputs))
        std_in = np.zeros(len(inputs + extra_in + inputs))
    else:
        mean_in = np.zeros(len(inputs + extra_in))
        std_in = np.zeros(len(inputs + extra_in))

    mean_out = np.zeros(len(outputs))
    std_out = np.zeros(len(outputs))

    for i, j in zip(range(len(inputs + extra_in)), inputs + extra_in):
        mean_in[i] = mean_dict[j]
        std_in[i] = std_dict[j]
    if Lateral:
        mean_in[-len(inputs) :] = mean_in[: len(inputs)]
        std_in[-len(inputs) :] = std_in[: len(inputs)]

    for i, j in zip(range(len(outputs)), outputs):
        mean_out[i] = mean_dict[j]
        std_out[i] = std_dict[j]

    std_dict = {"s_in": std_in, "s_out": std_out, "m_in": mean_in, "m_out": mean_out}

    return std_dict


def get_train_test_ranges(N_samples, N_val, lag, hist, interval):
    s_train = lag * hist  # 1*0=0
    e_train = s_train + N_samples * interval  # 0 + 4000*1 = 4000
    e_test = e_train + interval * N_val  # 4000 + 1*300 = 4300
    return s_train, e_train, e_test


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
