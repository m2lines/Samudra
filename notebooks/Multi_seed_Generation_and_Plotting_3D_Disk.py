#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[2]:


import sys

sys.path.append("../src/")


# In[3]:


import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from constants import EXTRA_VARS, INPT_VARS, OUT_VARS
from datasets import InferenceDataset, TrainDataset, get_train_test_ranges
from models.rollout import generate_model_rollout
from utils.train import extract_wet_mask


# ## 3D Fields

# ### G3D All

# In[4]:


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
### Convnext unet
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# All Levels - CM4
# with initialize_config_dir(
#     version_base=None,
#     config_dir="/pscratch/sd/s/suryad/Ocean_Emulator/configs",
# ):
#     args = compose(
#         config_name="exp/eval_unet_global_3D_all_CM4",
#         overrides=[
#             "output_dir=./temp/{0}_ConvNextUNetCM4Hist1Epochs70Epoch37".format(
#                 str(datetime.now())[:10]
#             ),
#             "network={0}_ConvNextUNetCM4Hist1Epochs70Epoch37".format(
#                 str(datetime.now())[:10]
#             ),
#             "ckpt_path=['/pscratch/sd/s/suryad/Ocean_Emulator/train_3D/2024-12-12-convnextunet_CM4_hist1_allvars_completerun/batch_size=4,epochs=70,hist=1,rand_seed=15,region=global_3D,scheduler=True,unet.ch_width=[157,200,250,300,400],wandb.mode=online/saved_nets/convnextunet_epoch_37_beststeps_4_global_3D_all_N_train_13800_Lateral_Data_025_no_smooth.pt']",
#             "hist=1",
#             "unet.ch_width=[157,200,250,300,400]",
#             "run_gen_pred=True",
#             "pred_names=null",
#             "pred_paths=null",
#             "+dataset_name=CM4",
#             "train_region=global_3D",
#             "region=global_3D",
#             "model_name_replace=Convnext",
#             "depth_mode=all",
#         ],
#     )


# All Levels - CM4 - SAT TOS
# with initialize_config_dir(
#     version_base=None,
#     config_dir="/pscratch/sd/s/suryad/Ocean_Emulator/configs",
# ):
#     args = compose(
#         config_name="exp/eval_unet_global_3D_all_CM4",
#         overrides=[
#             "output_dir=./temp/{0}_ConvNextUNetCM4Hist0_SAT_TOS_Epochs70Epoch55".format(
#                 str(datetime.now())[:10]
#             ),
#             "network={0}_ConvNextUNetCM4Hist0_SAT_TOS_Epochs70Epoch55".format(
#                 str(datetime.now())[:10]
#             ),
#             "ckpt_path=['/pscratch/sd/s/suryad/Ocean_Emulator/train_3D/2024-12-20-convnextunet_CM4_hist0_with_SAT_tos/batch_size=4,exp_num_extra=3D_all_SAT_tos,exp_num_in=3D_noFast_all,exp_num_out=3D_noFast_all,hist=0,rand_seed=15,region=global_3D,scheduler=True,unet.ch_width=[157,200,250,300,400],wandb.mode=online/saved_nets/convnextunet_epoch_55_beststeps_4_global_3D_all_N_train_13800_Lateral_Data_025_no_smooth.pt']",
#             "hist=0",
#             "unet.ch_width=[157,200,250,300,400]",
#             "run_gen_pred=True",
#             "exp_num_in=3D_noFast_all",
#             "exp_num_out=3D_noFast_all",
#             "exp_num_extra=3D_all_SAT_tos",
#             "pred_names=null",
#             "pred_paths=null",
#             "+dataset_name=CM4",
#             "train_region=global_3D",
#             "region=global_3D",
#             "model_name_replace=Convnext",
#             "depth_mode=all",
#         ],
#     )


# All Levels - CM4 No Fast ins/outs
with initialize_config_dir(
    version_base=None,
    config_dir="/pscratch/sd/s/suryad/Ocean_Emulator/configs",
):
    args = compose(
        config_name="exp/eval_unet_global_3D_all_CM4",
        overrides=[
            "output_dir=./temp/{0}_ConvNextUNetCM4Hist1NofastinoutEpochs70Epoch55_6000_0_2000".format(
                str(datetime.now())[:10]
            ),
            "network={0}_ConvNextUNetCM4Hist1NofastinoutEpochs70Epoch55_6000_0_2000".format(
                str(datetime.now())[:10]
            ),
            "ckpt_path=['/pscratch/sd/s/suryad/Ocean_Emulator/train_3D/2024-11-19-convnextunet_CM4_hist1/batch_size=4,epochs=70,exp_num_in=3D_noFast_all,exp_num_out=3D_noFast_all,hist=1,rand_seed=15,region=global_3D,scheduler=True,unet.ch_width=[157,200,250,300,400],wandb.mode=online/saved_nets/convnextunet_epoch_55_steps_4_global_3D_all_N_train_13800_Lateral_Data_025_no_smooth.pt']",
            "hist=1",
            "unet.ch_width=[157,200,250,300,400]",
            "run_gen_pred=True",
            "pred_names=null",
            "pred_paths=null",
            "exp_num_in=3D_noFast_all",
            "exp_num_out=3D_noFast_all",
            "+dataset_name=CM4",
            "train_region=global_3D",
            "region=global_3D",
            "model_name_replace=Convnext",
            "depth_mode=all",
            "N_samples=6000",
            "N_val=0",
            "N_test=600",
        ],
    )


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
### SWIN
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# All Levels - CM4 No Fast ins/outs
# with initialize_config_dir(
#     version_base=None,
#     config_dir="/pscratch/sd/s/suryad/Ocean_Emulator/configs",
# ):
#     args = compose(
#         config_name="exp/eval_swin_global_3D_all_CM4",
#         overrides=[
#             "output_dir=./temp/{0}_SwinCM4Hist1NofastinoutEpochs70Epoch67".format(
#                 str(datetime.now())[:10]
#             ),
#             "network={0}_SwinCM4Hist1NofastinoutEpochs70Epoch67".format(
#                 str(datetime.now())[:10]
#             ),
#             "ckpt_path=['/pscratch/sd/s/suryad/Ocean_Emulator/train_3D/2024-11-23-swinv1_CM4_hist1_nofast/batch_size=4,epochs=70,exp_num_in=3D_noFast_all,exp_num_out=3D_noFast_all,hist=1,rand_seed=15,region=global_3D,scheduler=True,wandb.mode=online/saved_nets/swin_epoch_67_beststeps_4_global_3D_all_N_train_13800_Lateral_Data_025_no_smooth.pt']",
#             "hist=1",
#             "run_gen_pred=True",
#             "pred_names=null",
#             "pred_paths=null",
#             "exp_num_in=3D_noFast_all",
#             "exp_num_out=3D_noFast_all",
#             "+dataset_name=CM4",
#             "train_region=global_3D",
#             "region=global_3D",
#             "model_name_replace=Swin",
#             "depth_mode=all",
#         ],
#     )

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

print("Done")


# ## Init

# In[5]:


inputs_str = INPT_VARS[args.training.exp_num_in]
extra_in_str = EXTRA_VARS[args.training.exp_num_extra]
outputs_str = OUT_VARS[args.training.exp_num_out]
levels = args.training.exp_num_in.split("_")[-1]
if "all" in levels:
    levels = 19
elif "2D" in levels:
    levels = 1
else:
    levels = int(levels)

str_in = "".join([i + "_" for i in inputs_str])
str_ext = "".join([i + "_" for i in extra_in_str])
str_out = "".join([i + "_" for i in outputs_str])

print("inputs: " + str_in)
print("extra inputs: " + str_ext)
print("outputs: " + str_out)
print("levels: " + str(levels))

N_atm = len(extra_in_str)  # Number of atmosphere variables
N_in = len(inputs_str)
if args.training.lateral:
    N_extra = (
        N_atm + N_in
    )  # Number of atmosphere variables + Lateral boundary variables
else:
    N_extra = N_atm  # Number of atmosphere variables
N_out = len(outputs_str)

num_in = int((args.data.hist + 1) * N_in + N_extra)
num_out = int((args.data.hist + 1) * len(outputs_str))

print("Number of inputs: ", num_in)  # 3 (ocean speeds + ocean temp)(t) +
# 3 (atm wind stresses + atm temp)(t) +
# 3 (boundary ocean speeds + boundary ocean temp)(t) -> 3 (ocean speeds + ocean temp)(t+1)
print("Number of outputs: ", num_out)  # 3

if "swin" in args.training.network.lower():
    lat = torch.tensor(data.lat.values).to(dtype=torch.float32).unsqueeze(0).cuda()
    lon = torch.tensor(data.lon.values).to(dtype=torch.float32).unsqueeze(0).cuda()
    mask = ~torch.tensor(data.wetmask.isel(lev=0).values).cuda()

    model = instantiate(
        args.swin,
        in_channels=num_in,
        output_channels=num_out,
        wet=wet.cuda(),
        hist=args.data.hist,
        lat=lat,
        lon=lon,
        land_mask=mask,
    )
elif "convnext" in args.training.network.lower():
    if args.unet.ch_width[0] != num_in:
        print(
            "Changing ch_width to match number of inputs {0} -> {1}".format(
                args.unet.ch_width[0], num_in
            )
        )
        args.unet.ch_width[0] = num_in

# Post-fix strings
str_train = (
    "steps_"
    + str(args.data.steps)
    + "_"
    + args.train_region
    + "_"
    + args.data.depth_mode
    + "_N_train_4000"
    + "_Lateral_Data_025_no_smooth"
)
str_save = (
    "steps_"
    + str(args.data.steps)
    + "_"
    + args.train_region
    + "_"
    + args.data.region
    + "_"
    + args.data.depth_mode
    + "+N_samples_"
    + str(args.data.N_samples)
)
post_model_name = (
    "Train_"
    + args.train_region
    + "_Test_"
    + args.data.region
    + "_"
    + args.data.depth_mode
    + "_N_train_"
    + str(args.data.N_samples)
    + "_Lateral_Data_025_no_smooth"
)
post_pred_name = (
    args.data.region
    + "_"
    + args.data.depth_mode
    + "_N_samples_"
    + str(args.data.N_samples)
)

# Getting start and end indices of train and test
s_train, e_train, e_test = get_train_test_ranges(
    args.data.N_samples,
    args.data.N_val,
    args.data.lag,
    args.data.hist,
    args.data.interval,
)
dataset_name = args.dataset_name

if "OM4" in dataset_name or "CM4" in dataset_name:
    timestep_str = "\\times 5"
else:
    raise ValueError("Dataset not recognized")


# In[6]:


print("Calculating mask tensors")

wet_zarr = xr.open_zarr(os.path.join("/pscratch/sd/s/suryad/data", args.data.wet_file))
wet = extract_wet_mask(wet_zarr, outputs_str, args.data.hist)
print("Wet resolution:", wet.shape)
# time_vec = inputs[0].time.data
# time_test = time_vec[e_test : (e_test + args.data.lag * args.data.N_test)]


# In[7]:


import os

import torch
import torch.utils.data as data
import xarray as xr
from einops import rearrange


class InferenceDataset(torch.utils.data.Dataset):
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
        hist,
        ind_start,
        long_rollout,
        device="cuda",
    ):
        super().__init__()
        self.device = device

        self.size = n_samples
        self.lag = lag
        self.interval = interval
        self.hist = hist
        self.ind_start = ind_start

        assert self.interval == 1
        assert self.lag == 1

        data = data.isel(time=slice(self.ind_start, None))
        self.inputs = data[inputs_str + extra_in_str]
        self.outputs = data[outputs_str]
        self.inputs_no_extra = data[inputs_str]
        self.extras = data[extra_in_str]

        # This class will be used only for validation and rollouts
        # Rolling indices to keep track of histories/past states:
        # HIST=0 ; 0->[0, 1]; 1->[1, 2]; 2->[2, 3]; 3->[3, 4]
        # HIST=1 ; 0->[[0, 1], [2, 3]]; 1->[[2, 3], [4, 5]]; 2->[[4, 5], [6, 7]]; 3->[[6, 7], [8, 9]]
        # HIST=2 ; 0->[[0, 1, 2], [3, 4, 5]]; 1->[[3, 4, 5], [6, 7, 8]]; 2->[[6, 7, 8], [9, 10, 11]]; 3->[[9, 10, 11], [12, 13, 14]]
        indices = xr.DataArray(
            np.arange(data.time.size),
            dims=["time"],
            coords={"time": data.time},
        )
        total_steps = 2 * self.hist + 1
        rolling_indices = (
            indices.rolling(time=len(data.time) - total_steps, center=False)
            .construct("window_dim")
            .astype(int)
        )
        rolling_indices = rolling_indices.transpose("window_dim", "time").isel(
            time=slice(len(data.time) - total_steps - 1, None)
        )  # Remove first few null indices
        self.rolling_indices = rolling_indices.isel(
            window_dim=slice(0, None, self.hist + 1)
        )  # Skip indices based on history

        if long_rollout:
            window0 = self.rolling_indices.isel(window_dim=0)
            print(
                "Long rollout will begin with input and produce output from time index {0} and {1} respectively".format(
                    window0.isel(time=0).values + ind_start,
                    window0.isel(time=self.hist + 1).values + ind_start,
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
        print(
            "Out: ",
            (self.ind_start + x_index.isel(time=slice(self.hist + 1, None))).values,
            end=" ",
        )
        data_in = self.inputs_no_extra.isel(time=x_index).isel(
            time=slice(None, self.hist + 1)
        )
        data_in = (
            (data_in - self.inputs_no_extra_mean) / self.inputs_no_extra_std
        ).fillna(0)
        data_in = (
            data_in.to_array()
            .transpose("window_dim", "time", "variable", "y", "x")
            .to_numpy()
        )
        data_in = rearrange(
            data_in, "window_dim time variable y x -> window_dim (time variable) y x"
        )
        if len(self.extras.variables) != 0:
            data_in_boundary = self.extras.isel(time=x_index).isel(time=self.hist)
            data_in_boundary = (
                (data_in_boundary - self.extras_mean) / self.extras_std
            ).fillna(0)
            data_in_boundary = (
                data_in_boundary.to_array()
                .transpose("window_dim", "variable", "y", "x")
                .to_numpy()
            )
            data_in = np.concatenate((data_in, data_in_boundary), axis=1)

        label = self.outputs.isel(time=x_index).isel(time=slice(self.hist + 1, None))
        label = ((label - self.out_mean) / self.out_std).fillna(0)
        label = (
            label.to_array()
            .transpose("window_dim", "time", "variable", "y", "x")
            .to_numpy()
        )
        label = rearrange(
            label, "window_dim time variable y x -> window_dim (time variable) y x"
        )

        items = (torch.from_numpy(data_in).float(), torch.from_numpy(label).float())

        return items


# In[8]:


import xarray as xr

assert args.data.depth_mode == "surface" or args.data.depth_mode == "all"

if args.data.depth_mode == "surface":
    wet = torch.load(os.path.join("/pscratch/sd/s/suryad/data", args.surface_wet_file))
    wet_bool = np.array(wet.cpu()).astype(bool)
    data = xr.open_zarr(os.path.join("/pscratch/sd/s/suryad/data", args.data.data_zarr))
    data_mean = xr.open_zarr(
        os.path.join("/pscratch/sd/s/suryad/data", args.data.data_means_zarr)
    )
    data_std = xr.open_zarr(
        os.path.join("/pscratch/sd/s/suryad/data", args.data.data_stds_zarr)
    )
elif args.data.depth_mode == "all":
    data = xr.open_zarr(os.path.join("/pscratch/sd/s/suryad/data", args.data.data_zarr))
    data_mean = xr.open_zarr(
        os.path.join("/pscratch/sd/s/suryad/data", args.data.data_means_zarr)
    )
    data_std = xr.open_zarr(
        os.path.join("/pscratch/sd/s/suryad/data", args.data.data_stds_zarr)
    )

    ### Smoothening
    if args.data.smooth:
        for var in outputs_str:
            if "uo" in var or "vo" in var:
                window = window_size
                print(f"Smoothing {var} with window size {window}")
                data[var] = (
                    data[var]
                    .rolling(time=window, min_periods=1, center=False)
                    .mean()
                    .compute()
                )


# In[9]:


# # Dont change this
# target_start_date = '2014-09-30'
# target_end_date = '2022-12-14'

# copy_start_date = '2014-09-30'
# copy_end_date = '2022-12-14'

# # copy_start_date = '2005-09-30'
# # copy_end_date = '2013-12-14'

# # copy_start_date = '1995-09-30'
# # copy_end_date = '2003-12-14'

# for var in ['hfds']:
#     old_hfds_anoms = data[var].isel(time=slice(e_test, e_test + args.data.N_test)).copy()
#     assert ((old_hfds_anoms.fillna(0).values == data[var].isel(time=slice(e_test, e_test + args.data.N_test)).fillna(0).values).all())
#     period_data = data[var].sel(time=slice(copy_start_date, copy_end_date))
#     data[var].loc[dict(time=slice(target_start_date, target_end_date))] = period_data + np.reshape(np.arange(period_data.time.size)* (0.0136986301),(-1,1,1))
#     assert not ((old_hfds_anoms.fillna(0).values == data[var].isel(time=slice(e_test, e_test + args.data.N_test)).fillna(0).values).all())
#     if copy_start_date != target_start_date:
#         # assert ((data[var].sel(time=slice(copy_start_date, copy_end_date)).fillna(0).values * 1.5 == data[var].isel(time=slice(e_test, e_test + args.data.N_test)).fillna(0).values).all())
#         pass


# In[10]:


# import matplotlib.pyplot as plt
# for var in ['hfds']:
#     # Plotting
#     dim = 'time'
#     title = 'hfds'
#     hfds_timeseries = (old_hfds_anoms*data['areacello']).mean(['x', 'y'])
#     # Yearly average
#     # hfds_timeseries = hfds_timeseries.groupby('time.year').mean('time')
#     # dim = 'year'
#     # title = f'{var} yearly averaged'

#     hfds_timeseries.plot(c='k', label='No trend added')
#     poly_coeffs = hfds_timeseries.polyfit(dim=dim, deg=1)
#     trend = xr.polyval(hfds_timeseries[dim], poly_coeffs.polyfit_coefficients).compute()
#     trend.plot(c='k')

#     hfds_timeseries_1wm2 = (data[var].isel(time=slice(e_test, e_test + args.data.N_test))*data['areacello']).mean(['x', 'y'])

#     # Yearly average
#     # hfds_timeseries_1wm2 = hfds_timeseries_1wm2.groupby('time.year').mean('time')
#     # dim = 'year'
#     # title = f'{var} yearly averaged'

#     hfds_timeseries_1wm2.plot(c='b', label='trend added')
#     poly_coeffs = hfds_timeseries_1wm2.polyfit(dim=dim, deg=1)
#     trend = xr.polyval(hfds_timeseries_1wm2[dim], poly_coeffs.polyfit_coefficients).compute()
#     trend.plot(c='b')
#     plt.legend()
#     plt.ylabel(title)


# In[11]:


# total_heat_flux = (old_hfds_anoms*data['areacello']).sum(['x','y'])
# heat_added = np.trapz(total_heat_flux.values)
# print(heat_added)
# total_heat_flux = (data[var].isel(time=slice(e_test, e_test + args.data.N_test))*data['areacello']).sum(['x','y'])
# heat_added_new = np.trapz(total_heat_flux.values)
# print(heat_added_new)
# print("Additional Heat added: {:.2e}".format(heat_added_new - heat_added))


# In[12]:


train_data = TrainDataset(
    data,
    inputs_str,
    extra_in_str,
    outputs_str,
    wet,
    data_mean,
    data_std,
    args.data.N_samples,
    args.data.lag,
    args.data.interval,
    args.data.hist,
    args.data.steps,
    device="cuda",
)

test_data = InferenceDataset(
    data,
    inputs_str,
    extra_in_str,
    outputs_str,
    wet,
    data_mean,
    data_std,
    args.data.N_test,
    args.data.lag,
    args.data.interval,
    args.data.hist,
    e_test,
    long_rollout=True,
    device="cuda",
)
# test_data[0]


# In[19]:


print(len(test_data[0:3]))
print(test_data[0:3][0].shape)


# In[ ]:


# In[12]:


# Model
print("Loading model " + args.training.network)
if "swin" in args.training.network.lower():
    lat = torch.tensor(data.lat.values).to(dtype=torch.float32).unsqueeze(0).cuda()
    lon = torch.tensor(data.lon.values).to(dtype=torch.float32).unsqueeze(0).cuda()
    mask = ~torch.tensor(data.wetmask.isel(lev=0).values).cuda()

    model = instantiate(
        args.swin,
        in_channels=num_in,
        output_channels=num_out,
        wet=wet.cuda(),
        hist=args.data.hist,
        lat=lat,
        lon=lon,
        land_mask=mask,
    )
elif "unet" in args.training.network.lower():
    model = instantiate(args.unet, n_out=num_out, wet=wet.cuda(), hist=args.data.hist)

full_model_path = args.ckpt_path
full_model_name = args.training.network + "_" + post_model_name
output_channels = model.num_output_channels

model = model.to(args.training.device)
ckpt_path = args.ckpt_path
model = model


# In[13]:


# Stats
mean_in = test_data.in_mean.to_array().to_numpy().reshape(-1)
std_in = test_data.in_std.to_array().to_numpy().reshape(-1)
mean_out = test_data.out_mean.to_array().to_numpy().reshape(-1)
std_out = test_data.out_std.to_array().to_numpy().reshape(-1)

test_data.norm_vals = {
    "s_out": std_out,
    "s_in": std_in,
    "m_out": mean_out,
    "m_in": mean_in,
}


# In[14]:


# clim
# clim = None
# if args.save_clim_data:
#     print("Saving clim")
#     clim = np.zeros((total_steps, *wet.shape, N_out))
#     for i in range(N_out):
#         clim[:, :, :, i] = (
#             outputs[i].groupby("time.dayofyear").mean("time").data
#         )
#     torch.save(
#         clim,
#         Path(args.data_dir) / "clim_cnn_{0}.pt".format(str_save),
#     )

# else:
#     print("Loading clim")
#     clim = torch.load(
#         Path(args.data_dir) / "clim_cnn_{0}.pt".format(str_save)
#     )


# Getting area tensor
print("Computing area tensor")
# grids = xr.open_dataset(
#     os.path.join("/pscratch/sd/s/suryad/data", args.data.grid_file)
# ).rename({"dx": "dxu", "dy": "dyu"})

# area = torch.from_numpy(grids["area_C"].to_numpy()).to(device="cpu")
# dx = grids["dxu"].to_numpy()
# dy = grids["dyu"].to_numpy()

grids = xr.open_dataset(
    os.path.join("/pscratch/sd/s/suryad/data", args.data.grid_file)
).rename({"xu_ocean": "x", "yu_ocean": "y"})
area = torch.from_numpy(grids["area_C"].to_numpy()).to(device="cpu")


pred_model_path = Path("/pscratch/sd/s/suryad/Ocean_Emulator/Preds") / full_model_name
if not os.path.isdir(pred_model_path):
    os.makedirs(pred_model_path)

Nb = args.training.Nb
hist = args.data.hist
lag = args.data.lag
N_test = args.data.N_test
N_samples = args.data.N_samples
output_dir = args.output_dir
region = args.data.region
steps = args.data.steps
network = args.model_name_replace

pred_region = args.data.region
pred_names = args.pred_names if args.pred_names else []
pred_paths = args.pred_paths if args.pred_paths else []

JUPYTER_MODE = False


# ## Generation

# In[15]:


def generate_model_rollout(
    N_eval,
    test_data,
    model,
    hist,
    N_out,
    N_extra,
    initial_input=None,
    Nb=0,
    region="global",
    train=False,
):
    N_test = test_data.size

    model.eval()
    model_pred = np.zeros((N_eval, *test_data[0][0].shape[2:], N_out))

    with torch.no_grad():
        outs = model.inference(test_data, initial_input, num_steps=N_eval // (hist + 1))
    for i in range(N_eval // (hist + 1)):
        pred_temp = outs[i]
        pred_temp = torch.nan_to_num(pred_temp)
        pred_temp = torch.clip(pred_temp, min=-1e5, max=1e5)
        C, H, W = pred_temp.shape
        pred_temp = torch.reshape(pred_temp, (hist + 1, C // (hist + 1), H, W))
        model_pred[i * (hist + 1) : (i + 1) * (hist + 1)] = torch.swapaxes(
            torch.swapaxes(pred_temp, 3, 1), 2, 1
        ).cpu()

    if train:
        return model_pred
    else:
        return (
            model_pred * test_data.norm_vals["s_out"] + test_data.norm_vals["m_out"],
            outs,
        )


# In[16]:


def generate_pred_lateral():
    print("Generation Pred begin...")
    for rand_ind, model_path in enumerate(args.ckpt_path):
        print("Random seed: ", rand_ind + 1)
        print(
            "Saving to : ",
            pred_model_path
            / (
                "Pred_lateral_Fast_Data_025_"
                + post_pred_name
                + "_rand_seed_"
                + str(rand_ind + 1)
                + ".zarr"
            ),
        )
        # try:
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cuda"))["model"]
        )
        # except:
        #     model.load_state_dict(
        #         torch.load(model_path, map_location=torch.device("cuda"))
        #     )

        model_pred, _ = generate_model_rollout(
            N_test,
            test_data,
            model,
            hist,
            N_out,
            N_extra,
            initial_input=None,
            Nb=0,
            region=region,
        )

        print("data_gen")
        da = xr.DataArray(data=model_pred, dims=["time", "x", "y", "var"])
        ds = da.to_dataset(name="predictions")
        ds.attrs["model_path"] = str(model_path)
        ds.attrs["start_step"] = e_test + hist + 1
        ds.attrs["end_step"] = e_test + hist + 1 + N_test
        ds.to_zarr(
            pred_model_path
            / (
                "Pred_lateral_Fast_Data_025_"
                + post_pred_name
                + "_rand_seed_"
                + str(rand_ind + 1)
                + ".zarr"
            ),
            mode="w",
        )
        print(f"Model pred shape {model_pred.shape}")


# In[17]:


get_ipython().run_cell_magic(
    "time", "", "if args.run_gen_pred:\n    generate_pred_lateral()\n"
)


# In[ ]:
