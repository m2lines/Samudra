import sys

sys.path.append("../src/")

from constants import INPT_VARS, EXTRA_VARS, OUT_VARS
from hydra.utils import instantiate
from pathlib import Path
import os
from utils.train_utils import extract_wet
from utils.data_utils import (
    get_train_test_ranges,
    data_CNN_Disk,
    data_CNN_Disk_steps,
)

import numpy as np
import torch
import xarray as xr

from hydra import compose, initialize_config_dir
from datetime import datetime
import os


########################################################

# All Levels - v0.2.1, Hist = 1, hfds_anoms, No Fast ins/outs
with initialize_config_dir(
    version_base=None,
    config_dir="/pscratch/sd/s/suryad/Ocean_Emulator/configs",
):
    args = compose(
        config_name="exp/eval_unet_global_3D_all_hfds_anoms",
        overrides=[
            "output_dir=./temp/{0}_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNofastinoutEpochs70Epoch55_4144".format(
                str(datetime.now())[:10]
            ),
            "network={0}_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNofastinoutEpochs70Epoch55_4144".format(
                str(datetime.now())[:10]
            ),
            "ckpt_path=[/pscratch/sd/s/suryad/Ocean_Emulator/train_3D/from_greene/nofastvars_all/convnextunet_epoch_55_steps_4_global_3D_all_N_train_4000_Lateral_Data_025_no_smooth.pt]",
            "hist=1",
            "unet.ch_width=[157,200,250,300,400]",
            "run_gen_pred=True",
            "pred_names=null",
            "pred_paths=null",
            "N_samples=4000",
            "N_val=141",
            "N_test=600",
            "exp_num_in=3D_noFast_all",
            "exp_num_out=3D_noFast_all",
            "+dataset_name=OM4",
            "train_region=global_3D",
            "region=global_3D",
            "model_name_replace=Convnext",
            "depth_mode=all",
        ],
    )

# All Levels - v0.2.1, Hist = 1, hfds_cuminteg, No Fast ins/outs
# with initialize_config_dir(
#     version_base=None,
#     config_dir="/pscratch/sd/s/suryad/Ocean_Emulator/configs",
# ):
#     args = compose(
#         config_name="exp/eval_unet_global_3D_all_hfds_cuminteg_1975",
#         overrides=[
#             "output_dir=./temp/{0}_ConvNextUNetTrain3Dv021Eval3DhfdscumintegNsamples1850NofastinoutEpochs70Epoch55".format(
#                 str(datetime.now())[:10]
#             ),
#             "network={0}_ConvNextUNetTrain3Dv021Eval3DhfdscumintegNsamples1850NofastinoutEpochs70Epoch55".format(
#                 str(datetime.now())[:10]
#             ),
#             "ckpt_path=['/pscratch/sd/s/suryad/Ocean_Emulator/train_3D/2024-10-23-convnextunet_v021_hist1_1975_onlyseasonalizedcuminteg/batch_size=4,epochs=70,exp_num_in=3D_noFast_all,exp_num_out=3D_noFast_all,hist=1,rand_seed=15,region=global_3D,scheduler=True,unet.ch_width=[157,200,250,300,400],wandb.mode=online/saved_nets/convnextunet_epoch_55_beststeps_4_global_3D_all_N_train_2850_Lateral_Data_025_no_smooth.pt']",
#             "hist=1",
#             "unet.ch_width=[157,200,250,300,400]",
#             "run_gen_pred=True",
#             "pred_names=null",
#             "pred_paths=null",
#             "N_samples=1850",
#             "N_val=50",
#             "N_test=600",
#             "exp_num_in=3D_noFast_all",
#             "exp_num_out=3D_noFast_all",
#             "+dataset_name=OM4",
#             "train_region=global_3D",
#             "region=global_3D",
#             "model_name_replace=Convnext",
#             "depth_mode=all",
#         ],
#     )

suffix = "hfds_anoms"
print("Done")

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
    model = instantiate(
        args.swin,
        in_channels=num_in,
        output_channels=num_out,
        pretrain_img_size=[180, 360],
        wet=wet.cuda(),
        hist=args.data.hist,
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
    args.data.region + "_" + args.data.depth_mode + "_N_samples_" + str(args.data.N_samples)
)

# Getting start and end indices of train and test
s_train, e_train, e_test = get_train_test_ranges(
    args.data.N_samples, args.data.N_val, args.data.lag, args.data.hist, args.data.interval
)
dataset_name = args.dataset_name

if "OM4" in dataset_name:
    timestep_str = "\\times 5"
else:
    raise ValueError("Dataset not recognized")


print("Calculating mask tensors")

# ds = xr.open_zarr(os.path.join("/pscratch/sd/s/suryad/data", args.raw_data_zarr))
# ds = ds.drop(["tauuo", "tauvo", "hfds"])
# lev_map = {str(lev).replace('.', '_'): i for i, lev in enumerate(ds["lev"].values)}


# def get_wet_mask(inputs, device="cpu"):
#     wet = xr.zeros_like(inputs[0][0])
#     # inputs[0][0,12,12] = np.nan
#     for ds in inputs:
#         wet += np.isnan(ds[0])

#     wet_nan = xr.where(wet != 0, np.nan, 1).to_numpy()
#     wet = np.isnan(xr.where(wet == 0, np.nan, 0))
#     wet = np.nan_to_num(wet.to_numpy())
#     wet = torch.from_numpy(wet).type(torch.float32).to(device=device)
#     return wet, wet_nan


# if args.data.depth_mode == "surface":
#     inputs, extra_in, outputs = gen_3D_data(
#         os.path.join("/pscratch/sd/s/suryad/data", args.raw_data_zarr),
#         inputs_str,
#         extra_in_str,
#         outputs_str,
#         args.data.lag,
#         depth_mode=args.data.depth_mode,
#     )
#     wet, wet_nan = get_wet_mask(inputs, "cpu")
#     wet_bool = np.array(wet.cpu()).astype(bool)
#     wet_lap = compute_laplacian_wet(wet_nan, 4)  # hardcoded
#     wet_lap = xr.where(wet_lap == 0, 1, np.nan)
#     wet_lap = np.nan_to_num(wet_lap)


# elif args.data.depth_mode == "all":
#     wet_stacked = []
#     wet_nan_stacked = []
#     for i, lev in enumerate(ds["lev"].values):
#         inputs = []
#         inputs.append(ds["uo"].sel(lev=lev))
#         inputs.append(ds["vo"].sel(lev=lev))
#         inputs.append(ds["thetao"].sel(lev=lev))
#         inputs.append(ds["so"].sel(lev=lev))
#         if i == 0:
#             inputs.append(ds["zos"])

#         inputs = tuple(inputs)
#         wet, wet_nan = get_wet_mask(inputs)
#         wet_stacked.append(wet)
#         wet_nan_stacked.append(wet_nan)

#     wet_3D = torch.stack(wet_stacked)
#     wet_nan_3D = np.stack(wet_nan_stacked)

#     final_wet = []
#     final_wet_nan = []
#     for var in inputs_str:
#         try:
#             level = lev_map[var.split("lev_")[-1]]
#         except:
#             print("Exception at : ", var)
#             level = 0
#         final_wet.append(wet_3D[level])
#         final_wet_nan.append(wet_nan_3D[level])

#     wet = torch.stack(final_wet)
#     wet_nan = np.stack(final_wet_nan)

#     wet_bool = np.array(wet.cpu()).astype(bool)
#     # wet_lap = compute_laplacian_wet(wet_nan, 4)  # hardcoded
#     # wet_lap = xr.where(wet_lap == 0, 1, np.nan)
#     # wet_lap = np.nan_to_num(wet_lap)

# wet = torch.concat([wet] * (args.data.hist + 1), dim=0)

wet_zarr = xr.open_zarr(os.path.join("/pscratch/sd/s/suryad/data", args.data.wet_file))
wet = extract_wet(wet_zarr, outputs_str, args.data.hist)
print("Wet resolution:", wet.shape)
# time_vec = inputs[0].time.data
# time_test = time_vec[e_test : (e_test + args.data.lag * args.data.N_test)]


import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.ndimage import gaussian_filter
from einops import rearrange
import os


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


train_data = data_CNN_Disk_steps(
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
    1,
    device="cuda",
)

test_data = data_CNN_Disk(
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

# Model
print("Loading model " + args.training.network)
if "swin" in args.training.network.lower():
    model = instantiate(
        args.swin,
        in_channels=num_in,
        output_channels=num_out,
        pretrain_img_size=[180, 360],
        wet=wet.cuda(),
        hist=args.data.hist,
    )
elif "unet" in args.training.network.lower():
    model = instantiate(args.unet, n_out=num_out, wet=wet.cuda(), hist=args.data.hist)

full_model_path = args.ckpt_path
full_model_name = args.training.network + "_" + post_model_name
output_channels = model.output_channels

model = model.to(args.training.device)
ckpt_path = args.ckpt_path
model = model

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


def send_data_to_cpu():
    test_data.set_device(device="cpu")


past_inputs_str = ["past_" + s for s in inputs_str]
current_inputs_str = ["cur_" + s for s in inputs_str]
extra_in_str
all_str = past_inputs_str + current_inputs_str + extra_in_str
all_outputs_str = past_inputs_str + current_inputs_str

import torch
import shap
import numpy as np
import matplotlib.pyplot as plt

# Clear cache
torch.cuda.empty_cache()

device = "cuda"

model = model.to(device=device)
model.wet = model.wet.to(device=device)

import torch
import torch.nn as nn


class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        # Forward pass through the original model
        # Adjust the model call and output processing as needed
        # For example, if your model expects a list, pass [x]
        self.model.eval()
        predictions = self.model.forward_once(x)

        total_prediction = predictions.sum(dim=(2, 3))  # Shape: [batch_size]

        return total_prediction


wrapped_model = WrappedModel(model)


# Prepare input tensor
input_state = train_data[0]
input_tensor = input_state[0].unsqueeze(0).to(device=device)  # Shape: [1, 81, H, W]
print(f"Input tensor shape: {input_tensor.shape}")


def generate_background_tensor(num_samples=8):
    """Generates a background tensor with the specified number of samples."""
    background_samples = []
    l = len(train_data)
    for i in range(num_samples):
        idx = np.random.randint(0, l)
        data = train_data[idx][0].unsqueeze(0).to(device=device)
        background_samples.append(data)
    background_tensor = torch.cat(background_samples, dim=0)
    return background_tensor


def compute_shap_values(input_tensor, background_tensor):
    """Computes SHAP values for the input tensor using the given background tensor."""
    explainer = shap.DeepExplainer(wrapped_model, background_tensor)
    shap_values = explainer.shap_values(input_tensor, check_additivity=False)
    return shap_values


# Parameters
num_iterations = 20  # Number of iterations to average the SHAP values
num_samples_per_iteration = 8  # Background tensor size for each iteration

# Initialize storage for aggregated SHAP values
shap_values_sum = None
import time

start = time.time()
for iteration in range(num_iterations):
    print(
        f"Iteration {iteration + 1}/{num_iterations}: Generating background and computing SHAP values..."
    )

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Generate background tensor
    background_tensor = generate_background_tensor(num_samples_per_iteration)

    # Compute SHAP values
    shap_values = compute_shap_values(input_tensor, background_tensor)

    # Accumulate the SHAP values for averaging
    if shap_values_sum is None:
        shap_values_sum = shap_values
    else:
        shap_values_sum += shap_values

# Average the SHAP values over all iterations
shap_values = shap_values_sum / num_iterations

print(f"Time taken: {time.time() - start:.2f} seconds")
############################################################
# Matplotlib
############################################################

import numpy as np
import matplotlib.pyplot as plt

# Assuming 'shap_values' is the SHAP values array with shape (1, 82, 180, 360, 78)

# Remove the batch dimension
shap_values_array = shap_values[0]  # Shape: (82, 180, 360, 78)

# Take the absolute value
shap_values_abs = np.abs(shap_values_array)

# Aggregate over spatial dimensions (height and width)
shap_values_sum_spatial = shap_values_abs.sum(axis=(1, 2))  # Shape: (82, 78)

# Aggregate over output channels
shap_values_total = shap_values_sum_spatial.sum(axis=1)  # Shape: (82,)

# Rank the input channels
channel_indices = np.arange(
    1, shap_values_total.shape[0] + 1
)  # Channels numbered from 1 to 82
sorted_indices = np.argsort(-shap_values_total)  # Negative sign for descending order
ranked_channels = channel_indices[sorted_indices]
ranked_importance = shap_values_total[sorted_indices]

# Plot the channel importance
labels = [all_str[i - 1] for i in ranked_channels]

# Create positions for the bars
positions = np.arange(len(ranked_channels))

plt.figure(figsize=(max(12, len(labels) * 0.2), 6))
plt.bar(positions, ranked_importance, tick_label=labels)
plt.xlabel("Input Channel")
plt.ylabel("Total SHAP Value")
plt.title("Input Channel Importance Ranking using SHAP")
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig("shap_ranking" + suffix)
plt.close()


import numpy as np
import matplotlib.pyplot as plt

# Remove the batch dimension
shap_values_array = shap_values[0]  # Shape: (82, 180, 360, 78)

# Take the absolute value
shap_values_abs = np.abs(shap_values_array)

# Aggregate over spatial dimensions (height and width)
shap_values_sum_spatial = shap_values_abs.sum(axis=(1, 2))  # Shape: (82, 78)

# Use output variable names from 'outputs_str'
output_var_names = all_outputs_str  # Corrected from 'all_outputs_str'

# Verify lengths
assert (
    len(output_var_names) == shap_values_sum_spatial.shape[1]
), "Mismatch in number of output variables"

# Prepare x-axis positions and labels
num_inputs = shap_values_sum_spatial.shape[0]  # 82 input channels
positions = np.arange(num_inputs)
labels = all_str  # Ensure 'all_str' is defined

# Verify lengths
assert len(labels) == num_inputs, "Mismatch in number of input channels"

# Determine grid size
num_outputs = shap_values_sum_spatial.shape[1]  # 78 output channels
num_cols = 13
num_rows = int(np.ceil(num_outputs / num_cols))

fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 10, num_rows * 8))
axes = axes.flatten()

for output_idx in range(num_outputs):
    ax = axes[output_idx]
    shap_values_per_output = shap_values_sum_spatial[:, output_idx]

    # Plot the SHAP values
    ax.bar(positions, shap_values_per_output, tick_label=labels)
    ax.set_title(output_var_names[output_idx], fontsize=10)
    ax.set_ylabel("SHAP Value", fontsize=8)
    ax.tick_params(axis="y", labelsize=6)

    # Adjust x-axis labels
    # ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)

# Remove any unused subplots
for idx in range(num_outputs, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig("shap_ranking_per_feature" + suffix)
plt.close()
