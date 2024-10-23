import sys

sys.path.append("../src/")

from constants import INPT_VARS, EXTRA_VARS, OUT_VARS
from hydra.utils import instantiate
from pathlib import Path
import os
from matplotlib.animation import FuncAnimation

from utils.train_utils import extract_wet
from utils.data_utils import (
    get_train_test_ranges,
    data_CNN_Disk,
    data_CNN_Disk_steps,
    gen_3D_data,
)
from utils.eval_utils import (
    generate_model_rollout,
    compute_KE,
    compute_nino34,
    compute_amo,
    gen_KE_range,
    gen_value_range,
    compute_mean_single,
)
from utils.climate_utils import compute_laplacian_wet

import numpy as np
import torch
import xarray as xr
import copy

from hydra import compose, initialize_config_dir
import copy
from datetime import datetime
import os

# ########################################################
# 1 2002
# All Levels - Hist = 1, HFDS Anom, 100 year emulation, 1975, NetZeroHf
# with initialize_config_dir(
#     version_base=None,
#     config_dir="/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/configs",
# ):
#     args = compose(
#         config_name="exp/eval_unet_global_3D_all_hfds_anoms",
#         overrides=[
#             "output_dir=./temp/{0}_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHf1975Epochs70Epoch55Years100_10repeat_2002".format(
#                 str("2024-09-25")[:10]
#             ),
#             "network={0}_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHf1975Epochs70Epoch55Years100_10repeat_2002".format(
#                 str("2024-09-25")[:10]
#             ),
#             "ckpt_path=[/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_3D/from_greene/hist1_hfds_anom_1975/convnextunet_epoch_55_steps_4_global_3D_all_N_train_2850_Lateral_Data_025_no_smooth.pt]",
#             "hist=1",
#             "unet.ch_width=[157,200,250,300,400]",
#             "run_gen_pred=True",
#             "N_samples=0",
#             "N_val=0",
#             "N_test=29200",
#             "data_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_100_years_10repeat_2002_netzerohf",
#             "data_means_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_1975_means",
#             "data_stds_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_1975_stds",
#             "pred_names=null",
#             "pred_paths=null",
#             "+dataset_name=OM4_2002_repeat",
#             "+save_factor=200",
#             "train_region=global_3D",
#             "region=global_3D",
#             "model_name_replace=Convnext",
#             "depth_mode=all",
#         ],
#     )

# ########################################################
# 2 2002 Temp Only
# All Levels - Hist = 1, HFDS Anom, 100 year emulation, 1975, TempOnly, NetZeroHf
with initialize_config_dir(
    version_base=None,
    config_dir="/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/configs",
):
    args = compose(
        config_name="exp/eval_unet_global_3D_all_hfds_anoms",
        overrides=[
            "output_dir=./temp/{0}_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHfTempOnly1975Epochs70Epoch55Years100_10repeat_2002".format(
                str("2024-09-25")[:10]
            ),
            "network={0}_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHfTempOnly1975Epochs70Epoch55Years100_10repeat_2002".format(
                str("2024-09-25")[:10]
            ),
            "ckpt_path=[/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_3D/2024-09-09-convnextunet_v021_hist1_hfds_anom_1975_nofast/nofast/saved_nets/convnextunet_epoch_55_beststeps_4_global_3D_all_N_train_2850_Lateral_Data_025_no_smooth.pt]",
            "hist=1",
            "unet.ch_width=[157,200,250,300,400]",
            "run_gen_pred=True",
            "N_samples=0",
            "N_val=0",
            "N_test=29200",
            "data_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_100_years_10repeat_2002_netzerohf",
            "data_means_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_1975_means",
            "data_stds_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_1975_stds",
            "exp_num_in=3D_noFast_all",
            "exp_num_out=3D_noFast_all",
            "pred_names=null",
            "pred_paths=null",
            "+dataset_name=OM4_2002_repeat",
            "+save_factor=200",
            "train_region=global_3D",
            "region=global_3D",
            "model_name_replace=Convnext",
            "depth_mode=all",
        ],
    )
# ########################################################
# 3 1998
# All Levels - Hist = 1, HFDS Anom, 100 year emulation, 1975, NetZeroHf
# with initialize_config_dir(
#     version_base=None,
#     config_dir="/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/configs",
# ):
#     args = compose(
#         config_name="exp/eval_unet_global_3D_all_hfds_anoms",
#         overrides=[
#             "output_dir=./temp/{0}_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHf1975Epochs70Epoch55Years100_10repeat_1998".format(
#                 str("2024-09-12")[:10]
#             ),
#             "network={0}_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHf1975Epochs70Epoch55Years100_10repeat_1998".format(
#                 str("2024-09-12")[:10]
#             ),
#             "ckpt_path=[/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_3D/from_greene/hist1_hfds_anom_1975/convnextunet_epoch_55_steps_4_global_3D_all_N_train_2850_Lateral_Data_025_no_smooth.pt]",
#             "hist=1",
#             "unet.ch_width=[157,200,250,300,400]",
#             "run_gen_pred=True",
#             "N_samples=0",
#             "N_val=0",
#             "N_test=29200",
#             "data_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_100_years_10repeat_1998_netzerohf",
#             "data_means_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_1975_means",
#             "data_stds_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_1975_stds",
#             "pred_names=null",
#             "pred_paths=null",
#             "+dataset_name=OM4_1998_repeat",
#             "+save_factor=200",
#             "train_region=global_3D",
#             "region=global_3D",
#             "model_name_replace=Convnext",
#             "depth_mode=all",
#         ],
#     )

# ########################################################
# 4 1998 Temp Only
# All Levels - Hist = 1, HFDS Anom, 100 year emulation, 1975, TempOnly, NetZeroHf
# with initialize_config_dir(
#     version_base=None,
#     config_dir="/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/configs",
# ):
#     args = compose(
#         config_name="exp/eval_unet_global_3D_all_hfds_anoms",
#         overrides=[
#             "output_dir=./temp/{0}_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHfTempOnly1975Epochs70Epoch55Years100_10repeat_1998".format(
#                 str(datetime.now())[:10]
#             ),
#             "network={0}_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHfTempOnly1975Epochs70Epoch55Years100_10repeat_1998".format(
#                 str(datetime.now())[:10]
#             ),
#             "ckpt_path=[/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/train_3D/2024-09-09-convnextunet_v021_hist1_hfds_anom_1975_nofast/nofast/saved_nets/convnextunet_epoch_55_beststeps_4_global_3D_all_N_train_2850_Lateral_Data_025_no_smooth.pt]",
#             "hist=1",
#             "unet.ch_width=[157,200,250,300,400]",
#             "run_gen_pred=True",
#             "N_samples=0",
#             "N_val=0",
#             "N_test=29200",
#             "data_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_100_years_10repeat_1998_netzerohf",
#             "data_means_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_1975_means",
#             "data_stds_zarr=3D_data_OM4_5daily_v0.2.1_with_hfds_anom_1975_stds",
#             "exp_num_in=3D_noFast_all",
#             "exp_num_out=3D_noFast_all",
#             "pred_names=null",
#             "pred_paths=null",
#             "+dataset_name=OM4_1998_repeat",
#             "+save_factor=200",
#             "train_region=global_3D",
#             "region=global_3D",
#             "model_name_replace=Convnext",
#             "depth_mode=all",
#         ],
#     )

print("Done")

inputs_str = INPT_VARS[args.exp_num_in]
extra_in_str = EXTRA_VARS[args.exp_num_extra]
outputs_str = OUT_VARS[args.exp_num_out]
levels = args.exp_num_in.split("_")[-1]
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
if args.lateral:
    N_extra = (
        N_atm + N_in
    )  # Number of atmosphere variables + Lateral boundary variables
else:
    N_extra = N_atm  # Number of atmosphere variables
N_out = len(outputs_str)

num_in = int((args.hist + 1) * N_in + N_extra)
num_out = int((args.hist + 1) * len(outputs_str))

print("Number of inputs: ", num_in)  # 3 (ocean speeds + ocean temp)(t) +
# 3 (atm wind stresses + atm temp)(t) +
# 3 (boundary ocean speeds + boundary ocean temp)(t) -> 3 (ocean speeds + ocean temp)(t+1)
print("Number of outputs: ", num_out)  # 3

if "swin" in args.network.lower():
    pass
    # model = instantiate(
    #     args.swin,
    #     in_channels=num_in,
    #     output_channels=num_out,
    #     pretrain_img_size=[180, 360],
    #     wet=wet.cuda(),
    #     hist=args.hist,
    # )
elif "convnext" in args.network.lower():
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
    + str(args.steps)
    + "_"
    + args.train_region
    + "_"
    + args.depth_mode
    + "_N_train_4000"
    + "_Lateral_Data_025_no_smooth"
)
str_save = (
    "steps_"
    + str(args.steps)
    + "_"
    + args.train_region
    + "_"
    + args.region
    + "_"
    + args.depth_mode
    + "+N_samples_"
    + str(args.N_samples)
)
post_model_name = (
    "Train_"
    + args.train_region
    + "_Test_"
    + args.region
    + "_"
    + args.depth_mode
    + "_N_train_"
    + str(args.N_samples)
    + "_Lateral_Data_025_no_smooth"
)
post_pred_name = (
    args.region + "_" + args.depth_mode + "_N_samples_" + str(args.N_samples)
)

# Getting start and end indices of train and test
s_train, e_train, e_test = get_train_test_ranges(
    args.N_samples, args.N_val, args.lag, args.hist, args.interval
)
dataset_name = args.dataset_name

if "OM4" in dataset_name:
    timestep_str = "\\times 5"
else:
    raise ValueError("Dataset not recognized")


print("Calculating mask tensors")
wet_zarr = xr.open_zarr(os.path.join("/vast/sd5313/data/m2lines/3D_ocean_data", args.wet_file))
wet = extract_wet(wet_zarr, outputs_str, args.hist)
print("Wet resolution:", wet.shape)
print("e_test: ", e_test)

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
        x_index = xr.Variable(
            ["window_dim", "time"], rolling_idx
        )
        print("Out: ", (self.ind_start + x_index.isel(time=slice(self.hist + 1, None))).values, end=' ')
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

assert args.depth_mode == "surface" or args.depth_mode == "all"

data = xr.open_zarr(os.path.join("/vast/sd5313/data/m2lines/3D_ocean_data", args.data_zarr))
if args.data_zarr== "3D_data_OM4_5daily_v0.2.1_with_hfds_anom_100_years_10repeat_netzerohfds_nojump3xcc":
    print("Updating climate forced runs!")
    data['hfds'] = data['hfds'] + .017462726 
    data['hfds_anomalies'] = data['hfds_anomalies'] + .017462726 
    data['hfds'] = data['hfds'] + np.reshape(np.arange(data.time.size)* (0.0136986301-4.01369026e-04),(-1,1,1))
    data['hfds_anomalies'] = data['hfds_anomalies'] + np.reshape(np.arange(data.time.size)* (0.0136986301-4.01369026e-04),(-1,1,1))

repeats = 4
data = xr.concat([data] * repeats, dim="time")
data['time'] = np.arange(data.time.size)
data


data_mean = xr.open_zarr(
    os.path.join("/vast/sd5313/data/m2lines/3D_ocean_data", args.data_means_zarr)
)
data_std = xr.open_zarr(
    os.path.join("/vast/sd5313/data/m2lines/3D_ocean_data", args.data_stds_zarr)
)
train_data = data_CNN_Disk_steps(
    data,
    inputs_str,
    extra_in_str,
    outputs_str,
    wet,
    data_mean,
    data_std,
    args.N_samples,
    args.lag,
    args.interval,
    args.hist,
    args.steps,
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
    data.time.size,
    args.lag,
    args.interval,
    args.hist,
    e_test,
    long_rollout=True,
    device="cuda",
)
# test_data[0]

# Model
print("Loading model " + args.network)
if "swin" in args.network.lower():
    model = instantiate(
        args.swin,
        in_channels=num_in,
        output_channels=num_out,
        pretrain_img_size=[180, 360],
        wet=wet.cuda(),
        hist=args.hist,
    )
elif "unet" in args.network.lower():
    model = instantiate(args.unet, n_out=num_out, wet=wet.cuda(), hist=args.hist)

full_model_path = args.ckpt_path
full_model_name = args.network + "_" + post_model_name
output_channels = model.output_channels

model = model.to(args.device)
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

# Getting area tensor
print("Computing area tensor")
grids = xr.open_dataset(os.path.join("/vast/sd5313/data/m2lines/3D_ocean_data", args.grid_file)).rename({"xu_ocean": "x", "yu_ocean": "y"})

area = torch.from_numpy(grids["area_C"].to_numpy()).to(device="cpu")
pred_model_path = Path("/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds") / full_model_name
if not os.path.isdir(pred_model_path):
    os.makedirs(pred_model_path)

Nb = args.Nb
hist = args.hist
lag = args.lag
N_test = args.N_test
N_samples = args.N_samples
output_dir = args.output_dir
region = args.region
steps = args.steps
network = args.model_name_replace

pred_region = args.region
pred_names = args.pred_names if args.pred_names else []
pred_paths = args.pred_paths if args.pred_paths else []

JUPYTER_MODE = False
    
    
out_mean = torch.tensor(test_data.out_mean.to_array().to_numpy()).to(device="cuda")
out_std = torch.tensor(test_data.out_std.to_array().to_numpy()).to(device="cuda")

### Generate
def generate_pred_lateral():
    print("Generation Pred begin...")
    for rand_ind, model_path in enumerate(args.ckpt_path):
        print("Random seed: ", rand_ind + 1)
        # try:
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cuda"))["model"]
        )
        # except:
        #     model.load_state_dict(
        #         torch.load(model_path, map_location=torch.device("cuda"))
        #     )
        pred_path = pred_model_path / (
                        "Pred_lateral_Fast_Data_025_"
                        + post_pred_name
                        + "_rand_seed_"
                        + str(rand_ind + 1)
                        + ".zarr"
                    )
        save_factor = args.save_factor
        print(f"Using save_factor {save_factor} to produce {N_test // save_factor} steps each iteration")
        outs = None
        if not os.path.isdir(pred_path):
            start = 0
            print(f"Starting save from {start} for Pred path {pred_path}")
            initial_input = None
        else:
            pred = xr.open_zarr(pred_path)
            start = pred.time.size
            print(f"Restarting save from {start} for Pred path {pred_path}")
            last_pred_numpy = pred.isel(time=slice(-2,None)).to_array().to_numpy().squeeze()
            last_pred = torch.tensor(last_pred_numpy).to(device="cuda")
            assert last_pred.shape[:3] == torch.Size([2, 180, 360])
            last_pred = (last_pred - out_mean) / out_std
            initial_input = torch.swapaxes(torch.swapaxes(last_pred, 3, 2), 2, 1).reshape(-1, 180, 360)
            

        for i in range(start, N_test, N_test // save_factor):
            start_time = time.time()
            test_data = data_CNN_Disk(
                data,
                inputs_str,
                extra_in_str,
                outputs_str,
                wet,
                data_mean,
                data_std,
                args.N_test,
                args.lag,
                args.interval,
                args.hist,
                e_test + i,
                long_rollout=True,
                device="cuda",
            )
            
            test_data.norm_vals = {
                "s_out": std_out,
                "s_in": std_in,
                "m_out": mean_out,
                "m_in": mean_in,
            }
            if outs is not None:
                initial_input = outs[-1]
                
            model_pred, outs = generate_model_rollout(
                N_test // save_factor,
                test_data,
                model,
                hist,
                N_in,
                N_extra,
                initial_input,
                Nb,
                region,
            )
            da = xr.DataArray(
                data=model_pred,
                dims=["time", "x", "y", "var"],
            )

            if i == 0:
                da.to_zarr(pred_path, mode="w")
            else:
                da.to_zarr(pred_path, mode="a", append_dim="time")
            print("Saved: ", i, " to ", i + N_test // save_factor)
            print(f"Time taken for generating {N_test // save_factor} predictions: {time.time() - start_time} seconds")


import time 
start_time = time.time()
if args.run_gen_pred:
    generate_pred_lateral()
    
print(f"Time taken for generating {N_test} predictions: {time.time() - start_time} seconds")