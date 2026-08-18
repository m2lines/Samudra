#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8

# ### Imports

# In[1]:


import datetime

datetime.datetime.now()


# In[2]:


import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from xarrayutils.plotting import linear_piecewise_scale

sys.path.append("../ocean_emulators_main/")
from ocean_emulators.dataset_validation import ds_prediction_validate


# ### Data

# In[3]:


def post_processor(ds: xr.Dataset, ds_truth: xr.Dataset, ls) -> xr.Dataset:
    """Converts the prediction output to an xarray dataset with the same dimensions/variables as input"""
    # correct swapped dimensions and warn
    if len(ds.x) == 180 and len(ds.y) == 360:
        ds = ds.rename({"x": "x_i", "y": "y_i"}).rename({"x_i": "y", "y_i": "x"})
        warnings.warn(
            "Swapped x and y dimensions detected. Fixing this now, but should be corrected upstream"
        )
    key = list(ds.variables.keys())[0]
    da = ds[key]
    n_lev = 19
    if set(ls) - {"zos"} == set(["uo", "vo", "thetao", "so"]):
        variables = ["uo", "vo", "thetao", "so"]
    elif set(ls) - {"zos"} == set(["thetao", "so"]):
        variables = ["thetao", "so"]
    elif set(ls) - {"zos"} == set(["uo", "vo"]):
        variables = ["uo", "vo"]
    elif set(ls) - {"zos"} == set(["thetao"]):
        variables = ["thetao"]
    slices = [slice(i, i + n_lev) for i in range(0, len(variables) * n_lev, n_lev)]
    var_slices = {k: sl for k, sl in zip(variables, slices)}
    variables = {
        k: da.isel(var=sl).rename({"var": "lev"}) for k, sl in var_slices.items()
    }
    if "zos" in ls:
        variables["zos"] = da.isel(var=-1).squeeze()

    ds_out = xr.Dataset(variables)
    for var in ds_out.data_vars:
        if "lev" in ds_out[var].dims:
            ds_out[var] = ds_out[var].where(ds_truth.wetmask)
        else:
            ds_out[var] = ds_out[var].where(ds_truth.wetmask.isel(lev=0))

    ## attach all coordinates from input
    ds_out = ds_out.assign_coords({co: ds_truth[co] for co in ds_truth.coords})
    ds_out.attrs = ds.attrs

    return ds_out


# In[4]:


##### Paths
# Modes - onlytemp, TS, slow, all
pred_dict = {
    "pred_1": {
        "mode": "slow",
        "name": "Thermo Hist 1 Epoch 55 6000_0_2000",
        "path": "/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2025-02-01_ConvNextUNetCM4Hist1NofastinoutEpochs70Epoch55_6000_0_2000_Train_global_3D_Test_global_3D_all_N_train_6000_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_6000_rand_seed_1.zarr",
    },
    # "pred_2": {
    #     "mode": "slow",
    #     "name": "Thermo Hist 1 Epoch 55",
    #     "path": "/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2024-11-22_ConvNextUNetCM4Hist1NofastinoutEpochs70Epoch55_Train_global_3D_Test_global_3D_all_N_train_13800_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_13800_rand_seed_1.zarr",
    # },
}

#
# "name": "Thermo Hist 1 Epoch 55 2000",
# "path": "/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2025-02-01_ConvNextUNetCM4Hist1NofastinoutEpochs70Epoch55_2000_Train_global_3D_Test_global_3D_all_N_train_12400_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_12400_rand_seed_1.zarr",

# "name": "Thermo Hist 1 Epoch 55 6000_0_2000",
# "path": "/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2025-02-01_ConvNextUNetCM4Hist1NofastinoutEpochs70Epoch55_6000_0_2000_Train_global_3D_Test_global_3D_all_N_train_6000_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_6000_rand_seed_1.zarr",

dataset_name = "CM4"  # CM4, OM4


# pred_dict = {
#     "pred_1": {
#         "mode": "slow",
#         "name": "Thermo",
#         "path": "/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2024-09-20_ConvNextUNetTrain3Dv021Eval3Dhfdsanoms1975NofastinoutEpochs70Epoch55_Train_global_3D_Test_global_3D_all_N_train_2850_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_2850_rand_seed_1.zarr",
#     },
#     "pred_2": {
#         "mode": "all",
#         "name": "Thermo+Dynamic",
#         "path": "/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2024-09-20_ConvNextUNetTrain3Dv021Eval3Dhfdsanoms1975Epochs70Epoch55_Train_global_3D_Test_global_3D_all_N_train_2850_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_2850_rand_seed_1.zarr",
#     },
# }

# dataset_name = "OM4"  # CM4, OM4


prefix_path = dataset_name
clist = ["#ff807a", "#1e8685", "#ffb579", "#63c8ab"]

# Rollout details
levels = 19


# In[5]:


def get_om4_gt(slice_time=True):
    ds_input = xr.open_zarr(
        os.path.join("/pscratch/sd/s/suryad/data", "OM4_5daily_v0.2.1.zarr")
    )
    ds_groundtruth = ds_input.sel(time=slice("1975-01-01", None))
    if slice_time:
        ds_groundtruth = ds_input.isel(time=slice(2903, 2903 + 600))

    return ds_groundtruth.isel(lev=slice(None, levels))


def get_cm4_gt(slice_time=True):
    import fsspec

    fs_osn = fsspec.filesystem(
        "s3",
        profile="ocean_emulator",  ## This is the profile name you configured above.
    )
    mapper = fs_osn.get_mapper(
        "emulators/jbusecke/ocean-emulators/CM4_5daily_v0.4.0.zarr"
    )
    ds_input = xr.open_zarr(mapper, consolidated=True)
    ds_groundtruth = ds_input.drop_vars(["lat_b", "lon_b"])

    if slice_time:
        ds_groundtruth = ds_input.isel(time=slice(13941, 13941 + 600))

    return ds_groundtruth.isel(lev=slice(None, levels))


def get_dataset(slice_time=True):
    if dataset_name == "OM4":
        ds_groundtruth = get_om4_gt(slice_time)
    elif dataset_name == "CM4":
        ds_groundtruth = get_cm4_gt(slice_time)
    else:
        raise ValueError("Incorrect dataset name")

    ds_groundtruth = ds_groundtruth.astype("float32")
    return ds_groundtruth


# In[6]:


import copy
import json

import pandas as pd

ds_groundtruth = get_dataset()

output_path = (
    "../outputs/"
    + str(datetime.now())[:10]
    + "_"
    + prefix_path
    + "_"
    + "_".join([pred_dict[k]["name"] for k in pred_dict.keys()])
)
print("Using Output Folder : ", output_path)
if not os.path.isdir(os.path.join(output_path)):
    os.makedirs(os.path.join(output_path))

compare_info_dict = copy.deepcopy(pred_dict)

start_step, end_step = None, None
for k in pred_dict.keys():
    if pred_dict[k]["mode"] == "slow":
        pred_dict[k]["ls"] = ["thetao", "so", "zos"]
    elif pred_dict[k]["mode"] == "onlytemp":
        pred_dict[k]["ls"] = ["thetao"]
    elif pred_dict[k]["mode"] == "TS":
        pred_dict[k]["ls"] = ["thetao", "so"]
    else:
        pred_dict[k]["ls"] = ["uo", "vo", "thetao", "so", "zos"]

    Pred_path = pred_dict[k]["path"]
    ds_prediction_raw = xr.open_zarr(Pred_path)

    if "model_path" in ds_prediction_raw.attrs:
        compare_info_dict[k]["model_path"] = ds_prediction_raw.attrs["model_path"]

    if "start_step" in ds_prediction_raw.attrs:
        compare_info_dict[k]["start_step"] = ds_prediction_raw.attrs["start_step"]
        compare_info_dict[k]["end_step"] = ds_prediction_raw.attrs["end_step"]

        if start_step is None and end_step is None:
            start_step = ds_prediction_raw.attrs["start_step"]
            end_step = ds_prediction_raw.attrs["end_step"]

        elif (
            start_step != ds_prediction_raw.attrs["start_step"]
            or end_step != ds_prediction_raw.attrs["end_step"]
        ):
            assert False, "Mismatched indices"

        ds_groundtruth = get_dataset(slice_time=False)
        ds_groundtruth = ds_groundtruth.isel(time=slice(start_step, end_step))

    ds_prediction = post_processor(
        ds_prediction_raw, ds_groundtruth, pred_dict[k]["ls"]
    )

    ds_prediction_validate(ds_prediction)
    pred_dict[k]["ds_prediction"] = ds_prediction

with open(os.path.join(output_path, "compare_info.txt"), "w") as f:
    f.write(json.dumps(compare_info_dict, sort_keys=True, indent=4))


# In[7]:


get_ipython().run_line_magic("matplotlib", "inline")


# In[8]:


def profile_mean(ds: xr.Dataset) -> xr.Dataset:
    return ds.weighted(ds.areacello).mean(["x", "y"])


# In[9]:


with ProgressBar():
    print("Ground truth " + dataset_name)
    ds_groundtruth = ds_groundtruth.assign(
        KE=0.5 * (ds_groundtruth.uo**2 + ds_groundtruth.vo**2) * 1020
    )
    profile_groundtruth = profile_mean(ds_groundtruth).load()

    for k in pred_dict.keys():
        print(k)
        if "uo" in pred_dict[k]["ls"]:
            pred_dict[k]["ds_prediction"] = pred_dict[k]["ds_prediction"].assign(
                KE=0.5
                * (
                    pred_dict[k]["ds_prediction"].uo ** 2
                    + pred_dict[k]["ds_prediction"].vo ** 2
                )
                * 1020
            )
            pred_dict[k]["ls"].append("KE")
        pred_dict[k]["profile_prediction"] = profile_mean(
            pred_dict[k]["ds_prediction"]
        ).load()


# In[10]:


data = ds_groundtruth
data = data.drop_vars(["lon", "lat"])
data = data.transpose("time", "lev", ...)
data["y"] = data.y.assign_attrs(long_name="latitude")
data["x"] = data.x.assign_attrs(long_name="longitude")
data["thetao"] = data["thetao"].assign_attrs(
    long_name=r"${\theta_O}$", units=r"$\degree C$"
)
data["lev"] = data["lev"].assign_attrs(long_name="depth", units="m")
data["so"] = data["so"].assign_attrs(long_name=r"${s}$", units=r"psu")
data["zos"] = data["zos"].assign_attrs(long_name=r"SSH", units=r"m")
data["vo"] = data["vo"].assign_attrs(long_name=r"${v}$", units=r"m/s")
data["uo"] = data["uo"].assign_attrs(long_name=r"${u}$", units=r"m/s")

for k in pred_dict.keys():
    pred_dict[k]["ds_prediction"] = pred_dict[k]["ds_prediction"].transpose(
        "time", "lev", ...
    )

for k in pred_dict.keys():
    pred_dict[k]["ds_prediction"]["y"] = pred_dict[k]["ds_prediction"].y.assign_attrs(
        long_name="latitude"
    )
    pred_dict[k]["ds_prediction"]["x"] = pred_dict[k]["ds_prediction"].x.assign_attrs(
        long_name="longitude"
    )
    pred_dict[k]["ds_prediction"]["thetao"] = pred_dict[k]["ds_prediction"][
        "thetao"
    ].assign_attrs(long_name=r"${\theta_O}$", units=r"$\degree C$")
    pred_dict[k]["ds_prediction"]["lev"] = pred_dict[k]["ds_prediction"][
        "lev"
    ].assign_attrs(long_name="depth", units="m")
    if "so" in pred_dict[k]["ls"]:
        pred_dict[k]["ds_prediction"]["so"] = pred_dict[k]["ds_prediction"][
            "so"
        ].assign_attrs(long_name=r"${s}$", units=r"psu")
    if "zos" in pred_dict[k]["ls"]:
        pred_dict[k]["ds_prediction"]["zos"] = pred_dict[k]["ds_prediction"][
            "zos"
        ].assign_attrs(long_name=r"SSH", units=r"m")
    if "uo" in pred_dict[k]["ls"]:
        pred_dict[k]["ds_prediction"]["vo"] = pred_dict[k]["ds_prediction"][
            "vo"
        ].assign_attrs(long_name=r"${v}$", units=r"m/s")
        pred_dict[k]["ds_prediction"]["uo"] = pred_dict[k]["ds_prediction"][
            "uo"
        ].assign_attrs(long_name=r"${u}$", units=r"m/s")


# In[11]:


var_list = {
    "vo": r"$v$ $( m/s )$",
    "uo": r"$u$ $( m/s )$",
    "thetao": r"$T$ $( ^\circ C )$",
    "so": r"$so$ $( psu )$",
    "zos": r"$zos$ $( m )$",
    "KE": r"$KE$ $( J/m^2 )$",
    "OHC": r"$OHC$ $Anomaly$ $( ZJ )$",
}


def remove_climatology(ds):
    # Compute the climatology on the detrended data
    climatology = ds.groupby("time.dayofyear").mean("time").compute()

    # Remove the seasonal cycle (climatology) from the detrended data
    day_of_year = ds["time"].dt.dayofyear
    res = (ds - climatology.sel(dayofyear=day_of_year)).compute()

    return res


# In[12]:


# Create folder paths
timeseries_path = os.path.join(output_path, f"Timeseries")
if not os.path.isdir(timeseries_path):
    os.makedirs(timeseries_path)

ohc_path = os.path.join(output_path, f"OHC")
if not os.path.isdir(ohc_path):
    os.makedirs(ohc_path)

temp_path = os.path.join(output_path, f"Temperature")
if not os.path.isdir(temp_path):
    os.makedirs(temp_path)

salinity_path = os.path.join(output_path, f"Salinity")
if not os.path.isdir(salinity_path):
    os.makedirs(salinity_path)

pdfs_path = os.path.join(output_path, f"PDFs")
if not os.path.isdir(pdfs_path):
    os.makedirs(pdfs_path)

enso_path = os.path.join(output_path, f"ENSO")
if not os.path.isdir(enso_path):
    os.makedirs(enso_path)

metrics_path = os.path.join(output_path, f"Metrics")
if not os.path.isdir(metrics_path):
    os.makedirs(metrics_path)

movie_path = os.path.join(output_path, f"Movies")
if not os.path.isdir(movie_path):
    os.makedirs(movie_path)


# ### Timeseries Plots

# In[13]:


### Plotting timeseries for each variable for each level
for v in ["uo", "vo", "thetao", "so", "zos", "KE"]:
    if not os.path.isdir(os.path.join(timeseries_path, f"{v}_timeseries")):
        os.makedirs(os.path.join(timeseries_path, f"{v}_timeseries"))

    plt.clf()
    plt.rcParams.update({"font.size": 20})
    plt.figure(figsize=[18, 10])

    if v == "zos":
        profile_groundtruth[v].plot(label=dataset_name, c="k")
        for i, k in enumerate(pred_dict.keys()):
            if v in pred_dict[k]["ls"]:
                pred_dict[k]["profile_prediction"][v].plot(
                    label=pred_dict[k]["name"], c=clist[i]
                )
        min_val, max_val = plt.ylim()
        plt.ylim(min_val - 0.05, max_val + 0.05)
        plt.xlabel("Time")
        plt.ylabel(var_list[v])
        plt.legend()
        plt.savefig(
            os.path.join(timeseries_path, f"{v}_timeseries/0.png"),
            bbox_inches="tight",
            dpi=600,
        )
        plt.close()
    else:
        for lev in range(levels):
            plt.clf()
            plt.rcParams.update({"font.size": 20})
            plt.figure(figsize=[18, 10])
            profile_groundtruth[v].isel(lev=lev).plot(label=dataset_name, c="k")
            min_val, max_val = plt.ylim()
            for i, k in enumerate(pred_dict.keys()):
                if v in pred_dict[k]["ls"]:
                    pred_dict[k]["profile_prediction"][v].isel(lev=lev).plot(
                        label=pred_dict[k]["name"], c=clist[i]
                    )
            if v == "thetao":
                plt.ylim(min_val - 0.25, max_val + 0.25)
            elif v == "so":
                plt.ylim(min_val - 0.2, max_val + 0.2)
            elif v == "KE":
                plt.ylim(min_val - 0.5, max_val + 0.5)

            plt.xlabel("Time")
            plt.ylabel(var_list[v])
            plt.legend()
            plt.savefig(
                os.path.join(timeseries_path, f"{v}_timeseries/{lev}.png"),
                bbox_inches="tight",
                dpi=600,
            )
            plt.close()


# In[14]:


# Short Timeseries plots
import os

import matplotlib.pyplot as plt

shallow_levels = [2.5, 775]
num_shallow_levels = len(shallow_levels)

plt.rcParams.update({"font.size": 14})

num_plots = 0
for var in ["thetao"]:
    if "lev" in pred_dict[k]["ds_prediction"][var].coords:
        num_plots += num_shallow_levels  # One plot per level
    else:
        num_plots += 1  # One plot for scalar variables

# Set grid size dynamically based on the number of required plots
cols = 2  # Number of columns
rows = 1

fig, axes = plt.subplots(rows, cols, figsize=(16, 3))
axes = axes.flatten()  # Flatten the 2D array of axes for easy access

plot_idx = 0  # Initialize plot index to track subplot positions

# Loop over each variable to plot its time series
for v in ["thetao"]:
    if v == "zos":
        ax = axes[plot_idx]

        # Ground truth plot
        profile_groundtruth[v].plot(ax=ax, label=dataset_name, c="k")
        min_val = profile_groundtruth[v].min()
        max_val = profile_groundtruth[v].max()

        for i, k in enumerate(pred_dict.keys()):
            pred_dict[k]["profile_prediction"][v].plot(
                ax=ax, label=pred_dict[k]["name"], c=clist[i]
            )

        ax.set_ylim(min_val - 0.05, max_val + 0.05)
        ax.set_title(f"{v}")
        ax.set_xlabel("Time")
        ax.set_ylabel(var_list[v])

        plot_idx += 1

    else:
        for lev in shallow_levels:
            ax = axes[plot_idx]

            # Ground truth plot
            profile_groundtruth[v].sel(lev=lev).plot(ax=ax, label=dataset_name, c="k")
            mins, maxs = plt.ylim()

            for i, k in enumerate(pred_dict.keys()):
                pred_dict[k]["profile_prediction"][v].sel(lev=lev).plot(
                    ax=ax, label=pred_dict[k]["name"], c=clist[i]
                )

            # Adjust y-axis limits
            if v == "thetao":
                if lev > 100:
                    ax.set_ylim(mins - 0.02, maxs + 0.02)
                # else:
                #     ax.set_ylim(mins - 0.25, maxs + 0.25)
            elif v == "so":
                ax.set_ylim(mins - 0.2, maxs + 0.2)

            ax.set_title(f"{lev}m" + r" $\theta_O$", fontsize=14)
            ax.set_xlabel("Time")
            ax.set_ylabel(r"$\theta_O$ [$\degree C$]")

            plot_idx += 1
            if plot_idx >= rows * cols:
                break

# Adjust layout to avoid overlap and place the legend outside the plot
fig.tight_layout(rect=[0, 0, 0.85, 0.96])

# Create a single legend for all plots
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.455, 0.9), ncols=3)

# Remove any empty subplots
for idx in range(num_plots, len(axes)):
    fig.delaxes(axes[idx])

# Save the figure
output_file = os.path.join(
    timeseries_path, "temperature_timeseries_grid_shallow_both.png"
)
plt.savefig(output_file, bbox_inches="tight", dpi=600)


# In[15]:


import os

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

shallow_levels = [2.5, 775, 2400]  # Define shallow depth levels
num_shallow_levels = len(shallow_levels)

plt.rcParams.update({"font.size": 14})

variables = [
    "so",
    "uo",
    "vo",
    "thetao",
]  # List of variables for rows: salinity, zonal velocity, meridional velocity
cols = len(shallow_levels)  # Number of columns corresponds to shallow levels
rows = len(variables)  # One row per variable

fig, axes = plt.subplots(
    rows, cols, figsize=(16, 12), gridspec_kw={"wspace": 0.27, "hspace": 0.2}
)  # Adjust figure size for more rows
axes = axes.reshape(rows, cols)  # Reshape axes for easy access by row and column

# Define labels for each variable
var_labels = {
    "so": "S [psu]",
    "uo": "uo [m/s]",
    "vo": "vo [m/s]",
    "thetao": r"$\theta_O$ [$\degree C$]",
}

# Loop over each variable and plot profiles for each shallow level
for row_idx, var in enumerate(variables):
    for col_idx, lev in enumerate(shallow_levels):
        ax = axes[row_idx, col_idx]  # Access subplot by row and column

        if var == "thetao" and (lev == 2.5 or lev == 775):
            fig.delaxes(ax)
            continue

        # Ground truth plot for each variable at the specified level
        profile_groundtruth[var].sel(lev=lev).plot(ax=ax, label=dataset_name, c="k")
        mins, maxs = ax.get_ylim()

        for i, k in enumerate(pred_dict.keys()):
            if var in pred_dict[k]["ls"]:
                pred_dict[k]["profile_prediction"][var].sel(lev=lev).plot(
                    ax=ax, label=pred_dict[k]["name"], c=clist[i]
                )

        # Adjust y-axis limits based on the variable

        if var == "thetao":
            if lev > 2000:
                ax.set_ylim(mins - 0.01, maxs + 0.01)
            elif lev > 100:
                ax.set_ylim(mins - 0.01, maxs + 0.01)
            else:
                ax.set_ylim(mins - 0.25, maxs + 0.25)
            ax.set_title(f"{lev}m " + r"$\theta_O$", fontsize=14)
        elif var == "so":  # Salinity
            if lev > 2000:
                ax.set_ylim(mins - 0.001, maxs + 0.001)
            elif lev > 100:
                ax.set_ylim(mins - 0.004, maxs + 0.004)
            else:
                ax.set_ylim(mins - 0.1, maxs + 0.1)
            ax.set_title(f"{lev}m $S$", fontsize=14)
            handles, labels = ax.get_legend_handles_labels()
        if v in pred_dict[k]["ls"]:
            if var == "uo":  # Zonal velocity
                if lev > 2000:
                    ax.set_ylim(mins - 0.0003, maxs + 0.0003)
                elif lev > 100:
                    ax.set_ylim(mins - 0.0005, maxs + 0.0005)
                else:
                    ax.set_ylim(mins - 0.005, maxs + 0.005)
                ax.set_title(f"{lev}m $uo$", fontsize=14)
            elif var == "vo":  # Meridional velocity
                if lev > 2000:
                    ax.set_ylim(mins - 0.0002, maxs + 0.0002)
                elif lev > 100:
                    ax.set_ylim(mins - 0.0004, maxs + 0.0004)
                else:
                    ax.set_ylim(mins - 0.005, maxs + 0.005)
                ax.set_title(f"{lev}m $vo$", fontsize=14)

        # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        # Set y-axis label only on the leftmost column
        if col_idx == 0 or var == "thetao":
            ax.set_ylabel(var_labels[var])  # Set y-axis label based on variable
        else:
            ax.set_ylabel("")
            # ax.set_yticklabels([])  # Hide y-axis tick labels for other columns

        # Set x-axis label and ticks only on the bottommost row
        if row_idx == rows - 1:
            ax.set_xlabel("Time")
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])


# Adjust layout to avoid overlap and place the legend outside the plot
# fig.tight_layout(rect=[0, 0, 0.85, 0.96])

# Create a single legend for all plots
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.90), ncols=3)

# Save the figure with an updated filename
output_file = os.path.join(timeseries_path, "timeseries_grid_shallow_all_vars.png")
plt.savefig(output_file, bbox_inches="tight", dpi=600)
# plt.show()
# plt.close(fig)


# In[16]:


import os

import matplotlib.pyplot as plt

shallow_levels = [2.5, 105, 550]

num_shallow_levels = len(shallow_levels)

plt.rcParams.update({"font.size": 12})
num_plots = 0
for var in ["thetao"]:
    if "lev" in pred_dict[k]["ds_prediction"][var].coords:
        num_plots += num_shallow_levels  # One plot per level
    else:
        num_plots += 1  # One plot for scalar variables

# Set grid size dynamically based on the number of required plots
cols = 3  # Number of columns
rows = int(np.ceil(num_plots / cols))

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 3))
axes = axes.flatten()  # Flatten the 2D array of axes for easy access

plot_idx = 0  # Initialize plot index to track subplot positions

# Loop over each variable to plot its time series
for v in ["thetao"]:
    if v == "zos":
        # Handle 'zos' separately (no levels)
        ax = axes[plot_idx]
        profile_groundtruth[v].plot(ax=ax, label=dataset_name, c="k")

        for i, k in enumerate(pred_dict.keys()):
            if v in pred_dict[k]["ls"]:
                pred_dict[k]["profile_prediction"][v].plot(
                    ax=ax, label=pred_dict[k]["name"], c=clist[i]
                )

        # Adjust y-axis limits and formatting
        min_val, max_val = ax.get_ylim()
        ax.set_ylim(min_val - 0.05, max_val + 0.05)
        ax.set_title(f"{v}", fontsize=14)
        ax.set_xlabel("Time")
        ax.set_ylabel(var_list[v])

        plot_idx += 1  # Move to the next subplot

    else:
        # For other variables, loop over each level
        for lev in shallow_levels:
            ax = axes[plot_idx]
            profile_groundtruth[v].sel(lev=lev).plot(ax=ax, label=dataset_name, c="k")

            min_val, max_val = ax.get_ylim()
            for i, k in enumerate(pred_dict.keys()):
                if v in pred_dict[k]["ls"]:
                    pred_dict[k]["profile_prediction"][v].sel(lev=lev).plot(
                        ax=ax, label=pred_dict[k]["name"], c=clist[i]
                    )

            # Adjust y-axis limits for specific variables
            if v == "thetao":
                ax.set_ylim(min_val - 0.25, max_val + 0.25)
            elif v == "so":
                ax.set_ylim(min_val - 0.2, max_val + 0.2)

            ax.set_title(
                r"$\theta_O$"
                + f" at {pred_dict[k]['profile_prediction'][v].sel(lev=lev).lev.item()}m"
                + r" ($\degree C$)",
                fontsize=14,
            )
            ax.set_xlabel("Time")
            ax.set_ylabel(var_list[v])

            plot_idx += 1  # Move to the next subplot

            if plot_idx >= rows * cols:
                break  # Stop if the grid is full

# Adjust layout to avoid overlap and place the legend outside the plot
fig.tight_layout(rect=[0, 0, 0.85, 0.96])  # Leave space on the right for the legend

# Create a single legend for all plots
handles, labels = ax.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="center left",
    ncols=3,
    bbox_to_anchor=(0.34, -0.03),
    fontsize=12,
)

# fig.suptitle("Time Series Plots", fontsize=24)

# Remove any empty subplots
for idx in range(num_plots, len(axes)):
    fig.delaxes(axes[idx])

# Save the figure
output_file = os.path.join(timeseries_path, "temp_timeseries_grid_shallow_skipped.png")
plt.savefig(output_file, bbox_inches="tight", dpi=600)


# In[17]:


var = "thetao"
Days_to_Eq = 0

plt.rcdefaults()
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(
    1, 1, figsize=(16, 3), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
)

thetao = (
    ds_groundtruth["thetao"]
    .weighted(ds_groundtruth["areacello"] * ds_groundtruth["dz"])
    .mean(["x", "y", "lev"])
)
thetao = thetao.rename(r"$\theta_O$")
thetao = thetao.assign_attrs(units=r"$\degree C$")

for i, k in enumerate(pred_dict.keys()):
    thetao_pred = (
        pred_dict[k]["ds_prediction"][var]
        .weighted(ds_groundtruth["areacello"] * ds_groundtruth["dz"])
        .mean(["x", "y", "lev"])
    )
    thetao_pred.plot(ax=ax, label=pred_dict[k]["name"], c=clist[i])
    coeffs_ = np.polyfit(
        np.arange(thetao_pred[Days_to_Eq:].size), thetao_pred[Days_to_Eq:], 1
    )
    print(f"{pred_dict[k]['name']}: {coeffs_[0] * 73}")

thetao.plot(ax=ax, label=dataset_name, c="k")
coeffs_OHC_ground_trend = np.polyfit(
    np.arange(thetao[Days_to_Eq:].size), thetao[Days_to_Eq:], 1
)
print("OHC: ", coeffs_OHC_ground_trend[0] * 73)

ax.legend(ncol=3)
# ax.set_ylim([3.230, 3.245])
plt.savefig(
    os.path.join(timeseries_path, f"Global_Thetao_Timeseries"),
    bbox_inches="tight",
    dpi=600,
)
# plt.show()


# In[18]:


from matplotlib.ticker import ScalarFormatter

var = "so"

plt.rcdefaults()
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(
    1, 1, figsize=(16, 3), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
)

salinity = (
    ds_groundtruth["so"]
    .weighted(ds_groundtruth["areacello"] * ds_groundtruth["dz"])
    .mean(["x", "y", "lev"])
)
salinity = salinity.rename("S")
salinity = salinity.assign_attrs(units="psu")

for i, k in enumerate(pred_dict.keys()):
    salinity_pred = (
        pred_dict[k]["ds_prediction"][var]
        .weighted(ds_groundtruth["areacello"] * ds_groundtruth["dz"])
        .mean(["x", "y", "lev"])
    )
    salinity_pred.plot(ax=ax, label=pred_dict[k]["name"], c=clist[i])
    coeffs_ = np.polyfit(
        np.arange(salinity_pred[Days_to_Eq:].size), salinity_pred[Days_to_Eq:], 1
    )
    print(f"{pred_dict[k]['name']}: {coeffs_[0] * 73}")

salinity.plot(ax=ax, label=dataset_name, c="k")
coeffs_salinity_ground_trend = np.polyfit(
    np.arange(salinity[Days_to_Eq:].size), salinity[Days_to_Eq:], 1
)
print("OHC: ", coeffs_salinity_ground_trend[0] * 73)

ax.legend(ncol=3, loc="upper left")
ax.set_ylim([34.723, 34.728])
ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
plt.savefig(
    os.path.join(timeseries_path, f"Global_Salinity_Timeseries"),
    bbox_inches="tight",
    dpi=600,
)


# ### OHC Plots

# In[19]:


import cartopy.crs as ccrs
import cmocean as cm


# #### OHC Timeseries

# In[20]:


c_p = 3850  # J/(kg C)
rho_0 = 1025  # kg/m^3
f = open(os.path.join(output_path, "compare_info.txt"), "a")

plt.rcdefaults()
fig, ax = plt.subplots(
    1, 1, figsize=(10, 3), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
)
plt.rcParams.update({"font.size": 9})

OHC = ((data["thetao"] * c_p * rho_0) * data["areacello"] * data["dz"]).sum(
    ["x", "y", "lev"]
) / 1e21
OHC = remove_climatology(OHC)
OHC = OHC.rename("OHC Anomaly")
OHC = OHC.assign_attrs(units="ZJ")

for i, k in enumerate(pred_dict.keys()):
    OHC_pred = (
        (pred_dict[k]["ds_prediction"]["thetao"] * c_p * rho_0)
        * pred_dict[k]["ds_prediction"]["areacello"]
        * pred_dict[k]["ds_prediction"]["dz"]
    ).sum(["x", "y", "lev"]) / 1e21
    OHC_pred = remove_climatology(OHC_pred)
    OHC_pred = OHC_pred.rename("OHC Anomaly")
    OHC_pred = OHC_pred.assign_attrs(units="ZJ")
    OHC_pred.plot(ax=ax, label=pred_dict[k]["name"], c=clist[i])
    coeffs_OHC_pred_trend = np.polyfit(np.arange(OHC_pred.size), OHC_pred, 1)
    (pos,) = ax.plot(
        OHC_pred.time.data,
        np.arange(OHC_pred.size) * coeffs_OHC_pred_trend[0] + coeffs_OHC_pred_trend[1],
        c=clist[i],
        ls="--",
    )
    # ax[0].annotate(f'{coeffs_OHC_pred_trend[0]:.2e}',
    #          xy=(pos.get_xdata()[-1], pos.get_ydata()[-1]),
    #          xytext=(pos.get_xdata()[-2], pos.get_ydata()[-2]),
    #          fontsize=9, color=clist[i])
    f.write(f"\nOHC {pred_dict[k]['name']} Trend Slope : {coeffs_OHC_pred_trend[0]}")
    pred_dict[k]["OHC_slope"] = coeffs_OHC_pred_trend[0]

OHC.plot(ax=ax, label=dataset_name, c="k")
coeffs_OHC_trend = np.polyfit(np.arange(OHC.size), OHC, 1)
(pos,) = ax.plot(
    OHC.time.data,
    np.arange(OHC.size) * coeffs_OHC_trend[0] + coeffs_OHC_trend[1],
    c="k",
    ls="--",
)
# ax[0].annotate(f'{coeffs_OHC_trend[0]:.2e}',
#              xy=(pos.get_xdata()[0], pos.get_ydata()[0]),
#              xytext=(pos.get_xdata()[1], pos.get_ydata()[1]),
#              fontsize=9, color='k')
f.write(f"\nOHC GT Trend Slope : {coeffs_OHC_trend[0]}")
GT_ohc_slope = coeffs_OHC_trend[0]
# ax.legend()

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.89), ncols=3)

f.close()
plt.savefig(os.path.join(ohc_path, "OHC"), bbox_inches="tight", dpi=600)


# #### Depth wise OHC

# In[21]:


# %matplotlib inline
plt.rcParams.update({"font.size": 14})

Days_to_Eq = 0
c_p = 3850  # J/(kg C)
rho_0 = 1025  # kg/m^3
fig, ax = plt.subplots(
    3, 1, figsize=(10, 7.5), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
)
plt.rcParams.update({"font.size": 9})

f = open(os.path.join(output_path, "compare_info.txt"), "a")


# Upper - GT
OHC_truth_upper = (
    (data["thetao"].sel(lev=slice(0, 700)) * c_p * rho_0)
    * data["areacello"]
    * data["dz"]
).sum(["x", "y", "lev"]) / 1e21

OHC_truth_upper = remove_climatology(OHC_truth_upper)
OHC_truth_upper.plot(ax=ax[0], label=dataset_name, c="k")
coeffs_OHC_ground_trend = np.polyfit(
    np.arange(OHC_truth_upper[Days_to_Eq:].size), OHC_truth_upper[Days_to_Eq:], 1
)
(pos,) = ax[0].plot(
    OHC_truth_upper[Days_to_Eq:].time.data,
    np.arange(OHC_truth_upper[Days_to_Eq:].size) * coeffs_OHC_ground_trend[0]
    + coeffs_OHC_ground_trend[1],
    c="k",
    ls="--",
)
# ax[0].annotate(f'{coeffs_OHC_ground_trend[0]:.2e}',
#              xy=(pos.get_xdata()[0], pos.get_ydata()[0]),
#              xytext=(pos.get_xdata()[1], pos.get_ydata()[1]),
#              fontsize=9, color='k')
f.write(f"\nUpper - GT Trend Slope : {coeffs_OHC_ground_trend[0]}")
GT_upper = coeffs_OHC_ground_trend[0]
upper_trend_truth = coeffs_OHC_ground_trend[0] * 73
print("upper_trend_truth: ", upper_trend_truth)

# Upper - Pred
for i, k in enumerate(pred_dict.keys()):
    pred_dict[k]["OHC_pred_upper"] = (
        (pred_dict[k]["ds_prediction"]["thetao"].sel(lev=slice(0, 700)) * c_p * rho_0)
        * pred_dict[k]["ds_prediction"]["areacello"]
        * pred_dict[k]["ds_prediction"]["dz"]
    ).sum(["x", "y", "lev"]) / 1e21

    pred_dict[k]["OHC_pred_upper"] = remove_climatology(pred_dict[k]["OHC_pred_upper"])
    pred_dict[k]["OHC_pred_upper"] = pred_dict[k]["OHC_pred_upper"].rename("0-0.7km")
    pred_dict[k]["OHC_pred_upper"] = pred_dict[k]["OHC_pred_upper"].assign_attrs(
        units="ZJ"
    )
    pred_dict[k]["coeffs_OHC_pred_trend_upper"] = np.polyfit(
        np.arange(pred_dict[k]["OHC_pred_upper"][Days_to_Eq:].size),
        pred_dict[k]["OHC_pred_upper"][Days_to_Eq:],
        1,
    )
    pred_dict[k]["OHC_pred_upper"].plot(
        ax=ax[0], label=pred_dict[k]["name"], c=clist[i]
    )
    (pos,) = ax[0].plot(
        pred_dict[k]["OHC_pred_upper"][Days_to_Eq:].time.data,
        np.arange(pred_dict[k]["OHC_pred_upper"][Days_to_Eq:].size)
        * pred_dict[k]["coeffs_OHC_pred_trend_upper"][0]
        + pred_dict[k]["coeffs_OHC_pred_trend_upper"][1],
        c=clist[i],
        ls="--",
    )
    # ax[0].annotate(f'{pred_dict[k]["coeffs_OHC_pred_trend_upper"][0]:.2e}',
    #          xy=(pos.get_xdata()[-1], pos.get_ydata()[-1]),
    #          xytext=(pos.get_xdata()[-2], pos.get_ydata()[-2]),
    #          fontsize=9, color=clist[i])
    f.write(
        f"\nUpper - {pred_dict[k]['name']} Trend Slope : {pred_dict[k]['coeffs_OHC_pred_trend_upper'][0]}"
    )
    pred_dict[k]["upper_trend_pred"] = (
        pred_dict[k]["coeffs_OHC_pred_trend_upper"][0] * 73
    )
    print(pred_dict[k]["name"], " upper_trend_pred: ", pred_dict[k]["upper_trend_pred"])
    pred_dict[k]["total_trend_pred"] = pred_dict[k]["upper_trend_pred"]

# ax[0].legend()
ax[0].set_title("OHC Anomaly")


# Middle - GT
OHC_truth_mid = (
    (data["thetao"].sel(lev=slice(700, 2000)) * c_p * rho_0)
    * data["areacello"]
    * data["dz"]
).sum(["x", "y", "lev"]) / 1e21

OHC_truth_mid = remove_climatology(OHC_truth_mid)
OHC_truth_mid.plot(ax=ax[1], label=dataset_name, c="k")
coeffs_OHC_ground_trend = np.polyfit(
    np.arange(OHC_truth_mid[Days_to_Eq:].size), OHC_truth_mid[Days_to_Eq:], 1
)
(pos,) = ax[1].plot(
    OHC_truth_mid[Days_to_Eq:].time.data,
    np.arange(OHC_truth_mid[Days_to_Eq:].size) * coeffs_OHC_ground_trend[0]
    + coeffs_OHC_ground_trend[1],
    c="k",
    ls="--",
)
# ax[1].annotate(f'{coeffs_OHC_ground_trend[0]:.2e}',
#              xy=(pos.get_xdata()[0], pos.get_ydata()[0]),
#              xytext=(pos.get_xdata()[1], pos.get_ydata()[1]),
#              fontsize=9, color='k')
f.write(f"\nMiddle - GT Trend Slope : {coeffs_OHC_ground_trend[0]}")
GT_mid = coeffs_OHC_ground_trend[0]
mid_trend_truth = coeffs_OHC_ground_trend[0] * 73
print("mid_trend_truth: ", mid_trend_truth)

# Middle - Pred
for i, k in enumerate(pred_dict.keys()):
    pred_dict[k]["OHC_pred_mid"] = (
        (
            pred_dict[k]["ds_prediction"]["thetao"].sel(lev=slice(700, 2000))
            * c_p
            * rho_0
        )
        * pred_dict[k]["ds_prediction"]["areacello"]
        * pred_dict[k]["ds_prediction"]["dz"]
    ).sum(["x", "y", "lev"]) / 1e21

    pred_dict[k]["OHC_pred_mid"] = remove_climatology(pred_dict[k]["OHC_pred_mid"])
    pred_dict[k]["OHC_pred_mid"] = pred_dict[k]["OHC_pred_mid"].rename("0.7-2.0km")
    pred_dict[k]["OHC_pred_mid"] = pred_dict[k]["OHC_pred_mid"].assign_attrs(units="ZJ")
    pred_dict[k]["coeffs_OHC_pred_trend_mid"] = np.polyfit(
        np.arange(pred_dict[k]["OHC_pred_mid"][Days_to_Eq:].size),
        pred_dict[k]["OHC_pred_mid"][Days_to_Eq:],
        1,
    )
    pred_dict[k]["OHC_pred_mid"].plot(ax=ax[1], label=pred_dict[k]["name"], c=clist[i])
    (pos,) = ax[1].plot(
        pred_dict[k]["OHC_pred_mid"][Days_to_Eq:].time.data,
        np.arange(pred_dict[k]["OHC_pred_mid"][Days_to_Eq:].size)
        * pred_dict[k]["coeffs_OHC_pred_trend_mid"][0]
        + pred_dict[k]["coeffs_OHC_pred_trend_mid"][1],
        c=clist[i],
        ls="--",
    )
    # ax[1].annotate(f'{pred_dict[k]["coeffs_OHC_pred_trend_mid"][0]:.2e}',
    #          xy=(pos.get_xdata()[-1], pos.get_ydata()[-1]),
    #          xytext=(pos.get_xdata()[-2], pos.get_ydata()[-2]),
    #          fontsize=9, color=clist[i])
    f.write(
        f"\nMiddle - {pred_dict[k]['name']} Trend Slope : {pred_dict[k]['coeffs_OHC_pred_trend_mid'][0]}"
    )
    pred_dict[k]["mid_trend_pred"] = pred_dict[k]["coeffs_OHC_pred_trend_mid"][0] * 73
    print(pred_dict[k]["name"], " mid_trend_pred: ", pred_dict[k]["mid_trend_pred"])
    pred_dict[k]["total_trend_pred"] += pred_dict[k]["mid_trend_pred"]

# ax[1].legend()
# ax[1].set_title("OHC Anomaly")

# Deep - GT
OHC_truth_deep = (
    (data["thetao"].sel(lev=slice(2000, None)) * c_p * rho_0)
    * data["areacello"]
    * data["dz"]
).sum(["x", "y", "lev"]) / 1e21

OHC_truth_deep = remove_climatology(OHC_truth_deep)
OHC_truth_deep.plot(ax=ax[2], label=dataset_name, c="k")
coeffs_OHC_ground_trend = np.polyfit(
    np.arange(OHC_truth_deep[Days_to_Eq:].size), OHC_truth_deep[Days_to_Eq:], 1
)
(pos,) = ax[2].plot(
    OHC_truth_deep[Days_to_Eq:].time.data,
    np.arange(OHC_truth_deep[Days_to_Eq:].size) * coeffs_OHC_ground_trend[0]
    + coeffs_OHC_ground_trend[1],
    c="k",
    ls="--",
)
# ax[2].annotate(f'{coeffs_OHC_ground_trend[0]:.2e}',
#              xy=(pos.get_xdata()[0], pos.get_ydata()[0]),
#              xytext=(pos.get_xdata()[1], pos.get_ydata()[1]),
#              fontsize=9, color='k')
f.write(f"\nDeep - GT Trend Slope : {coeffs_OHC_ground_trend[0]}")
GT_deep = coeffs_OHC_ground_trend[0]
deep_trend_truth = coeffs_OHC_ground_trend[0] * 73
print("deep_trend_truth: ", deep_trend_truth)

# Deep - Pred
for i, k in enumerate(pred_dict.keys()):
    pred_dict[k]["OHC_pred_deep"] = (
        (
            pred_dict[k]["ds_prediction"]["thetao"].sel(lev=slice(2000, None))
            * c_p
            * rho_0
        )
        * pred_dict[k]["ds_prediction"]["areacello"]
        * pred_dict[k]["ds_prediction"]["dz"]
    ).sum(["x", "y", "lev"]) / 1e21

    pred_dict[k]["OHC_pred_deep"] = remove_climatology(pred_dict[k]["OHC_pred_deep"])
    pred_dict[k]["OHC_pred_deep"] = pred_dict[k]["OHC_pred_deep"].rename("2.0-6.0km")
    pred_dict[k]["OHC_pred_deep"] = pred_dict[k]["OHC_pred_deep"].assign_attrs(
        units="ZJ"
    )
    pred_dict[k]["coeffs_OHC_pred_trend_deep"] = np.polyfit(
        np.arange(pred_dict[k]["OHC_pred_deep"][Days_to_Eq:].size),
        pred_dict[k]["OHC_pred_deep"][Days_to_Eq:],
        1,
    )
    pred_dict[k]["OHC_pred_deep"].plot(ax=ax[2], label=pred_dict[k]["name"], c=clist[i])
    (pos,) = ax[2].plot(
        pred_dict[k]["OHC_pred_deep"][Days_to_Eq:].time.data,
        np.arange(pred_dict[k]["OHC_pred_deep"][Days_to_Eq:].size)
        * pred_dict[k]["coeffs_OHC_pred_trend_deep"][0]
        + pred_dict[k]["coeffs_OHC_pred_trend_deep"][1],
        c=clist[i],
        ls="--",
    )
    # ax[2].annotate(f'{pred_dict[k]["coeffs_OHC_pred_trend_deep"][0]:.2e}',
    #      xy=(pos.get_xdata()[-1], pos.get_ydata()[-1]),
    #      xytext=(pos.get_xdata()[-2], pos.get_ydata()[-2]),
    #      fontsize=9, color=clist[i])
    f.write(
        f"\nDeep - {pred_dict[k]['name']} Trend Slope : {pred_dict[k]['coeffs_OHC_pred_trend_deep'][0]}"
    )

    pred_dict[k]["deep_trend_pred"] = pred_dict[k]["coeffs_OHC_pred_trend_deep"][0] * 73
    print(pred_dict[k]["name"], " deep_trend_pred: ", pred_dict[k]["deep_trend_pred"])
    pred_dict[k]["total_trend_pred"] += pred_dict[k]["deep_trend_pred"]

# ax[2].legend()
# ax[2].set_title("OHC Anomaly")
total_trend_truth = upper_trend_truth + mid_trend_truth + deep_trend_truth

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.91), ncols=3)

f.write(
    f"\nGT Trend Ratio (Upper, Mid, Deep): {upper_trend_truth / total_trend_truth:.2f}, {mid_trend_truth / total_trend_truth:.2f}, {deep_trend_truth / total_trend_truth:.2f}"
)
for k in pred_dict.keys():
    f.write(
        f"\n{pred_dict[k]['name']} Trend Ratio (Upper, Mid, Deep): {pred_dict[k]['upper_trend_pred'] / pred_dict[k]['total_trend_pred']:.2f}, {pred_dict[k]['mid_trend_pred'] / pred_dict[k]['total_trend_pred']:.2f}, {pred_dict[k]['deep_trend_pred'] / pred_dict[k]['total_trend_pred']:.2f}"
    )
# ax[0].annotate(f'OHC portion of trend (truth, pred): ({upper_trend_truth/total_trend_truth:.2f}, {upper_trend_pred/total_trend_pred:.2f})',xy = (.2,.95), xycoords='axes fraction',
#             horizontalalignment='left', verticalalignment='top')
# ax[1].annotate(f'OHC portion of trend (truth, pred): ({mid_trend_truth/total_trend_truth:.2f}, {mid_trend_pred/total_trend_pred:.2f})',xy = (.2,.95), xycoords='axes fraction',
#             horizontalalignment='left', verticalalignment='top')
# ax[2].annotate(f'OHC portion of trend (truth, pred): ({deep_trend_truth/total_trend_truth:.2f}, {deep_trend_pred/total_trend_pred:.2f})',xy = (.2,.95), xycoords='axes fraction',
#             horizontalalignment='left', verticalalignment='top')
f.write("\n")
f.close()

plt.savefig(
    os.path.join(ohc_path, "OHC_Timeseries_depths"), bbox_inches="tight", dpi=600
)


# In[22]:


pd_data = []
pd_data.append(
    {"Model": dataset_name, "Upper": GT_upper, "Middle": GT_mid, "Deep": GT_deep}
)

for k in pred_dict.keys():
    pd_data.append(
        {
            "Model": pred_dict[k]["name"],
            "Upper": pred_dict[k]["coeffs_OHC_pred_trend_upper"][0],
            "Upper Slope Ratio": pred_dict[k]["coeffs_OHC_pred_trend_upper"][0]
            / GT_upper,
            "Middle": pred_dict[k]["coeffs_OHC_pred_trend_mid"][0],
            "Middle Slope Ratio": pred_dict[k]["coeffs_OHC_pred_trend_mid"][0] / GT_mid,
            "Deep": pred_dict[k]["coeffs_OHC_pred_trend_deep"][0],
            "Deep Slope Ratio": pred_dict[k]["coeffs_OHC_pred_trend_deep"][0] / GT_deep,
        }
    )

# Create a DataFrame
df = pd.DataFrame(pd_data)

# Define the file path
file_path = os.path.join(ohc_path, "depthwise_ohc_slopes_table.csv")

# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)


# #### Basin OHC

# In[23]:


import numpy as np
import xarray as xr


def get_mask_extent(mask: xr.DataArray):
    # Check for required dimensions
    if "y" not in mask.dims or "x" not in mask.dims:
        mask = mask.transpose("lat", "lon")
        mask["y"] = mask.lat
        mask["x"] = mask.lon
        mask = mask.where(mask != 0, np.nan)
    else:
        mask = mask.transpose("y", "x")

    mask = mask.assign_coords(x=(((mask.x + 180) % 360) - 180))
    # Create a boolean mask for non-NaN values
    non_nan_mask = ~np.isnan(mask.values)
    # Get the indices of non-NaN values
    non_nan_indices = np.argwhere(non_nan_mask)

    # Extract latitude and longitude coordinates
    latitudes = mask["y"].values
    longitudes = mask["x"].values

    # Determine the extent
    lat_min = round(latitudes[non_nan_indices[:, 0].min()], 2)
    lat_max = round(latitudes[non_nan_indices[:, 0].max()], 2)
    lon_min = longitudes[non_nan_indices[:, 1].min()]
    lon_max = longitudes[non_nan_indices[:, 1].max()]

    return {"lat": (lat_min, lat_max), "lon": (lon_min, lon_max)}


# In[24]:


def process_mask(mask):
    mask = mask.where(mask != 0, np.nan)
    mask = mask.transpose("lat", "lon")
    mask = mask.assign_coords(lat=data.y.values, lon=data.x.values)
    mask = mask.rename({"lat": "y", "lon": "x"})
    return mask


atlantic_mask0 = xr.open_dataset("/pscratch/sd/s/suryad/data/basin_At_noArctic.nc")[
    "basin"
]
atlantic_mask = atlantic_mask0.where(atlantic_mask0["lat"] >= -32)
atlantic_mask = process_mask(atlantic_mask)
pacific_mask0 = xr.open_dataset("/pscratch/sd/s/suryad/data/basin_Pa.nc")["basin"]
pacific_mask = pacific_mask0.where(pacific_mask0["lat"] >= -32)
pacific_mask = process_mask(pacific_mask)
indian_ocean_mask0 = xr.open_dataset("/pscratch/sd/s/suryad/data/basin_In.nc")["basin"]
indian_ocean_mask = indian_ocean_mask0.where(indian_ocean_mask0["lat"] >= -32)
indian_ocean_mask = process_mask(indian_ocean_mask)
southern_ocean_mask0 = xr.open_dataset("/pscratch/sd/s/suryad/data/basin_SO_32S.nc")[
    "basin"
]
southern_ocean_mask = process_mask(southern_ocean_mask0)
arctic_mask0 = xr.open_dataset("/pscratch/sd/s/suryad/data/basin_Arctic.nc")["basin"]
arctic_ocean_mask = process_mask(arctic_mask0)

extent = get_mask_extent(atlantic_mask)
print("Atlantic:", extent)
extent = get_mask_extent(pacific_mask)
print("Pacific:", extent)
extent = get_mask_extent(southern_ocean_mask)
print("Southern:", extent)
extent = get_mask_extent(indian_ocean_mask)
print("Indian:", extent)
extent = get_mask_extent(arctic_ocean_mask)
print("Arctic:", extent)


import matplotlib.patches as mpatches
import cartopy.crs as ccrs

plt.clf()
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(12, 6))
southern_ocean_mask.plot(ax=ax, cmap="Reds", add_colorbar=False, alpha=0.9)
pacific_mask.plot(ax=ax, cmap="Blues", add_colorbar=False, alpha=0.7)
indian_ocean_mask.plot(ax=ax, cmap="Greens", add_colorbar=False, alpha=0.7)
atlantic_mask.plot(ax=ax, cmap="Purples", add_colorbar=False, alpha=0.7)
arctic_ocean_mask.plot(ax=ax, cmap="copper", add_colorbar=False, alpha=0.7)

# Add coastlines
ax.coastlines(resolution="110m", linewidth=0.5)

# Add gridlines
gl = ax.gridlines(
    draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
)
gl.xlabel_style = {"size": 10, "color": "black"}
gl.ylabel_style = {"size": 10, "color": "black"}


# Create custom legend
legend_handles = [
    mpatches.Patch(color="blue", alpha=0.2, label="Pacific"),
    mpatches.Patch(color="red", alpha=0.4, label="Southern"),
    mpatches.Patch(color="green", alpha=0.4, label="Indian"),
    mpatches.Patch(color="purple", alpha=0.4, label="Atlantic"),
    mpatches.Patch(color="brown", alpha=0.4, label="Arctic"),
]

# Add legend to the plot
ax.legend(handles=legend_handles, loc="lower left", title="Regions")


plt.title("Basin Masks (Ray's Masks)")
plt.show()
plt.close()


# In[25]:


# Compute Basin Heat Content Time Series

f = open(os.path.join(output_path, "compare_info.txt"), "a")
masks = xr.Dataset(
    {
        "Atlantic": atlantic_mask,
        "Pacific": pacific_mask,
        "Southern": southern_ocean_mask,
        "Indian": indian_ocean_mask,
        "Arctic": arctic_ocean_mask,
    }
)

c_p = 3850  # J/(kg C)
rho_0 = 1025  # kg/m^3

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(
    2,
    3,
    figsize=(16, 8),
    gridspec_kw={
        "width_ratios": [1, 1, 1],
        "height_ratios": [1, 1],
        "wspace": 0.25,
        "hspace": 0.5,
    },
)

ax_flat = ax.flatten()

GT_regionwise_ohc = {}
GT_regionwise_ohc["Model"] = dataset_name
for j, k in enumerate(pred_dict.keys()):
    pred_dict[k]["regionwise_ohc"] = {}

for i, var in enumerate(list(masks.keys())):
    OHC = (
        (data["thetao"] * c_p * rho_0 * masks[var]) * data["areacello"] * data["dz"]
    ).sum(["x", "y", "lev"]) / 1e21

    OHC = remove_climatology(OHC)
    OHC = OHC.rename("OHC Anomaly")
    OHC = OHC.assign_attrs(units="ZJ")
    coeffs_OHC_trend = np.polyfit(np.arange(OHC.size), OHC, 1)
    OHC.plot(ax=ax_flat[i], label=dataset_name, c="k")
    (pos,) = ax_flat[i].plot(
        OHC.time.data,
        np.arange(OHC.size) * coeffs_OHC_trend[0] + coeffs_OHC_trend[1],
        c="k",
        ls="--",
    )
    # ax_flat[i].annotate(f'{coeffs_OHC_trend[0]:.2e}',
    #          xy=(pos.get_xdata()[0], pos.get_ydata()[0]),
    #          xytext=(pos.get_xdata()[1], pos.get_ydata()[1]),
    #          fontsize=9, color='k')
    f.write(f"\nOHC {var} GT Trend Slope : {coeffs_OHC_trend[0]}")
    GT_regionwise_ohc[var] = coeffs_OHC_trend[0]
    for j, k in enumerate(pred_dict.keys()):
        OHC_pred = (
            (pred_dict[k]["ds_prediction"]["thetao"] * c_p * rho_0 * masks[var])
            * pred_dict[k]["ds_prediction"]["areacello"]
            * pred_dict[k]["ds_prediction"]["dz"]
        ).sum(["x", "y", "lev"]) / 1e21

        OHC_pred = remove_climatology(OHC_pred)
        OHC_pred = OHC_pred.rename("OHC Anomaly")
        OHC_pred = OHC_pred.assign_attrs(units="ZJ")
        coeffs_OHC_pred_trend = np.polyfit(np.arange(OHC_pred.size), OHC_pred, 1)
        OHC_pred.plot(ax=ax_flat[i], label=pred_dict[k]["name"], c=clist[j])
        (pos,) = ax_flat[i].plot(
            OHC_pred.time.data,
            np.arange(OHC_pred.size) * coeffs_OHC_pred_trend[0]
            + coeffs_OHC_pred_trend[1],
            c=clist[j],
            ls="--",
        )
        # ax_flat[i].annotate(f'{coeffs_OHC_pred_trend[0]:.2e}',
        #      xy=(pos.get_xdata()[-1], pos.get_ydata()[-1]),
        #      xytext=(pos.get_xdata()[-2], pos.get_ydata()[-2]),
        #      fontsize=9, color=clist[j])
        f.write(
            f"\nOHC {var} {pred_dict[k]['name']} Trend Slope : {coeffs_OHC_pred_trend[0]}"
        )
        pred_dict[k]["regionwise_ohc"][var] = coeffs_OHC_pred_trend[0]

    ax_flat[i].set_title(var + " Ocean")

fig.delaxes(ax_flat[5])
handles, labels = ax_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncols=3)

f.write("\n")
f.close()
# plt.show()
plt.savefig(os.path.join(ohc_path, "OHC_Basin"), bbox_inches="tight", dpi=600)


# In[26]:


pd_data = []
pd_data.append(GT_regionwise_ohc)


for k in pred_dict.keys():
    d = {}
    d["Model"] = pred_dict[k]["name"]
    for var in masks.keys():
        d[var] = pred_dict[k]["regionwise_ohc"][var]
        d[var + " Slope Ratio"] = (
            pred_dict[k]["regionwise_ohc"][var] / GT_regionwise_ohc[var]
        )
    pd_data.append(d)

# Create a DataFrame
df = pd.DataFrame(pd_data)

# Define the file path
file_path = os.path.join(ohc_path, "regionwise_ohc_slopes_table.csv")

# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)


# In[27]:


# Compute Basin Heat Content Time Series

f = open(os.path.join(output_path, "compare_info.txt"), "a")
masks = xr.Dataset(
    {
        "Atlantic": atlantic_mask,
        "Pacific": pacific_mask,
        "Southern": southern_ocean_mask,
        "Indian": indian_ocean_mask,
        "Arctic": arctic_ocean_mask,
    }
)

c_p = 3850  # J/(kg C)
rho_0 = 1025  # kg/m^3
max_level = 700

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(
    2,
    3,
    figsize=(16, 8),
    gridspec_kw={
        "width_ratios": [1, 1, 1],
        "height_ratios": [1, 1],
        "wspace": 0.25,
        "hspace": 0.5,
    },
)

ax_flat = ax.flatten()

GT_regionwise_ohc = {}
GT_regionwise_ohc["Model"] = dataset_name
for j, k in enumerate(pred_dict.keys()):
    pred_dict[k]["regionwise_ohc"] = {}

for i, var in enumerate(list(masks.keys())):
    OHC = (
        (data["thetao"].sel(lev=slice(None, max_level)) * c_p * rho_0 * masks[var])
        * data["areacello"]
        * data["dz"]
    ).sum(["x", "y", "lev"]) / 1e21

    OHC = remove_climatology(OHC)
    OHC = OHC.rename(f"OHC Anomaly (Upto {max_level}m)")
    OHC = OHC.assign_attrs(units="ZJ")
    coeffs_OHC_trend = np.polyfit(np.arange(OHC.size), OHC, 1)
    OHC.plot(ax=ax_flat[i], label=dataset_name, c="k")
    (pos,) = ax_flat[i].plot(
        OHC.time.data,
        np.arange(OHC.size) * coeffs_OHC_trend[0] + coeffs_OHC_trend[1],
        c="k",
        ls="--",
    )
    # ax_flat[i].annotate(f'{coeffs_OHC_trend[0]:.2e}',
    #          xy=(pos.get_xdata()[0], pos.get_ydata()[0]),
    #          xytext=(pos.get_xdata()[1], pos.get_ydata()[1]),
    #          fontsize=9, color='k')
    f.write(f"\nOHC {var} GT Trend Slope : {coeffs_OHC_trend[0]}")
    GT_regionwise_ohc[var] = coeffs_OHC_trend[0]
    for j, k in enumerate(pred_dict.keys()):
        OHC_pred = (
            (
                pred_dict[k]["ds_prediction"]["thetao"].sel(lev=slice(None, max_level))
                * c_p
                * rho_0
                * masks[var]
            )
            * pred_dict[k]["ds_prediction"]["areacello"]
            * pred_dict[k]["ds_prediction"]["dz"]
        ).sum(["x", "y", "lev"]) / 1e21

        OHC_pred = remove_climatology(OHC_pred)
        OHC_pred = OHC_pred.rename(f"OHC Anomaly (Upto {max_level}m)")
        OHC_pred = OHC_pred.assign_attrs(units="ZJ")
        coeffs_OHC_pred_trend = np.polyfit(np.arange(OHC_pred.size), OHC_pred, 1)
        OHC_pred.plot(ax=ax_flat[i], label=pred_dict[k]["name"], c=clist[j])
        (pos,) = ax_flat[i].plot(
            OHC_pred.time.data,
            np.arange(OHC_pred.size) * coeffs_OHC_pred_trend[0]
            + coeffs_OHC_pred_trend[1],
            c=clist[j],
            ls="--",
        )
        # ax_flat[i].annotate(f'{coeffs_OHC_pred_trend[0]:.2e}',
        #      xy=(pos.get_xdata()[-1], pos.get_ydata()[-1]),
        #      xytext=(pos.get_xdata()[-2], pos.get_ydata()[-2]),
        #      fontsize=9, color=clist[j])
        f.write(
            f"\nOHC {var} {pred_dict[k]['name']} Trend Slope : {coeffs_OHC_pred_trend[0]}"
        )
        pred_dict[k]["regionwise_ohc"][var] = coeffs_OHC_pred_trend[0]

    ax_flat[i].set_title(var + " Ocean")

fig.delaxes(ax_flat[5])
handles, labels = ax_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=3)

f.write("\n")
f.close()
# plt.show()
plt.savefig(
    os.path.join(ohc_path, f"OHC_Basin_upto_{max_level}m"), bbox_inches="tight", dpi=600
)


# ### Temperature Plots

# In[28]:


import os

import matplotlib.pyplot as plt
import numpy as np

plt.clf()
plt.rcParams.update({"font.size": 14})

# Define colormap
new_cmap = cm.cm.thermal
new_cmap.set_bad("grey", 0.6)


num_basins = 1
num_models = len(pred_dict) + 1  # Including GT

# Create figure with appropriate layout
fig, ax = plt.subplots(
    num_basins,
    num_models,
    figsize=(16, 3),
    gridspec_kw={
        "width_ratios": [1] * num_models,
        "height_ratios": [1] * num_basins,
        "wspace": 0.02,
        "hspace": 0.5,
    },
)
ax = np.array(ax)  # Ensure ax is an array for easy indexing

# Set common color range for the colorbar
vmin, vmax = 0, 30

# Plot GT (original data)
da_temp = data["thetao"]  # Directly use temperature variable
section_mask = np.isnan(da_temp).all("x").isel(time=0)
da_temp_int_x = da_temp.weighted(data["areacello"]).mean(["x", "time"])
temp_pred = da_temp_int_x.where(~section_mask)
temp_pred = temp_pred.rename(r"$\theta_O$").assign_attrs(units=r"$\degree C$")
temp_pred["y"] = temp_pred.y.assign_attrs(long_name="latitude", units=r"$\degree$")
temp_pred["lev"] = temp_pred.lev.assign_attrs(long_name="depth", units="m")

im = temp_pred.plot(ax=ax[0], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
GT_temp_pred = temp_pred
ax[0].invert_yaxis()
ax[0].set_title(dataset_name, fontsize=14)
linear_piecewise_scale(1000, 5, ax=ax[0])
ax[0].axhline(1000, color="0.5", ls="--")
ax[0].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])

# Plot predictions from other models
for j, (model_key, model_data) in enumerate(pred_dict.items(), start=1):
    da_temp = model_data["ds_prediction"]["thetao"]  # Use temperature variable
    section_mask = np.isnan(da_temp).all("x").isel(time=0)
    da_temp_int_x = da_temp.weighted(data["areacello"]).mean(["x", "time"])
    temp_pred = da_temp_int_x.where(~section_mask)
    temp_pred = temp_pred.rename(r"$\theta_O$").assign_attrs(units=r"$\degree C$")
    temp_pred["y"] = temp_pred.y.assign_attrs(long_name="latitude", units=r"$\degree$")
    temp_pred["lev"] = temp_pred.lev.assign_attrs(long_name="depth", units="m")
    pred_dict[model_key]["temp_profile"] = temp_pred

    im = temp_pred.plot(
        ax=ax[j], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False
    )
    ax[j].invert_yaxis()
    ax[j].set_title(f"{pred_dict[model_key]['name']}", fontsize=14)
    linear_piecewise_scale(1000, 5, ax=ax[j])
    ax[j].axhline(1000, color="0.5", ls="--")
    # ax[j].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
    ax[j].set_yticks([])
    ax[j].set_ylabel("")

# Add shared colorbar for each row
cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label(r"$\theta_O$ [$\degree C$]")

# plt.show()
plt.savefig(
    os.path.join(temp_path, "Temperature_Global_Profile"),
    bbox_inches="tight",
    dpi=600,
)
# plt.close()


# In[29]:


try:
    new_cmap = cm.cm.balance
    new_cmap.set_bad("grey", 0.6)
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(16, 3),
        gridspec_kw={
            "width_ratios": [1, 1, 1],
            "height_ratios": [1],
            "wspace": 0.02,
            "hspace": 0.5,
        },
    )
    vmin, vmax = -0.5, 0.5
    for i, key in enumerate(pred_dict):
        im = (pred_dict[key]["temp_profile"] - GT_temp_pred).plot(
            ax=ax[i], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False
        )
        ax[i].invert_yaxis()
        ax[i].set_title(f"{pred_dict[key]['name']} - GT", fontsize=14)
        linear_piecewise_scale(1000, 5, ax=ax[i])
        ax[i].axhline(1000, color="0.5", ls="--")
        if i == 0:
            ax[i].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
        else:
            ax[i].set_yticks([])
            ax[i].set_ylabel("")

    i = i + 1
    im = (
        pred_dict["pred_1"]["temp_profile"] - pred_dict["pred_2"]["temp_profile"]
    ).plot(ax=ax[i], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    ax[i].invert_yaxis()
    ax[i].set_title(
        f"{pred_dict['pred_1']['name']} - {pred_dict['pred_2']['name']}", fontsize=14
    )
    linear_piecewise_scale(1000, 5, ax=ax[i])
    ax[i].axhline(1000, color="0.5", ls="--")
    ax[i].set_yticks([])
    ax[i].set_ylabel("")
    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(r"$\theta_O$ [$\degree C$]")

    plt.savefig(
        os.path.join(temp_path, "Temperature_Diff_Global_Profile"),
        bbox_inches="tight",
        dpi=600,
    )
except:
    pass


# ### Salinity Plots

# In[30]:


import os

import cmocean as cm
import matplotlib.pyplot as plt
import numpy as np

plt.clf()
plt.rcParams.update({"font.size": 14})

# Define colormap
new_cmap = cm.cm.haline  # Salinity-specific colormap from cmocean
new_cmap.set_bad("grey", 0.6)

num_basins = 1
num_models = len(pred_dict) + 1  # Including GT

# Create figure with appropriate layout
fig, ax = plt.subplots(
    num_basins,
    num_models,
    figsize=(16, 3),
    gridspec_kw={
        "width_ratios": [1] * num_models,
        "height_ratios": [1] * num_basins,
        "wspace": 0.02,
        "hspace": 0.5,
    },
)
ax = np.array(ax)  # Ensure ax is an array for easy indexing

# Set common color range for the colorbar
vmin, vmax = 33, 36  # Typical salinity range, adjust as needed

# Plot GT (original data)
da_salinity = data["so"]  # Replace with salinity variable
section_mask = np.isnan(da_salinity).all("x").isel(time=0)
da_salinity_int_x = da_salinity.weighted(data["areacello"]).mean(["x", "time"])
salinity_pred = da_salinity_int_x.where(~section_mask)
salinity_pred = salinity_pred.rename(r"$S$").assign_attrs(units="psu")  # Salinity units
salinity_pred["y"] = salinity_pred.y.assign_attrs(
    long_name="latitude", units=r"$\degree$"
)
salinity_pred["lev"] = salinity_pred.lev.assign_attrs(long_name="depth", units="m")

im = salinity_pred.plot(
    ax=ax[0], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False
)
GT_salinity_pred = salinity_pred
ax[0].invert_yaxis()
ax[0].set_title(dataset_name, fontsize=14)
linear_piecewise_scale(1000, 5, ax=ax[0])
ax[0].axhline(1000, color="0.5", ls="--")
ax[0].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])

# Plot predictions from other models
for j, (model_key, model_data) in enumerate(pred_dict.items(), start=1):
    da_salinity = model_data["ds_prediction"]["so"]  # Replace with salinity variable
    section_mask = np.isnan(da_salinity).all("x").isel(time=0)
    da_salinity_int_x = da_salinity.weighted(data["areacello"]).mean(["x", "time"])
    salinity_pred = da_salinity_int_x.where(~section_mask)
    salinity_pred = salinity_pred.rename(r"$S$").assign_attrs(units="psu")
    salinity_pred["y"] = salinity_pred.y.assign_attrs(
        long_name="latitude", units=r"$\degree$"
    )
    salinity_pred["lev"] = salinity_pred.lev.assign_attrs(long_name="depth", units="m")
    pred_dict[model_key]["salinity_profile"] = salinity_pred

    im = salinity_pred.plot(
        ax=ax[j], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False
    )
    ax[j].invert_yaxis()
    ax[j].set_title(f"{pred_dict[model_key]['name']}", fontsize=14)
    linear_piecewise_scale(1000, 5, ax=ax[j])
    ax[j].axhline(1000, color="0.5", ls="--")
    if j == 0:
        ax[j].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
    else:
        ax[j].set_yticks([])
        ax[j].set_ylabel("")

# Add shared colorbar for each row
cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label(r"$S$ [psu]")  # Salinity colorbar label in psu

# Save the figure
# plt.show()
plt.savefig(
    os.path.join(salinity_path, "Salinity_Global_Profile"), bbox_inches="tight", dpi=600
)


# In[31]:


try:
    import cmocean as cm
    import matplotlib.pyplot as plt

    new_cmap = cm.cm.delta
    new_cmap.set_bad("grey", 0.6)
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(16, 3),
        gridspec_kw={
            "width_ratios": [1, 1, 1],
            "height_ratios": [1],
            "wspace": 0.02,
            "hspace": 0.5,
        },
    )
    vmin, vmax = -0.05, 0.05

    i = 0

    im = (pred_dict["pred_1"]["salinity_profile"] - GT_salinity_pred).plot(
        ax=ax[i], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False
    )
    ax[i].invert_yaxis()
    ax[i].set_title(f"{pred_dict['pred_1']['name']} - GT", fontsize=14)
    linear_piecewise_scale(1000, 5, ax=ax[i])
    ax[i].axhline(1000, color="0.5", ls="--")
    ax[i].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])

    i = i + 1
    im = (pred_dict["pred_2"]["salinity_profile"] - GT_salinity_pred).plot(
        ax=ax[i], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False
    )
    ax[i].invert_yaxis()
    ax[i].set_title(f"{pred_dict['pred_2']['name']} - GT", fontsize=14)
    linear_piecewise_scale(1000, 5, ax=ax[i])
    ax[i].axhline(1000, color="0.5", ls="--")
    # ax[i].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
    ax[i].set_yticks([])
    ax[i].set_ylabel("")

    i = i + 1
    im = (
        pred_dict["pred_1"]["salinity_profile"]
        - pred_dict["pred_2"]["salinity_profile"]
    ).plot(ax=ax[i], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
    ax[i].invert_yaxis()
    ax[i].set_title(
        f"{pred_dict['pred_1']['name']} - {pred_dict['pred_2']['name']}", fontsize=14
    )
    linear_piecewise_scale(1000, 5, ax=ax[i])
    ax[i].axhline(1000, color="0.5", ls="--")
    # ax[i].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
    ax[i].set_yticks([])
    ax[i].set_ylabel("")

    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(r"S [psu]")

    plt.savefig(
        os.path.join(salinity_path, "Salinity_Diff_Global_Profile"),
        bbox_inches="tight",
        dpi=600,
    )
except:
    pass


# In[32]:


rho_0 = 1025  # kg/m^3
f = open(os.path.join(output_path, "compare_info.txt"), "a")

plt.rcdefaults()
fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # Single axis for salinity plot
plt.rcParams.update({"font.size": 9})

salinity = ((data["so"] * rho_0) * data["areacello"] * data["dz"]).sum(
    ["x", "y", "lev"]
)
salinity = salinity.rename("Salinity")
salinity = salinity.assign_attrs(units="g")

for i, k in enumerate(pred_dict.keys()):
    if "so" in pred_dict[k]["ls"]:
        salinity_pred = (
            (pred_dict[k]["ds_prediction"]["so"] * rho_0)
            * pred_dict[k]["ds_prediction"]["areacello"]
            * pred_dict[k]["ds_prediction"]["dz"]
        ).sum(["x", "y", "lev"])
        salinity_pred = salinity_pred.rename("Salinity")
        salinity_pred = salinity_pred.assign_attrs(units="g")
        salinity_pred.plot(ax=ax, label=pred_dict[k]["name"], c=clist[i])
        coeffs_salinity_pred_trend = np.polyfit(
            np.arange(salinity_pred.size), salinity_pred, 1
        )
        (pos,) = ax.plot(
            salinity_pred.time.data,
            np.arange(salinity_pred.size) * coeffs_salinity_pred_trend[0]
            + coeffs_salinity_pred_trend[1],
            c=clist[i],
            ls="--",
        )
        f.write(
            f"\nSalinity {pred_dict[k]['name']} Trend Slope : {coeffs_salinity_pred_trend[0]}"
        )
        pred_dict[k]["salinity_slope"] = coeffs_salinity_pred_trend[0]

coeffs_salinity_trend = np.polyfit(np.arange(salinity.size), salinity, 1)
salinity.plot(ax=ax, label=dataset_name, c="k")
(pos,) = ax.plot(
    salinity.time.data,
    np.arange(salinity.size) * coeffs_salinity_trend[0] + coeffs_salinity_trend[1],
    c="k",
    ls="--",
)
f.write(f"\nSalinity GT Trend Slope : {coeffs_salinity_trend[0]}")
GT_salinity_slope = coeffs_salinity_trend[0]
ax.set_ylim([5.861e22, 5.8632e22])
ax.legend(ncol=3)
f.write("\n")
f.close()

print(coeffs_salinity_trend[0] * 73)
plt.savefig(os.path.join(salinity_path, "Salinity"), bbox_inches="tight", dpi=600)


# In[33]:


pd_data = []
pd_data.append(
    {
        "Model": dataset_name,
        "OHC": GT_ohc_slope,
        "Salinity": GT_salinity_slope,
    }
)

for k in pred_dict.keys():
    pd_data.append(
        {
            "Model": pred_dict[k]["name"],
            "OHC": pred_dict[k]["OHC_slope"],
            "OHC Slope Ratio": pred_dict[k]["OHC_slope"] / GT_ohc_slope,
            "Salinity": pred_dict[k]["salinity_slope"],
            "Salinity Slope Ratio": pred_dict[k]["salinity_slope"] / GT_salinity_slope,
        }
    )

# Create a DataFrame
df = pd.DataFrame(pd_data)

# Define the file path
file_path = os.path.join(output_path, "ohc_salinity_slopes_table.csv")

# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)


# ### Metrics

# #### Deseasonalized Salinity Trend

# In[34]:


f = open(os.path.join(metrics_path, "salinity_deseasonalized_info.txt"), "a")

plt.rcdefaults()
fig, ax = plt.subplots(
    1, 1, figsize=(10, 3), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
)
plt.rcParams.update({"font.size": 9})

salinity = data["so"].weighted(data["areacello"] * data["dz"]).mean(["x", "y", "lev"])

salinity = remove_climatology(salinity)
salinity = salinity.rename("Salinity")
salinity = salinity.assign_attrs(units="psu")

for i, k in enumerate(pred_dict.keys()):
    if "so" in pred_dict[k]["ls"]:
        salinity_pred = (
            pred_dict[k]["ds_prediction"]["so"]
            .weighted(
                pred_dict[k]["ds_prediction"]["areacello"]
                * pred_dict[k]["ds_prediction"]["dz"]
            )
            .mean(["x", "y", "lev"])
        )
        salinity_pred = remove_climatology(salinity_pred)
        salinity_pred = salinity_pred.rename("Salinity")
        salinity_pred = salinity_pred.assign_attrs(units="psu")
        salinity_pred.plot(ax=ax, label=pred_dict[k]["name"], c=clist[i])
        coeffs_salinity_pred_trend = np.polyfit(
            np.arange(salinity_pred.size), salinity_pred, 1
        )
        (pos,) = ax.plot(
            salinity_pred.time.data,
            np.arange(salinity_pred.size) * coeffs_salinity_pred_trend[0]
            + coeffs_salinity_pred_trend[1],
            c=clist[i],
            ls="--",
        )
        # ax[1].annotate(f'{coeffs_salinity_pred_trend[0]:.2e}',
        #          xy=(pos.get_xdata()[-1], pos.get_ydata()[-1]),
        #          xytext=(pos.get_xdata()[-2], pos.get_ydata()[-2]),
        #          fontsize=9, color=clist[i])
        f.write(
            f"\nSalinity {pred_dict[k]['name']} Trend Slope : {coeffs_salinity_pred_trend[0]}"
        )
        pred_dict[k]["salinity_slope"] = coeffs_salinity_pred_trend[0]


coeffs_salinity_trend = np.polyfit(np.arange(salinity.size), salinity, 1)
salinity.plot(ax=ax, label=dataset_name, c="k")
(pos,) = ax.plot(
    salinity.time.data,
    np.arange(salinity.size) * coeffs_salinity_trend[0] + coeffs_salinity_trend[1],
    c="k",
    ls="--",
)
# ax[1].annotate(f'{coeffs_salinity_trend[0]:.2e}',
#              xy=(pos.get_xdata()[0], pos.get_ydata()[0]),
#              xytext=(pos.get_xdata()[1], pos.get_ydata()[1]),
#              fontsize=9, color='k')
f.write(f"\nSalinity GT Trend Slope : {coeffs_salinity_trend[0]}")
GT_salinity_slope = coeffs_salinity_trend[0]
f.write("\n")
f.close()

print(coeffs_salinity_trend[0] * 73)
plt.savefig(
    os.path.join(salinity_path, "salinity_deseasonalized"), bbox_inches="tight", dpi=600
)


# #### Thetao MAE

# In[35]:


da_temp = data["thetao"]  # Directly use temperature variable
section_mask = np.isnan(da_temp).all("x").isel(time=0)
da_temp_int_x = da_temp.weighted(data["areacello"]).mean(["x", "time"])
temp_pred = da_temp_int_x.where(~section_mask)
GT_temp_pred = temp_pred

for j, (model_key, model_data) in enumerate(pred_dict.items(), start=1):
    da_temp = model_data["ds_prediction"]["thetao"]  # Use temperature variable
    section_mask = np.isnan(da_temp).all("x").isel(time=0)
    da_temp_int_x = da_temp.weighted(data["areacello"]).mean(["x", "time"])
    temp_pred = da_temp_int_x.where(~section_mask)
    pred_dict[model_key]["temp_profile"] = temp_pred

f = open(os.path.join(metrics_path, "thetao_mae_info.txt"), "a")

for i, key in enumerate(pred_dict):
    mae_key = np.abs((pred_dict[key]["temp_profile"] - GT_temp_pred)).mean().compute()
    f.write(f"\n Thetao {pred_dict[key]['name']} MAE : {mae_key.item()}")

f.close()


# #### SST MAE

# In[36]:


section_mask = np.isnan(data["thetao"]).isel(lev=0).isel(time=5)
SST_gt = data["thetao"].isel(lev=0).mean("time")
SST_gt = SST_gt.where(~section_mask)

for j, (model_key, model_data) in enumerate(pred_dict.items(), start=1):
    section_mask = (
        np.isnan(model_data["ds_prediction"]["thetao"]).isel(lev=0).isel(time=5)
    )
    SST_pred = model_data["ds_prediction"]["thetao"].isel(lev=0).mean("time")
    SST_pred = SST_pred.where(~section_mask)
    pred_dict[model_key]["sst"] = SST_pred

f = open(os.path.join(metrics_path, "sst_mae_info.txt"), "a")

for i, key in enumerate(pred_dict):
    mae_key = np.abs((pred_dict[key]["sst"] - SST_gt)).mean().compute()
    f.write(f"\n SST {pred_dict[key]['name']} MAE : {mae_key.item()}")

f.close()


# #### Drake Passage / Atlantic MAE CORR

# In[37]:


# # Drake Passage Full Depth
# day_start = -103
# window = 3
# regions = {
#     "Drake Passage": {"lon": 290, "lat_bnds": slice(-70, -55)},
#     "Atlantic": {"lon": 330, "lat_bnds": slice(-80, 90)},
# }
# surface = False

# N_days = 100
# for i, region in enumerate(["Drake Passage", "Atlantic"]):
#     for k in pred_dict.keys():
#         pred_dict[k][region] = {"mae_mean": 0, "cor_mean": 0}
#         for j in range(N_days):

#             bounds = regions[region]
#             var = "thetao"
#             if surface:
#                 level_slice = slice(0, 1000)
#             else:
#                 level_slice = slice(None)

#             depth_slice = (
#                 data[var]
#                 .sel(x=bounds["lon"], method="nearest")
#                 .sel(y=bounds["lat_bnds"], lev=level_slice)
#                 .isel(time=slice(day_start, day_start + window))
#                 .mean("time")
#             )
#             wet = np.array(xr.where(np.isnan(depth_slice), False, True))
#             area = (
#                 data["areacello"]
#                 .sel(x=bounds["lon"], method="nearest")
#                 .sel(y=bounds["lat_bnds"])
#                 * data["dz"]
#             )
#             area = area.values.transpose()[wet]

#             depth_slice_pred = (
#                 pred_dict[k]["ds_prediction"][var]
#                 .sel(x=bounds["lon"], method="nearest")
#                 .sel(y=bounds["lat_bnds"], lev=level_slice)
#                 .isel(time=slice(day_start, day_start + window))
#                 .mean("time")
#             )
#             mae = np.abs(depth_slice_pred - depth_slice)
#             mae = (
#                 mae
#                 * data["areacello"]
#                 .sel(x=bounds["lon"], method="nearest")
#                 .sel(y=bounds["lat_bnds"])
#                 * data["dz"]
#             ).sum(["y", "lev"]) / (
#                 data["areacello"]
#                 .sel(x=bounds["lon"], method="nearest")
#                 .sel(y=bounds["lat_bnds"])
#                 * data["dz"]
#             ).sum()
#             mae = mae.assign_attrs(long_name="MAE", units=r"${^oC}$")
#             pred_dict[k][region]["mae_mean"] += 1 / N_days * mae.values
#             cor = (
#                 area
#                 * depth_slice_pred.values[wet].flatten()
#                 * depth_slice.values[wet].flatten()
#             ).sum() / np.sqrt(
#                 (area * depth_slice_pred.values[wet].flatten() ** 2).sum()
#                 * (area * depth_slice.values[wet].flatten() ** 2).sum()
#             )
#             pred_dict[k][region]["cor_mean"] += 1 / N_days * cor


# In[38]:


# for k in pred_dict.keys():
#     print(pred_dict[k]["Drake Passage"])
#     print(pred_dict[k]["Atlantic"])

# # Create a list of dictionaries for the DataFrame
# pd_data = []
# for k in pred_dict.keys():
#     drake = pred_dict[k]["Drake Passage"]
#     atlantic = pred_dict[k]["Atlantic"]
#     pd_data.append(
#         {
#             "Prediction": pred_dict[k]["name"],
#             "Drake_Passage_MAE": drake["mae_mean"],
#             "Drake_Passage_COR": drake["cor_mean"],
#             "Atlantic_MAE": atlantic["mae_mean"],
#             "Atlantic_COR": atlantic["cor_mean"],
#         }
#     )

# # Create a DataFrame
# df = pd.DataFrame(pd_data)

# # Define the file path
# file_path = os.path.join(metrics_path, "drake_atlantic_predictions_table.csv")

# # Save the DataFrame to a CSV file
# df.to_csv(file_path, index=False)


# ### PDFs

# In[39]:


plt.rcParams.update({"font.size": 9})

for v in ["uo", "vo", "thetao", "so", "zos"]:
    print("v: ", v)
    plt.clf()
    plt.rcParams.update({"font.size": 18})
    plt.figure(figsize=[8, 6])
    min_val, max_val = ds_groundtruth[v].min().values, ds_groundtruth[v].max().values
    true_pdf, bins_true = np.histogram(
        ds_groundtruth[v], bins=150, density=True, range=(min_val, max_val)
    )

    for i, k in enumerate(pred_dict.keys()):
        if v in pred_dict[k]["ls"]:
            pdf_net, bins_net = np.histogram(
                pred_dict[k]["ds_prediction"][v],
                bins=bins_true,
                density=True,
                range=(min_val, max_val),
            )
            plt.semilogy(
                bins_net[:-1], pdf_net, label=pred_dict[k]["name"], c=clist[i], lw=2
            )

    plt.semilogy(bins_true[:-1], true_pdf, label=dataset_name, color="k", lw=2, ls="--")
    plt.legend()
    plt.xlabel(var_list[v])
    plt.ylabel(r"${p(}$" + var_list[v].split(" $")[0] + "${)}$")
    if v != "thetao":
        plt.ylim(
            [
                true_pdf.min(),
                true_pdf.max(),
            ]
        )
        if v == "KE":
            plt.xlim([0, 2500])
    else:
        plt.ylim(
            [
                0.01,
                true_pdf.max(),
            ]
        )
        plt.xlim([-2, 32])
    plt.savefig(os.path.join(pdfs_path, f"{v}.png"), bbox_inches="tight", dpi=600)
    # plt.show()


# In[40]:


import matplotlib
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({"font.size": 9})
# Create a figure
fig = plt.figure(figsize=(24, 15))
plt.rc("axes", titlesize=30)  # fontsize of the axes title
plt.rc("axes", labelsize=30)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=30)  # fontsize of the tick labels
plt.rc("ytick", labelsize=30)  # fontsize of the tick labels
plt.rc("legend", fontsize=20)  # legend fontsize
plt.rc("figure", titlesize=30)
# Manual positioning using add_axes with uniform width and height
width = 0.22
height = 0.3

# Top row: 3 plots, evenly spaced horizontally
axs = [
    fig.add_axes([0.05, 0.55, width, height]),  # First plot in top row
    fig.add_axes([0.38, 0.55, width, height]),  # Second plot in top row
    fig.add_axes([0.71, 0.55, width, height]),  # Third plot in top row
]

# Bottom row: 2 plots centered, manually positioned
axs += [
    fig.add_axes([0.22, 0.1, width, height]),  # First plot in bottom row
    fig.add_axes([0.54, 0.1, width, height]),  # Second plot in bottom row
]

# Plot PDFs
for i, v in enumerate(["thetao", "so", "zos", "uo", "vo"]):
    min_val, max_val = ds_groundtruth[v].min().values, ds_groundtruth[v].max().values
    true_pdf, bins_true = np.histogram(
        data[v], bins=150, density=True, range=(min_val, max_val)
    )
    axs[i].semilogy(bins_true[:-1], true_pdf, label=dataset_name, color="k", lw=8)

    for j, k in enumerate(pred_dict.keys()):
        if v in pred_dict[k]["ls"]:
            pdf_net, bins_net = np.histogram(
                pred_dict[k]["ds_prediction"][v],
                bins=150,
                density=True,
                range=(min_val, max_val),
            )
            axs[i].semilogy(
                bins_net[:-1], pdf_net, label=pred_dict[k]["name"], color=clist[j], lw=2
            )

    axs[i].xaxis.set_major_locator(MaxNLocator(5, prune="both"))
    if i == 0:
        axs[i].legend()
    axs[i].set_xlabel(r"" + data[v].long_name + "[" + data[v].units + "]")
    axs[i].set_ylabel(r"${p(}$" + data[v].long_name + " " + "${)}$")

    if v not in ["thetao", "SSH"]:
        axs[i].set_ylim([min(true_pdf.min(), pdf_net.min()) + 1e-5, true_pdf.max()])
    else:
        axs[i].set_ylim([1e-3, true_pdf.max()])
matplotlib.style.use("default")

# Save or show the figure
# plt.show()
plt.savefig(os.path.join(pdfs_path, "PDF_Plots_Short"), bbox_inches="tight", dpi=600)


# ### ENSO

# In[41]:


clim = data["thetao"].sel(lev=slice(0, 500)).groupby("time.dayofyear").mean().compute()
data_surface = data.sel(lev=slice(0, 500))
for k in pred_dict.keys():
    pred_dict[k]["ds_prediction_surface"] = pred_dict[k]["ds_prediction"].sel(
        lev=slice(0, 500)
    )
    pred_dict[k]["clim_pred"] = (
        pred_dict[k]["ds_prediction_surface"]["thetao"]
        .groupby("time.dayofyear")
        .mean()
        .compute()
    )


# In[42]:


def NinoIndexComputeClim(T, area, dt=5, window=150):
    T = T.load()
    T_clim = T.copy()
    T_clim = T_clim.sel(x=slice(190, 240), y=slice(-5, 5))
    area = area.sel(x=slice(190, 240), y=slice(-5, 5)).load()
    clim = T_clim.groupby("time.dayofyear").mean("time").compute()
    window = int(window / dt)
    for i, t in enumerate(T_clim.time.values):
        day = int(t.dayofyr)
        T_clim[i] = (T[i] - clim.sel(dayofyear=day)).data

    T_clim = T_clim.rolling(time=window).mean()
    # T_clim = (T_clim*area).sum(["x","y"])/area.sum(["x","y"])
    T_clim = T_clim.weighted(area).mean(["x", "y"])

    return T_clim[window:]


# In[43]:


nino_true_compute_clim = NinoIndexComputeClim(
    data_surface["thetao"][:, 0], data["areacello"]
)
nino_true_compute_clim = nino_true_compute_clim.rename("Nino 3.4")
nino_true_compute_clim = nino_true_compute_clim.assign_attrs(units=r"$\degree C$")

for k in pred_dict.keys():
    pred_dict[k]["nino_pred_compute_clim"] = NinoIndexComputeClim(
        pred_dict[k]["ds_prediction_surface"]["thetao"][:, 0],
        pred_dict[k]["ds_prediction"]["areacello"],
    )
    pred_dict[k]["nino_pred_compute_clim"] = pred_dict[k][
        "nino_pred_compute_clim"
    ].rename("Nino 3.4")
    pred_dict[k]["nino_pred_compute_clim"] = pred_dict[k][
        "nino_pred_compute_clim"
    ].assign_attrs(units=r"$\degree C$")


# In[44]:


import numpy as np

day_max = int(
    (
        np.argwhere(
            nino_true_compute_clim.values
            == np.nanmax(nino_true_compute_clim.values[5:-5])
        )
        + 30
    ).squeeze()
)
day_min = int(
    (
        np.argwhere(
            nino_true_compute_clim.values
            == np.nanmin(nino_true_compute_clim.values[5:-5])
        )
        + 30
    ).squeeze()
)


# In[45]:


plt.rcParams.update({"font.size": 14})
plt.figure(figsize=[10, 5])
nino_true_compute_clim.plot(label=dataset_name, c="k")
for i, k in enumerate(pred_dict.keys()):
    pred_dict[k]["nino_pred_compute_clim"].plot(label=pred_dict[k]["name"], c=clist[i])

ax = plt.gca()
ax.legend()
ax.set_title("Nino 3.4 Index")

plt.savefig(os.path.join(enso_path, "Climatology"), bbox_inches="tight", dpi=600)


# In[46]:


for k in pred_dict.keys():
    mae = np.abs(
        (pred_dict[k]["nino_pred_compute_clim"] - nino_true_compute_clim).mean(["time"])
    )
    cor = (
        (pred_dict[k]["nino_pred_compute_clim"] * nino_true_compute_clim)
        / np.sqrt(
            (pred_dict[k]["nino_pred_compute_clim"] ** 2) * (nino_true_compute_clim**2)
        )
    ).mean()
    print(pred_dict[k]["name"], "mae: ", mae.item(), "cor: ", cor.item())


# ### Profiles and Maps
#
# This works for two predictions only

# In[47]:


keys = list(pred_dict.keys())
# assert len(keys) >= 2, "Maps supported by atleast two keys"
key1 = keys[0]
if len(keys) == 1:
    key2 = keys[0]
elif len(keys) > 2:
    print("Maps only support two models for now!!! Using the first two keys")
    assert False
else:
    key2 = keys[1]


# #### ENSO Maps

# In[48]:


from xarrayutils.plotting import box_plot

plt.rcParams.update({"font.size": 14})
fig, axs = plt.subplot_mosaic(
    [
        ["time series", "map"],
        ["nino_true", "nina_true"],
        ["nino_pred", "nina_pred"],
        ["nino_pred_temp", "nina_pred_temp"],
        ["colorbar", "colorbar"],
    ],
    figsize=(16, 9),
    per_subplot_kw={"map": dict(projection=ccrs.Robinson(190))},
    gridspec_kw={
        "width_ratios": [1, 1],
        "height_ratios": [0.5, 0.3, 0.3, 0.3, 0.05],
        "wspace": 0.25,
        "hspace": 0.5,
    },
)

############################################
# Time Series
############################################
pred_dict[key1]["nino_pred_compute_clim"].plot(
    label=pred_dict[key1]["name"], c=clist[0], ax=axs["time series"]
)
pred_dict[key2]["nino_pred_compute_clim"].plot(
    label=pred_dict[key2]["name"], c=clist[1], ax=axs["time series"]
)

nino_true_compute_clim.plot(label=dataset_name, c="k", ax=axs["time series"])
nino_true_compute_clim.isel(time=slice(day_max - 30, day_max - 30 + 1)).drop_vars(
    ["dz", "lev"]
).plot.scatter(s=80, c="k", ax=axs["time series"])
nino_true_compute_clim.isel(time=slice(day_min - 30, day_min - 30 + 1)).drop_vars(
    ["dz", "lev"]
).plot.scatter(s=80, c="k", ax=axs["time series"])

axs["time series"].set_title("")
axs["time series"].set_xlabel("")
axs["time series"].legend()
# axs['time series'].set_title('Nino 3.4 Index')

############################################
# El Nino Tropics Profiles
############################################

day_start = day_max
window = 3

time_slice = slice(
    pred_dict[key2]["ds_prediction_surface"]["time"][day_start],
    pred_dict[key2]["ds_prediction_surface"]["time"][day_start + window],
)
times = pred_dict[key2]["ds_prediction"]["time"][day_start : day_start + window].data
days_of_year = [i.dayofyr for i in times]
true_clim_to_remove = clim.sel(dayofyear=days_of_year).rename({"dayofyear": "time"})
true_clim_to_remove["time"] = times
pred_clim_to_remove = (
    pred_dict[key1]["clim_pred"]
    .sel(dayofyear=days_of_year)
    .rename({"dayofyear": "time"})
)
pred_clim_to_remove["time"] = times
pred_clim_to_remove_temp = (
    pred_dict[key2]["clim_pred"]
    .sel(dayofyear=days_of_year)
    .rename({"dayofyear": "time"})
)
pred_clim_to_remove_temp["time"] = times

tropics_profile = (
    data_surface["thetao"].sel(time=time_slice, x=slice(118, 260), y=slice(-5, 5))
    - true_clim_to_remove.sel(x=slice(118, 260), y=slice(-5, 5))
).mean(["time", "y"])
tropics_profile_pred = (
    pred_dict[key1]["ds_prediction_surface"]["thetao"][
        day_start : day_start + window
    ].sel(lev=slice(0, 500), x=slice(118, 260), y=slice(-5, 5))
    - pred_clim_to_remove.sel(x=slice(118, 260), y=slice(-5, 5))
).mean(["time", "y"])
tropics_profile_pred_temp = (
    pred_dict[key2]["ds_prediction_surface"]["thetao"][
        day_start : day_start + window
    ].sel(lev=slice(0, 500), x=slice(118, 260), y=slice(-5, 5))
    - pred_clim_to_remove_temp.sel(x=slice(118, 260), y=slice(-5, 5))
).mean(["time", "y"])
tropics_profile = tropics_profile.rename("Anomaly")
tropics_profile_pred = tropics_profile_pred.rename("Anomaly")
tropics_profile_pred_temp = tropics_profile_pred_temp.rename("Anomaly")
tropics_profile = tropics_profile.assign_attrs(units=r"${^oC}$")
tropics_profile_pred = tropics_profile_pred.assign_attrs(units=r"${^oC}$")
tropics_profile_pred_temp = tropics_profile_pred_temp.assign_attrs(units=r"${^oC}$")
tropics_profile_pred_temp["x"] = tropics_profile_pred_temp["x"].assign_attrs(
    units=r"${^o}$"
)

tropics_profile.plot.pcolormesh(
    ax=axs["nino_true"], y="lev", cmap=cm.cm.diff, vmin=-2, vmax=2, add_colorbar=False
)
axs["nino_true"].set_title("Nino Conditions GT")
axs["nino_true"].set_xlabel("")
axs["nino_true"].invert_yaxis()
tropics_profile_pred.plot.pcolormesh(
    ax=axs["nino_pred"], y="lev", cmap=cm.cm.diff, vmin=-2, vmax=2, add_colorbar=False
)
axs["nino_pred"].set_title(f"Nino Conditions {pred_dict[key1]['name']}")
axs["nino_pred"].set_xlabel("")
axs["nino_pred"].invert_yaxis()
tropics_profile_pred_temp.plot.pcolormesh(
    ax=axs["nino_pred_temp"],
    y="lev",
    cmap=cm.cm.diff,
    vmin=-2,
    vmax=2,
    add_colorbar=False,
)
axs["nino_pred_temp"].set_title(f"Nino Conditions {pred_dict[key2]['name']}")
# axs['nino_pred_temp'].set_xlabel('')
axs["nino_pred_temp"].invert_yaxis()

## Stats
# for name,pred in zip([pred_dict[key1]["name"], pred_dict[key2]["name"]], [tropics_profile_pred, tropics_profile_pred_temp]):
#     mae = np.abs((pred - tropics_profile).mean(['x', 'lev']))
#     cor = ((pred*tropics_profile)).mean(['x', 'lev']) / np.sqrt((pred**2).mean(['x', 'lev']) * (tropics_profile**2).mean(['x', 'lev']))

#     print(name, "mae: ", mae.compute().item(), "cor: ", cor.compute().item())

for name, pred in zip(
    [pred_dict[key1]["name"], pred_dict[key2]["name"]],
    [tropics_profile_pred, tropics_profile_pred_temp],
):
    mae = np.abs((pred - tropics_profile).weighted(data["dz"]).mean(["x", "lev"]))
    cor = (pred * tropics_profile).weighted(data["dz"]).mean(["x", "lev"]) / np.sqrt(
        (pred**2).weighted(data["dz"]).mean(["x", "lev"])
        * (tropics_profile**2).weighted(data["dz"]).mean(["x", "lev"])
    )

    print(
        name,
        "El Nino mae: ",
        mae.compute().item(),
        "El Nino cor: ",
        cor.compute().item(),
    )

############################################
# La Nina Tropics Profiles
############################################

day_start = day_min
window = 3

time_slice = slice(
    pred_dict[key2]["ds_prediction"]["time"][day_start],
    pred_dict[key2]["ds_prediction"]["time"][day_start + window],
)
times = pred_dict[key2]["ds_prediction"]["time"][day_start : day_start + window].data
days_of_year = [i.dayofyr for i in times]
true_clim_to_remove = clim.sel(dayofyear=days_of_year).rename({"dayofyear": "time"})
true_clim_to_remove["time"] = times
pred_clim_to_remove = (
    pred_dict[key1]["clim_pred"]
    .sel(dayofyear=days_of_year)
    .rename({"dayofyear": "time"})
)
pred_clim_to_remove["time"] = times
pred_clim_to_remove_temp = (
    pred_dict[key2]["clim_pred"]
    .sel(dayofyear=days_of_year)
    .rename({"dayofyear": "time"})
)
pred_clim_to_remove_temp["time"] = times

tropics_profile = (
    data_surface["thetao"].sel(time=time_slice, x=slice(118, 260), y=slice(-5, 5))
    - true_clim_to_remove.sel(x=slice(118, 260), y=slice(-5, 5))
).mean(["time", "y"])
tropics_profile_pred = (
    pred_dict[key1]["ds_prediction_surface"]["thetao"][
        day_start : day_start + window
    ].sel(lev=slice(0, 500), x=slice(118, 260), y=slice(-5, 5))
    - pred_clim_to_remove.sel(x=slice(118, 260), y=slice(-5, 5))
).mean(["time", "y"])
tropics_profile_pred_temp = (
    pred_dict[key2]["ds_prediction_surface"]["thetao"][
        day_start : day_start + window
    ].sel(lev=slice(0, 500), x=slice(118, 260), y=slice(-5, 5))
    - pred_clim_to_remove_temp.sel(x=slice(118, 260), y=slice(-5, 5))
).mean(["time", "y"])
tropics_profile = tropics_profile.rename("Anomaly")
tropics_profile_pred = tropics_profile_pred.rename("Anomaly")
tropics_profile_pred_temp = tropics_profile_pred_temp.rename("Anomaly")
tropics_profile = tropics_profile.assign_attrs(units=r"${^oC}$")
tropics_profile_pred = tropics_profile_pred.assign_attrs(units=r"${^oC}$")
tropics_profile_pred_temp = tropics_profile_pred_temp.assign_attrs(units=r"${^oC}$")
tropics_profile_pred_temp["x"] = tropics_profile_pred_temp["x"].assign_attrs(
    units=r"${^o}$"
)

tropics_profile.plot.pcolormesh(
    ax=axs["nina_true"], y="lev", cmap=cm.cm.diff, vmin=-2, vmax=2, add_colorbar=False
)
axs["nina_true"].set_title("Nina Conditions GT")
axs["nina_true"].set_xlabel("")
axs["nina_true"].invert_yaxis()
tropics_profile_pred.plot.pcolormesh(
    ax=axs["nina_pred"], y="lev", cmap=cm.cm.diff, vmin=-2, vmax=2, add_colorbar=False
)
axs["nina_pred"].set_title(f"Nina Conditions {pred_dict[key1]['name']}")
axs["nina_pred"].set_xlabel("")
axs["nina_pred"].invert_yaxis()
tropics_profile_pred_temp.plot.pcolormesh(
    ax=axs["nina_pred_temp"],
    y="lev",
    cmap=cm.cm.diff,
    vmin=-2,
    vmax=2,
    cbar_ax=axs["colorbar"],
    cbar_kwargs={
        "orientation": "horizontal",
        "shrink": 0.3,
        "extend": "both",
    },
)
axs["nina_pred_temp"].set_title(f"Nina Conditions {pred_dict[key2]['name']}")
# axs['nina_pred_temp'].set_xlabel('')
axs["nina_pred_temp"].invert_yaxis()

## Stats
# for name,pred in zip([pred_dict[key1]["name"], pred_dict[key2]["name"]], [tropics_profile_pred, tropics_profile_pred_temp]):
#     mae = np.abs((pred - tropics_profile).mean(['x', 'lev']))
#     cor = ((pred*tropics_profile)).mean(['x', 'lev']) / np.sqrt((pred**2).mean(['x', 'lev']) * (tropics_profile**2).mean(['x', 'lev']))

#     print(name, "mae: ", mae.compute().item(), "cor: ", cor.compute().item())

for name, pred in zip(
    [pred_dict[key1]["name"], pred_dict[key2]["name"]],
    [tropics_profile_pred, tropics_profile_pred_temp],
):
    mae = np.abs((pred - tropics_profile).weighted(data["dz"]).mean(["x", "lev"]))
    cor = (pred * tropics_profile).weighted(data["dz"]).mean(["x", "lev"]) / np.sqrt(
        (pred**2).weighted(data["dz"]).mean(["x", "lev"])
        * (tropics_profile**2).weighted(data["dz"]).mean(["x", "lev"])
    )

    print(
        name,
        "La Nina mae: ",
        mae.compute().item(),
        "La Nina cor: ",
        cor.compute().item(),
    )

############################################
# Maps
############################################
# maybe define this centrally and use for all plots from this variable?
bound_east = 118  # i think these are outdated?
bound_west = 260
bound_north = 5
bound_south = -5
# nino 3.4 box
# nino 3.4 box
nino_east = 190
nino_west = 240

axs["map"].set_extent([70, 320, -25, 25], crs=ccrs.PlateCarree())
axs["map"].stock_img()
axs["map"].coastlines(color="0.3", lw=0.5)
gl = axs["map"].gridlines(draw_labels=True, color="0.4")
box_plot(
    [bound_east, bound_west, bound_south, bound_north],
    ax=axs["map"],
    color="orange",
    transform=ccrs.PlateCarree(),
    label="Full Profile",
)
box_plot(
    [nino_east, nino_west, bound_south, bound_north],
    ax=axs["map"],
    color="red",
    ls="--",
    transform=ccrs.PlateCarree(),
    label="Niño 3.4",
)
axs["map"].legend(bbox_to_anchor=[0.5, -0.5], loc="center", ncol=2)

plt.savefig(
    os.path.join(enso_path, "Nino_Figure_Short_with_map.png"),
    bbox_inches="tight",
    dpi=600,
)


# #### OHC Maps

# In[49]:


def raw_ohc(ds):
    c_p = 3850  # J/(kg C)
    rho_0 = 1025  # kg/m^3
    ohc = ds.thetao * c_p * rho_0  # C*J/(kg C)*kg/m^3 = J/m^3
    return ohc


def vertical_ohc(ds):
    ohc_raw = raw_ohc(ds)
    ohc_intz = ohc_raw.weighted(ds.dz).sum("lev")
    # multiply by area to get Joules
    ohc_intz = ohc_intz * ds.areacello

    return ohc_intz


def ohc_map(ohc_intz):
    # return last year - first year
    return ohc_intz.isel(time=slice(-73, None)).mean("time") - ohc_intz.isel(
        time=slice(0, 73)
    ).mean("time")


ohc_truth = vertical_ohc(ds_groundtruth)
ohc_truth_timeseries = ohc_truth.sum(["x", "y"]).load()

for k in pred_dict.keys():
    pred_dict[k]["ohc_prediction"] = vertical_ohc(pred_dict[k]["ds_prediction"])
    pred_dict[k]["ohc_prediction_timeseries"] = (
        pred_dict[k]["ohc_prediction"].sum(["x", "y"]).load()
    )

ohc_truth_map = ohc_map(ohc_truth).load()
ohc_truth_map["y"] = ohc_truth_map.y.assign_attrs(
    long_name="latitude", units=r"$\degree$"
)
ohc_truth_map["x"] = ohc_truth_map.x.assign_attrs(
    long_name="longitude", units=r"$\degree$"
)
ohc_truth_map = ohc_truth_map.assign_attrs(units="J").rename("Ocean Heat Content")
pred_dict[key1]["ohc_prediction_map"] = ohc_map(
    pred_dict[key1]["ohc_prediction"]
).load()
pred_dict[key1]["ohc_prediction_map"] = (
    pred_dict[key1]["ohc_prediction_map"]
    .assign_attrs(units="J")
    .rename("Ocean Heat Content")
)
pred_dict[key1]["ohc_prediction_map"]["y"] = pred_dict[key1][
    "ohc_prediction_map"
].y.assign_attrs(long_name="latitude", units=r"$\degree$")
pred_dict[key1]["ohc_prediction_map"]["x"] = pred_dict[key1][
    "ohc_prediction_map"
].x.assign_attrs(long_name="longitude", units=r"$\degree$")
pred_dict[key2]["ohc_prediction_map"] = ohc_map(
    pred_dict[key2]["ohc_prediction"]
).load()
pred_dict[key2]["ohc_prediction_map"] = (
    pred_dict[key2]["ohc_prediction_map"]
    .assign_attrs(units="J")
    .rename("Ocean Heat Content")
)
pred_dict[key2]["ohc_prediction_map"]["y"] = pred_dict[key2][
    "ohc_prediction_map"
].y.assign_attrs(long_name="latitude", units=r"$\degree$")
pred_dict[key2]["ohc_prediction_map"]["x"] = pred_dict[key2][
    "ohc_prediction_map"
].x.assign_attrs(long_name="longitude", units=r"$\degree$")

plt.clf()
fig, ax = plt.subplots(
    1,
    4,
    figsize=(12, 3),
    layout="constrained",
    gridspec_kw={
        "width_ratios": [0.98, 0.98, 0.98, 0.06],
        "height_ratios": [1],
        "wspace": 0.04,
        "hspace": 0.4,
    },
)
plt.rcParams.update({"font.size": 9})

new_cmap = cm.cm.balance
mask = ds_groundtruth.wetmask.isel(lev=0)
ohc_truth_map = ohc_truth_map.where(mask)
pred_dict[key1]["ohc_prediction_map"] = pred_dict[key1]["ohc_prediction_map"].where(
    mask
)
pred_dict[key2]["ohc_prediction_map"] = pred_dict[key2]["ohc_prediction_map"].where(
    mask
)
new_cmap.set_bad("grey", 0.6)

vmax = 2.5e19

ohc_truth_map.plot(ax=ax[0], vmax=vmax, cmap=new_cmap, add_colorbar=False)
ax[0].set_title(dataset_name)

pred_dict[key1]["ohc_prediction_map"].plot(
    ax=ax[1], vmax=vmax, cmap=new_cmap, add_colorbar=False
)
ax[1].set_title(pred_dict[key1]["name"])

pred_dict[key2]["ohc_prediction_map"].plot(
    ax=ax[2], vmax=vmax, cmap=new_cmap, cbar_ax=ax[3]
)
ax[2].set_title(pred_dict[key2]["name"])

plt.savefig(os.path.join(ohc_path, "OHC_map.png"), bbox_inches="tight", dpi=600)


# In[50]:


# OHC Map + Bias
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

Days_to_Eq = 0
c_p = 3850  # J/(kg C)
rho_0 = 1025  # kg/m^3
zeta_joules_factor = 1e21  # Conversion factor to ZJ

plt.rcParams.update({"font.size": 14})
fig, axs = plt.subplots(
    2,
    3,
    figsize=(16, 6),
    subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"wspace": 0.02, "hspace": 0.23},
)
axs = axs.flatten()


def ohc_map(ohc_intz):
    # return last 1 year - first 1 year
    return ohc_intz.isel(time=slice(-73, None)).mean("time") - ohc_intz.isel(
        time=slice(0, 73)
    ).mean("time")


# Define a common plotting function for Cartesian lat-lon grids


def plot_ohc(ax, ohc_data, title, i):
    # Configure colormap and set color for NaN values (land)
    colormap = (
        cmocean.cm.balance
    )  # cmocean.cm.thermal  # Using thermal colormap from cmocean
    colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
    mean = ohc_data.mean().compute().item()
    std = ohc_data.std().compute().item()
    vmin = mean - 8 * std
    vmax = mean + 8 * std
    im = ax.pcolormesh(
        ohc_data["x"],
        ohc_data["y"],
        ohc_data,
        shading="auto",
        cmap=colormap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.set_title(title, fontsize=14)
    # Set longitude and latitude labels
    gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    if i > 0:
        gl.left_labels = False
    return im


def plot_diff_ohc(ax, ohc_data, gt_ohc_data, title, i):
    # Configure colormap and set color for NaN values (land)
    colormap = cmocean.cm.balance  # Using thermal colormap from cmocean
    colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
    bias_ohc = ohc_data - gt_ohc_data
    mean = ohc_data.mean().compute().item()
    std = ohc_data.std().compute().item()
    vmin = mean - 8 * std
    vmax = mean + 8 * std
    im = ax.pcolormesh(
        bias_ohc["x"],
        bias_ohc["y"],
        bias_ohc,
        shading="auto",
        cmap=colormap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.set_title(title, fontsize=14)
    # Set longitude and latitude labels
    gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    if i > 4:
        gl.left_labels = False
    return im


# Calculate Ocean Heat Content for different scenarios and convert to Zeta Joules
titles = [dataset_name, pred_dict[key1]["name"], pred_dict[key2]["name"]]
bias_titles = [pred_dict[key1]["name"] + " Bias", pred_dict[key2]["name"] + " Bias"]
datasets = [data, pred_dict[key1]["ds_prediction"], pred_dict[key2]["ds_prediction"]]

for i, (ax, title, ds) in enumerate(zip(axs, titles, datasets)):
    section_mask = np.isnan(ds["thetao"]).all("lev").isel(time=5)
    OHC_pred = (
        (ds["thetao"][Days_to_Eq:] * c_p * rho_0 / zeta_joules_factor)
        .weighted(ds["areacello"] * ds["dz"])
        .sum(["lev"])
        .compute()
    )
    OHC_pred = ohc_map(OHC_pred)
    OHC_pred = OHC_pred.where(~section_mask)
    OHC_pred = OHC_pred.rename("Ocean Heat Content")
    OHC_pred["y"] = OHC_pred.y.assign_attrs(long_name="latitude", units=r"${^o}$")
    OHC_pred["x"] = OHC_pred.x.assign_attrs(long_name="longitude", units=r"${^o}$")
    OHC_pred = OHC_pred.assign_attrs(units="ZJ")

    if i == 0:
        gt_ohc = OHC_pred
    elif i == 1:
        pred1_ohc = OHC_pred
    elif i == 2:
        pred2_ohc = OHC_pred

    # Plot using the Cartesian lat-lon grid
    im = plot_ohc(ax, OHC_pred, title, i)

# Add colorbar
cbar = fig.colorbar(im, ax=axs[:3], orientation="vertical", fraction=0.01, pad=0.02)
cbar.set_label("Ocean Heat Content [ZJ]", fontsize=14)

im = plot_diff_ohc(axs[4], pred1_ohc, gt_ohc, bias_titles[0], 4)
im = plot_diff_ohc(axs[5], pred2_ohc, gt_ohc, bias_titles[1], 5)

# Add colorbar
cbar = fig.colorbar(im, ax=axs[3:], orientation="vertical", fraction=0.01, pad=0.02)
cbar.set_label("Ocean Heat Content [ZJ]", fontsize=14)

fig.delaxes(axs[3])

# Save or display the plot
plt.savefig(os.path.join(ohc_path, "OHC_Global_map.png"), bbox_inches="tight", dpi=600)
# plt.show()


# In[51]:


def map_bias_avg(da, fig, title="", **kwargs):
    var_name = kwargs["var_name"]

    data_pred1 = da[0]
    data_pred2 = da[1]

    plt.clf()
    plt.rcParams.update({"font.size": 14})

    # Define colormap
    new_cmap = cm.cm.balance
    new_cmap.set_bad(color="grey", alpha=0.0)

    # Set common color range for the colorbar
    vmin, vmax = {
        "thetao": (-5, 5),
        "so": (-1, 1),
        "uo": (-0.01, 0.01),
        "vo": (-0.01, 0.01),
        "zos": (-1, 1),
        "OHC": (-0.1, 0.1),
    }[var_name]

    # Create figure with appropriate layout
    ax = fig.subplots(
        1,
        2,
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"wspace": 0.02, "hspace": 0.05},
    )
    ax = np.array(ax)  # Ensure ax is an array for easy indexing

    # Plot Predictions
    im = data_pred1.plot(
        ax=ax[0],
        cmap=new_cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax[0].add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.1)
    ax[0].set_title(pred_dict[key1]["name"] + " Bias", fontsize=14)
    gl = ax[0].gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    im = data_pred2.plot(
        ax=ax[1],
        cmap=new_cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax[1].add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.1)
    ax[1].set_title(pred_dict[key2]["name"] + " Bias", fontsize=14)
    gl = ax[1].gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])
    ax[1].set_yticks([])
    ax[1].set_ylabel("")

    # Add colorbar for plots
    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(var_list[var_name])

    # Add title
    plt.text(
        0.8,
        1.45,
        title,
        ha="center",
        va="bottom",
        transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
    )

    return ax, im


# In[52]:


first_year = slice(None, 73)
last_year = slice(-73, -1)
second_last_year = slice(-146, -74)
third_last_year = slice(-219, -147)


datasets = [data, pred_dict[key1]["ds_prediction"], pred_dict[key2]["ds_prediction"]]

for i, (ax, title, ds) in enumerate(zip(axs, titles, datasets)):
    section_mask = np.isnan(ds["thetao"]).all("lev")
    OHC_pred = (
        (ds["thetao"][Days_to_Eq:] * c_p * rho_0 / zeta_joules_factor)
        .weighted(ds["areacello"] * ds["dz"])
        .sum(["lev"])
        .compute()
    )
    OHC_pred = OHC_pred.where(~section_mask)
    OHC_pred = OHC_pred.rename("Ocean Heat Content")
    OHC_pred["y"] = OHC_pred.y.assign_attrs(long_name="latitude", units=r"${^o}$")
    OHC_pred["x"] = OHC_pred.x.assign_attrs(long_name="longitude", units=r"${^o}$")
    OHC_pred = OHC_pred.assign_attrs(units="ZJ")

    if i == 0:
        gt_ohc = OHC_pred
    elif i == 1:
        pred1_ohc = OHC_pred
    elif i == 2:
        pred2_ohc = OHC_pred


# In[53]:


da = xr.concat(
    [
        (
            pred1_ohc.isel(time=last_year).mean("time")
            - pred1_ohc.isel(time=second_last_year).mean("time")
        ).compute(),
        (
            pred2_ohc.isel(time=last_year).mean("time")
            - pred2_ohc.isel(time=second_last_year).mean("time")
        ).compute(),
    ],
    dim="dummy",
)
fig, ax = plt.subplots(figsize=(10, 10))
# increase size of plot
ax, im = map_bias_avg(
    da,
    fig,
    var_name="OHC",
    title="OHC Anomaly Bias (Last Year - Second Last Year)",
)

# plot
fig.tight_layout()
plt.savefig(
    os.path.join(ohc_path, "OHC_Bias_Map_Diff1_2.png"),
    bbox_inches="tight",
    dpi=600,
)


# In[54]:


da = xr.concat(
    [
        (
            pred1_ohc.isel(time=last_year).mean("time")
            - pred1_ohc.isel(time=third_last_year).mean("time")
        ).compute(),
        (
            pred2_ohc.isel(time=last_year).mean("time")
            - pred2_ohc.isel(time=third_last_year).mean("time")
        ).compute(),
    ],
    dim="dummy",
)
# increase size of plot
fig, ax = plt.subplots(figsize=(10, 10))
ax, im = map_bias_avg(
    da,
    fig,
    var_name="OHC",
    title="OHC Anomaly Bias (Last Year - Third Last Year)",
)

# plot
fig.tight_layout()
plt.savefig(
    os.path.join(ohc_path, "OHC_Bias_Map_Diff1_3.png"),
    bbox_inches="tight",
    dpi=600,
)


# In[55]:


da = xr.concat(
    [
        (
            pred1_ohc.isel(time=last_year).mean("time")
            - pred1_ohc.isel(time=first_year).mean("time")
        ).compute(),
        (
            pred2_ohc.isel(time=last_year).mean("time")
            - pred2_ohc.isel(time=first_year).mean("time")
        ).compute(),
    ],
    dim="dummy",
)
# increase size of plot
fig, ax = plt.subplots(figsize=(10, 10))
ax, im = map_bias_avg(
    da,
    fig,
    var_name="OHC",
    title="OHC Anomaly Bias (Last Year - First Year)",
)

# plot
fig.tight_layout()
plt.savefig(
    os.path.join(ohc_path, "OHC_Bias_Map_Diff_Last_First.png"),
    bbox_inches="tight",
    dpi=600,
)


# #### SST Map

# In[56]:


import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

Days_to_Eq = 0
plt.rcParams.update({"font.size": 14})
fig, axs = plt.subplots(
    2,
    3,
    figsize=(16, 6),
    subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"wspace": 0.02, "hspace": 0.23},
)
axs = axs.flatten()

# Define a common plotting function for Cartesian lat-lon grids


def plot_sst(ax, sst_data, title, i):
    colormap = cmocean.cm.thermal
    colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
    mean = sst_data.mean().compute().item()
    std = sst_data.std().compute().item()
    vmin = mean - std
    vmax = mean + std
    im = ax.pcolormesh(
        sst_data["x"],
        sst_data["y"],
        sst_data,
        shading="auto",
        cmap=colormap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.set_title(title, fontsize=14)
    gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    if i > 0:
        gl.left_labels = False
    return im


def plot_diff_sst(ax, sst_data, gt_sst_data, title, i):
    colormap = cmocean.cm.balance
    colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
    sst_bias = sst_data - gt_sst_data
    im = ax.pcolormesh(
        sst_bias["x"],
        sst_bias["y"],
        sst_bias,
        shading="auto",
        cmap=colormap,
        transform=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.set_title(title, fontsize=14)
    gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    if i > 4:
        gl.left_labels = False
    return im


# Calculate Sea Surface Temperature (SST) for different scenarios
titles = [dataset_name, pred_dict[key1]["name"], pred_dict[key2]["name"]]
bias_titles = [pred_dict[key1]["name"] + " Bias", pred_dict[key2]["name"] + " Bias"]
datasets = [data, pred_dict[key1]["ds_prediction"], pred_dict[key2]["ds_prediction"]]

for i, (ax, title, ds) in enumerate(zip(axs, titles, datasets)):
    section_mask = np.isnan(ds["thetao"]).isel(lev=0).isel(time=5)
    SST_pred = ds["thetao"].isel(lev=0).mean("time")
    SST_pred = SST_pred.where(~section_mask)
    SST_pred = SST_pred.rename("2.5m " + r"$\theta_O$")
    SST_pred["y"] = SST_pred.y.assign_attrs(long_name="latitude", units=r"${^o}$")
    SST_pred["x"] = SST_pred.x.assign_attrs(long_name="longitude", units=r"${^o}$")
    SST_pred = SST_pred.assign_attrs(units=r"$\degree C$")

    if i == 0:
        gt_sst = SST_pred
    elif i == 1:
        pred1_sst = SST_pred
    elif i == 2:
        pred2_sst = SST_pred

    # Plot using the Cartesian lat-lon grid
    im = plot_sst(ax, SST_pred, title, i)

# Add colorbar for SST plots
cbar = fig.colorbar(im, ax=axs[:3], orientation="vertical", fraction=0.01, pad=0.02)
cbar.set_label(r"$\theta_O$ [$\degree C$]", fontsize=14)

# Plot biases for SST
im = plot_diff_sst(axs[4], pred1_sst, gt_sst, bias_titles[0], 4)
im = plot_diff_sst(axs[5], pred2_sst, gt_sst, bias_titles[1], 5)

# Add colorbar for bias plots
cbar = fig.colorbar(im, ax=axs[3:], orientation="vertical", fraction=0.01, pad=0.02)
cbar.set_label(r"$\theta_O$ [$\degree C$]", fontsize=14)

# Remove the empty axis
fig.delaxes(axs[3])

# Save or display the plot
plt.savefig(os.path.join(temp_path, "SST_Global_map.png"), bbox_inches="tight", dpi=600)
# plt.show()


# In[57]:


# Single Snapshot (First, Middle, Last)
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

time_indices = [0, 300, 599]
Days_to_Eq = 0


def plot_sst(ax, sst_data, title, i):
    colormap = cmocean.cm.thermal
    colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
    mean = sst_data.mean().compute().item()
    std = sst_data.std().compute().item()
    vmin = mean - std
    vmax = mean + std
    im = ax.pcolormesh(
        sst_data["x"],
        sst_data["y"],
        sst_data,
        shading="auto",
        cmap=colormap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.set_title(title, fontsize=14)
    gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    if i > 0:
        gl.left_labels = False
    return im


def plot_diff_sst(ax, sst_data, gt_sst_data, title, i):
    colormap = cmocean.cm.balance
    colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
    sst_bias = sst_data - gt_sst_data
    im = ax.pcolormesh(
        sst_bias["x"],
        sst_bias["y"],
        sst_bias,
        shading="auto",
        cmap=colormap,
        transform=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.set_title(title, fontsize=14)
    gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    if i > 4:
        gl.left_labels = False
    return im


# Calculate Sea Surface Temperature (SST) for different scenarios

for t_index in time_indices:
    plt.rcParams.update({"font.size": 14})
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(16, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"wspace": 0.02, "hspace": 0.23},
    )
    axs = axs.flatten()

    # Define a common plotting function for Cartesian lat-lon grids

    titles = [
        dataset_name + f" t={t_index}",
        pred_dict[key1]["name"] + f" t={t_index}",
        pred_dict[key2]["name"] + f" t={t_index}",
    ]
    bias_titles = [pred_dict[key1]["name"] + " Bias", pred_dict[key2]["name"] + " Bias"]
    datasets = [
        data,
        pred_dict[key1]["ds_prediction"],
        pred_dict[key2]["ds_prediction"],
    ]

    for i, (ax, title, ds) in enumerate(zip(axs, titles, datasets)):
        section_mask = np.isnan(ds["thetao"]).isel(lev=0).isel(time=5)
        SST_pred = ds["thetao"].isel(lev=0).isel(time=t_index)
        SST_pred = SST_pred.where(~section_mask)
        SST_pred = SST_pred.rename("2.5m " + r"$\theta_O$")
        SST_pred["y"] = SST_pred.y.assign_attrs(long_name="latitude", units=r"${^o}$")
        SST_pred["x"] = SST_pred.x.assign_attrs(long_name="longitude", units=r"${^o}$")
        SST_pred = SST_pred.assign_attrs(units=r"$\degree C$")

        if i == 0:
            gt_sst = SST_pred
        elif i == 1:
            pred1_sst = SST_pred
        elif i == 2:
            pred2_sst = SST_pred

        # Plot using the Cartesian lat-lon grid
        im = plot_sst(ax, SST_pred, title, i)

    # Add colorbar for SST plots
    cbar = fig.colorbar(im, ax=axs[:3], orientation="vertical", fraction=0.01, pad=0.02)
    cbar.set_label(r"$\theta_O$ [$\degree C$]", fontsize=14)

    # Plot biases for SST
    im = plot_diff_sst(axs[4], pred1_sst, gt_sst, bias_titles[0], 4)
    im = plot_diff_sst(axs[5], pred2_sst, gt_sst, bias_titles[1], 5)

    # Add colorbar for bias plots
    cbar = fig.colorbar(im, ax=axs[3:], orientation="vertical", fraction=0.01, pad=0.02)
    cbar.set_label(r"$\theta_O$ [$\degree C$]", fontsize=14)

    # Remove the empty axis
    fig.delaxes(axs[3])

    # Save or display the plot
    plt.savefig(
        os.path.join(temp_path, f"SST_map_snapshot_t_{t_index}.png"),
        bbox_inches="tight",
        dpi=600,
    )
    # plt.show()
    plt.close()


# #### Salinity Map

# In[58]:


import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

Days_to_Eq = 0
plt.rcParams.update({"font.size": 14})
fig, axs = plt.subplots(
    2,
    3,
    figsize=(16, 6),
    subplot_kw={"projection": ccrs.PlateCarree()},
    gridspec_kw={"wspace": 0.02, "hspace": 0.23},
)
axs = axs.flatten()

# Define a common plotting function for Cartesian lat-lon grids


def plot_sst(ax, sst_data, title, i):
    colormap = cmocean.cm.thermal
    colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
    mean = sst_data.mean().compute().item()
    std = sst_data.std().compute().item()
    vmin = mean - std
    vmax = mean + std
    im = ax.pcolormesh(
        sst_data["x"],
        sst_data["y"],
        sst_data,
        shading="auto",
        cmap=colormap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.set_title(title, fontsize=14)
    gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    if i > 0:
        gl.left_labels = False
    return im


def plot_diff_sst(ax, sst_data, gt_sst_data, title, i):
    colormap = cmocean.cm.balance
    colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
    sst_bias = sst_data - gt_sst_data
    im = ax.pcolormesh(
        sst_bias["x"],
        sst_bias["y"],
        sst_bias,
        shading="auto",
        cmap=colormap,
        transform=ccrs.PlateCarree(),
        vmin=-0.5,
        vmax=0.5,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.set_title(title, fontsize=14)
    gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    if i > 4:
        gl.left_labels = False
    return im


# Calculate Sea Surface Salinity (SSS) for different scenarios
titles = [dataset_name, pred_dict[key1]["name"], pred_dict[key2]["name"]]
bias_titles = [pred_dict[key1]["name"] + " Bias", pred_dict[key2]["name"] + " Bias"]
datasets = [data, pred_dict[key1]["ds_prediction"], pred_dict[key2]["ds_prediction"]]

for i, (ax, title, ds) in enumerate(zip(axs, titles, datasets)):
    section_mask = np.isnan(ds["so"]).isel(lev=0).isel(time=5)
    SST_pred = ds["so"].isel(lev=0).mean("time")
    SST_pred = SST_pred.where(~section_mask)
    SST_pred = SST_pred.rename("2.5m " + r"$so$")
    SST_pred["y"] = SST_pred.y.assign_attrs(long_name="latitude", units=r"${^o}$")
    SST_pred["x"] = SST_pred.x.assign_attrs(long_name="longitude", units=r"${^o}$")
    SST_pred = SST_pred.assign_attrs(units=r"$psu$")

    if i == 0:
        gt_sst = SST_pred
    elif i == 1:
        pred1_sst = SST_pred
    elif i == 2:
        pred2_sst = SST_pred

    # Plot using the Cartesian lat-lon grid
    im = plot_sst(ax, SST_pred, title, i)

# Add colorbar for SST plots
cbar = fig.colorbar(im, ax=axs[:3], orientation="vertical", fraction=0.01, pad=0.02)
cbar.set_label(r"$so$ [$psu$]", fontsize=14)

# Plot biases for SST
im = plot_diff_sst(axs[4], pred1_sst, gt_sst, bias_titles[0], 4)
im = plot_diff_sst(axs[5], pred2_sst, gt_sst, bias_titles[1], 5)

# Add colorbar for bias plots
cbar = fig.colorbar(im, ax=axs[3:], orientation="vertical", fraction=0.01, pad=0.02)
cbar.set_label(r"$so$ [$psu$]", fontsize=14)

# Remove the empty axis
fig.delaxes(axs[3])

# Save or display the plot
plt.savefig(
    os.path.join(salinity_path, "SeaSurfaceSalinity_Global_map.png"),
    bbox_inches="tight",
    dpi=600,
)
# plt.show()


# In[59]:


# Single Snapshot (First, Middle, Last)
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

time_indices = [0, 300, 599]
Days_to_Eq = 0


def plot_sst(ax, sst_data, title, i):
    colormap = cmocean.cm.thermal
    colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
    mean = sst_data.mean().compute().item()
    std = sst_data.std().compute().item()
    vmin = mean - std
    vmax = mean + std
    im = ax.pcolormesh(
        sst_data["x"],
        sst_data["y"],
        sst_data,
        shading="auto",
        cmap=colormap,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.set_title(title, fontsize=14)
    gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    if i > 0:
        gl.left_labels = False
    return im


def plot_diff_sst(ax, sst_data, gt_sst_data, title, i):
    colormap = cmocean.cm.balance
    colormap.set_bad(color=(0.7, 0.7, 0.7, 0))
    sst_bias = sst_data - gt_sst_data
    im = ax.pcolormesh(
        sst_bias["x"],
        sst_bias["y"],
        sst_bias,
        shading="auto",
        cmap=colormap,
        transform=ccrs.PlateCarree(),
        vmin=-0.5,
        vmax=0.5,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.set_title(title, fontsize=14)
    gl = ax.gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    if i > 4:
        gl.left_labels = False
    return im


for t_index in time_indices:
    plt.rcParams.update({"font.size": 14})
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(16, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"wspace": 0.02, "hspace": 0.23},
    )
    axs = axs.flatten()

    # Calculate Sea Surface Salinity (SSS) for different scenarios
    titles = [
        dataset_name + f" t={t_index}",
        pred_dict[key1]["name"] + f" t={t_index}",
        pred_dict[key2]["name"] + f" t={t_index}",
    ]
    bias_titles = [pred_dict[key1]["name"] + " Bias", pred_dict[key2]["name"] + " Bias"]
    datasets = [
        data,
        pred_dict[key1]["ds_prediction"],
        pred_dict[key2]["ds_prediction"],
    ]

    for i, (ax, title, ds) in enumerate(zip(axs, titles, datasets)):
        section_mask = np.isnan(ds["so"]).isel(lev=0).isel(time=5)
        SST_pred = ds["so"].isel(lev=0).isel(time=t_index)
        SST_pred = SST_pred.where(~section_mask)
        SST_pred = SST_pred.rename("2.5m " + r"$so$")
        SST_pred["y"] = SST_pred.y.assign_attrs(long_name="latitude", units=r"${^o}$")
        SST_pred["x"] = SST_pred.x.assign_attrs(long_name="longitude", units=r"${^o}$")
        SST_pred = SST_pred.assign_attrs(units=r"$psu$")

        if i == 0:
            gt_sst = SST_pred
        elif i == 1:
            pred1_sst = SST_pred
        elif i == 2:
            pred2_sst = SST_pred

        # Plot using the Cartesian lat-lon grid
        im = plot_sst(ax, SST_pred, title, i)

    # Add colorbar for SST plots
    cbar = fig.colorbar(im, ax=axs[:3], orientation="vertical", fraction=0.01, pad=0.02)
    cbar.set_label(r"$so$ [$psu$]", fontsize=14)

    # Plot biases for SST
    im = plot_diff_sst(axs[4], pred1_sst, gt_sst, bias_titles[0], 4)
    im = plot_diff_sst(axs[5], pred2_sst, gt_sst, bias_titles[1], 5)

    # Add colorbar for bias plots
    cbar = fig.colorbar(im, ax=axs[3:], orientation="vertical", fraction=0.01, pad=0.02)
    cbar.set_label(r"$so$ [$psu$]", fontsize=14)

    # Remove the empty axis
    fig.delaxes(axs[3])

    # Save or display the plot
    plt.savefig(
        os.path.join(salinity_path, f"SSS_map_snapshot_t_{t_index}.png"),
        bbox_inches="tight",
        dpi=600,
    )
    # plt.show()


# #### Drake Passage / Atlantic Depth Profile

# In[60]:


# # Drake Passage Full Depth
# new_cmap = cm.cm.thermal

# new_cmap.set_bad("grey", 0.6)
# plt.rcParams.update({"font.size": 9})

# day_start = 597
# window = 3
# regions = {
#     "Drake Passage": {"lon": 290, "lat_bnds": slice(-70, -55)},
#     "Atlantic": {"lon": 330, "lat_bnds": slice(-80, 90)},
# }
# surface = False
# fig, ax = plt.subplots(
#     2,
#     3,
#     figsize=(10, 6),
#     gridspec_kw={
#         "width_ratios": [1, 1, 1],
#         "height_ratios": [0.9, 0.9],
#         "wspace": 0.25,
#         "hspace": 0.4,
#     },
# )

# for i, region in enumerate(["Drake Passage", "Atlantic"]):
#     bounds = regions[region]
#     var = "thetao"
#     if surface:
#         level_slice = slice(0, 1000)
#     else:
#         level_slice = slice(None)

#     depth_slice = (
#         data[var]
#         .sel(x=bounds["lon"], method="nearest")
#         .sel(y=bounds["lat_bnds"], lev=level_slice)
#         .isel(time=slice(day_start, day_start + window))
#         .mean("time")
#     )
#     depth_slice = depth_slice.assign_attrs(long_name=data[var].long_name, units=r"^oC")
#     depth_slice["y"] = depth_slice.y.assign_attrs(long_name="latitude")

#     max_val = np.ceil(depth_slice.max()).values

#     depth_slice.plot(
#         ax=ax[i, 0], add_colorbar=False, cmap=new_cmap, vmin=-2.5, vmax=max_val
#     )
#     ax[i, 0].invert_yaxis()
#     ax[i, 0].set_title(dataset_name)
#     linear_piecewise_scale(1000, 5, ax=ax[i, 0])
#     ax[i, 0].axhline(1000, color="0.5", ls="--")
#     ax[i, 0].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])

#     depth_slice_pred = (
#         pred_dict[key1]["ds_prediction"][var]
#         .sel(x=bounds["lon"], method="nearest")
#         .sel(y=bounds["lat_bnds"], lev=level_slice)
#         .isel(time=slice(day_start, day_start + window))
#         .mean("time")
#     )
#     depth_slice_pred = depth_slice_pred.assign_attrs(
#         long_name=data[var].long_name, units=r"^oC"
#     )
#     pred_plot = depth_slice_pred.plot(
#         ax=ax[i, 1], add_colorbar=False, cmap=new_cmap, vmin=-2.5, vmax=max_val
#     )
#     ax[i, 1].invert_yaxis()
#     ax[i, 1].set_ylabel("")
#     ax[i, 1].set_title(pred_dict[key1]["name"])
#     linear_piecewise_scale(1000, 5, ax=ax[i, 1])
#     ax[i, 1].axhline(1000, color="0.5", ls="--")
#     ax[i, 1].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])

#     depth_slice_pred = (
#         pred_dict[key2]["ds_prediction"][var]
#         .sel(x=bounds["lon"], method="nearest")
#         .sel(y=bounds["lat_bnds"], lev=level_slice)
#         .isel(time=slice(day_start, day_start + window))
#         .mean("time")
#     )
#     depth_slice_pred = depth_slice_pred.assign_attrs(
#         long_name=data[var].long_name, units=r"^oC"
#     )
#     pred_plot = depth_slice_pred.plot(
#         ax=ax[i, 2], add_colorbar=False, cmap=new_cmap, vmin=-2.5, vmax=max_val
#     )
#     ax[i, 2].invert_yaxis()
#     ax[i, 2].set_ylabel("")
#     ax[i, 2].set_title(pred_dict[key2]["name"])
#     linear_piecewise_scale(1000, 5, ax=ax[i, 2])
#     ax[i, 2].axhline(1000, color="0.5", ls="--")
#     ax[i, 2].set_yticks([0, 250, 500, 750, 1000, 3000, 5000])

#     fig.subplots_adjust(right=0.825)
#     cbar_ax = fig.add_axes([0.85, 0.125 + 0.45 * i, 0.015, 0.29])
#     cbar = fig.colorbar(pred_plot, cax=cbar_ax)
#     cbar.set_label(pred_dict[key2]["ds_prediction"][var].long_name, rotation=90)

# fig.text(
#     0.02,
#     0.5,
#     "Drake Passage" + r" lon: 70${^o}W$",
#     ha="center",
#     fontsize=14,
#     rotation=90,
# )
# fig.text(
#     0.02, 0.125, region + r" lon: 30${^o}W$", ha="center", fontsize=14, rotation=90
# )


# plt.savefig(
#     os.path.join(output_path, "Depth_Profiles_Short.png"), bbox_inches="tight", dpi=600
# )


# #### Variance Maps

# In[61]:


def detrend_and_remove_climatology(ds, var="zos"):
    # Detrend the data
    poly_coeffs = ds[var].polyfit(dim="time", deg=1)
    trend = xr.polyval(ds["time"], poly_coeffs.polyfit_coefficients).compute()

    # Remove the trend from the original data
    ssh_detrended = ds[var] - trend

    # Compute the climatology on the detrended data
    climatology = ssh_detrended.groupby("time.dayofyear").mean("time").compute()

    # Remove the seasonal cycle (climatology) from the detrended data
    day_of_year = ssh_detrended["time"].dt.dayofyear
    ssh_final = (ssh_detrended - climatology.sel(dayofyear=day_of_year)).compute()

    return ssh_final, climatology, trend


# In[62]:


with ProgressBar():
    ssh_groundtruth, _, _ = detrend_and_remove_climatology(ds_groundtruth, "zos")
    if "zos" in pred_dict[key1]["ls"]:
        ssh_prediction_temp, _, _ = detrend_and_remove_climatology(
            pred_dict[key1]["ds_prediction"], "zos"
        )
    else:
        ssh_prediction_temp = ssh_groundtruth * 0
    if "zos" in pred_dict[key2]["ls"]:
        ssh_prediction_all, _, _ = detrend_and_remove_climatology(
            pred_dict[key2]["ds_prediction"], "zos"
        )
    else:
        ssh_prediction_all = ssh_groundtruth * 0


# In[63]:


with ProgressBar():
    sst_2_5_groundtruth, _, _ = detrend_and_remove_climatology(
        ds_groundtruth.sel(lev=2.5), "thetao"
    )
    sst_2_5_prediction_temp, _, _ = detrend_and_remove_climatology(
        pred_dict[key1]["ds_prediction"].sel(lev=2.5), "thetao"
    )
    sst_2_5_prediction_all, _, _ = detrend_and_remove_climatology(
        pred_dict[key2]["ds_prediction"].sel(lev=2.5), "thetao"
    )

with ProgressBar():
    sst_550_groundtruth, _, _ = detrend_and_remove_climatology(
        ds_groundtruth.sel(lev=550), "thetao"
    )
    sst_550_prediction_temp, _, _ = detrend_and_remove_climatology(
        pred_dict[key1]["ds_prediction"].sel(lev=550), "thetao"
    )
    sst_550_prediction_all, _, _ = detrend_and_remove_climatology(
        pred_dict[key2]["ds_prediction"].sel(lev=550), "thetao"
    )

with ProgressBar():
    sst_1400_groundtruth, _, _ = detrend_and_remove_climatology(
        ds_groundtruth.sel(lev=1400), "thetao"
    )
    sst_1400_prediction_temp, _, _ = detrend_and_remove_climatology(
        pred_dict[key1]["ds_prediction"].sel(lev=1400), "thetao"
    )
    sst_1400_prediction_all, _, _ = detrend_and_remove_climatology(
        pred_dict[key2]["ds_prediction"].sel(lev=1400), "thetao"
    )


# In[64]:


fig, axs = plt.subplots(
    2,
    2,
    figsize=(16, 6),
    layout="constrained",
    gridspec_kw={
        "width_ratios": [1, 1],
        "height_ratios": [0.5, 0.5],
        "wspace": 0.05,
        "hspace": 0.05,
    },
)

# ssh_groundtruth_ts = (ssh_groundtruth*ds_groundtruth.areacello).sum(["x","y"])/ds_groundtruth.areacello.sum(["x","y"])
# ssh_prediction_temp_ts = (ssh_prediction_temp*ds_groundtruth.areacello).sum(["x","y"])/ds_groundtruth.areacello.sum(["x","y"])
# ssh_prediction_all_ts = (ssh_prediction_all*ds_groundtruth.areacello).sum(["x","y"])/ds_groundtruth.areacello.sum(["x","y"])

ssh_groundtruth_ts = ssh_groundtruth.weighted(ds_groundtruth.areacello).mean(["x", "y"])
ssh_prediction_temp_ts = ssh_prediction_temp.weighted(ds_groundtruth.areacello).mean(
    ["x", "y"]
)
ssh_prediction_all_ts = ssh_prediction_all.weighted(ds_groundtruth.areacello).mean(
    ["x", "y"]
)

ssh_groundtruth_ts = ssh_groundtruth_ts.assign_attrs(
    long_name="Mean SSH Anomaly", units="m"
).rename("Mean SSH Anomaly")
ssh_prediction_all_ts = ssh_prediction_all_ts.assign_attrs(
    long_name="Mean SSH Anomaly", units="m"
).rename("Mean SSH Anomaly")
ssh_prediction_temp_ts = ssh_prediction_temp_ts.assign_attrs(
    long_name="Mean SSH Anomaly", units="m"
).rename("Mean SSH Anomaly")

ssh_prediction_all_ts.plot(ax=axs[0, 0], label=pred_dict[key2]["name"], c=clist[0])
ssh_prediction_temp_ts.plot(ax=axs[0, 0], label=pred_dict[key1]["name"], c=clist[1])
ssh_groundtruth_ts.plot(ax=axs[0, 0], label=dataset_name, c="k")
axs[0, 0].legend(ncol=2, loc="lower right")
axs[0, 0].set_xlabel("")

count = 0
for depth in [2.5, 550, 1400]:
    if count == 0:
        ax = axs[1, 0]
        anoms = sst_2_5_groundtruth
        anoms_all = sst_2_5_prediction_all
        anoms_temp = sst_2_5_prediction_temp
    elif count == 1:
        ax = axs[0, 1]
        anoms = sst_550_groundtruth
        anoms_all = sst_550_prediction_all
        anoms_temp = sst_550_prediction_temp
    else:
        ax = axs[1, 1]
        anoms = sst_1400_groundtruth
        anoms_all = sst_1400_prediction_all
        anoms_temp = sst_1400_prediction_temp

    anoms_time_series = anoms.weighted(ds_groundtruth["areacello"]).mean(["x", "y"])
    anoms_all_time_series = anoms_all.weighted(ds_groundtruth["areacello"]).mean(
        ["x", "y"]
    )
    anoms_temp_time_series = anoms_temp.weighted(ds_groundtruth["areacello"]).mean(
        ["x", "y"]
    )

    anoms_time_series = anoms_time_series.assign_attrs(
        long_name=r"$\theta_O$ Anomaly " + str(depth) + "m", units=r"$^oC$"
    ).rename("Mean SSH Anomaly")
    anoms_all_time_series = anoms_all_time_series.assign_attrs(
        long_name=r"$\theta_O $ Anomaly " + str(depth) + "m", units=r"$^oC$"
    ).rename("Mean SSH Anomaly")
    anoms_temp_time_series = anoms_temp_time_series.assign_attrs(
        long_name=r"$\theta_O $ Anomaly " + str(depth) + "m", units=r"$^oC$"
    ).rename("Mean SSH Anomaly")

    count += 1

    anoms_all_time_series.plot(ax=ax, label=pred_dict[key2]["name"], c=clist[0])
    anoms_temp_time_series.plot(ax=ax, label=pred_dict[key1]["name"], c=clist[1])
    anoms_time_series.plot(ax=ax, label=dataset_name, c="k")
    ax.set_xlabel("")
    ax.set_title("")

plt.savefig(
    os.path.join(temp_path, "anomalies_SSH_temp_timeseries.png"),
    bbox_inches="tight",
    dpi=600,
)


# In[65]:


ssh_groundtruth = ssh_groundtruth.assign_attrs(
    long_name=r"SSH $\sigma$", units="m"
).rename(r"SSH $\sigma$")
ssh_prediction_all = ssh_prediction_all.assign_attrs(
    long_name=r"SSH $\sigma$", units="m"
).rename(r"SSH $\sigma$")
ssh_prediction_temp = ssh_prediction_temp.assign_attrs(
    long_name=r"SSH $\sigma$", units="m"
).rename(r"SSH $\sigma$")

sst_2_5_groundtruth = sst_2_5_groundtruth.assign_attrs(
    long_name=r"$\theta_O$ $\sigma$ 2.5m", units=r"$^oC$"
).rename(r"$\theta_O$ $\sigma$ 2.5m")
sst_2_5_prediction_temp = sst_2_5_prediction_temp.assign_attrs(
    long_name=r"$\theta_O $ $\sigma$ 2.5m", units=r"$^oC$"
).rename(r"$\theta_O $ $\sigma$ 2.5m")
sst_2_5_prediction_all = sst_2_5_prediction_all.assign_attrs(
    long_name=r"$\theta_O $ $\sigma$ 2.5m", units=r"$^oC$"
).rename(r"$\theta_O $ $\sigma$ 2.5m")

sst_550_groundtruth = sst_550_groundtruth.assign_attrs(
    long_name=r"$\theta_O$ $\sigma$ 550m", units=r"$^oC$"
).rename(r"$\theta_O$ $\sigma$ 550m")
sst_550_prediction_temp = sst_550_prediction_temp.assign_attrs(
    long_name=r"$\theta_O $ $\sigma$ 550m", units=r"$^oC$"
).rename(r"$\theta_O$ $\sigma$ 550m")
sst_550_prediction_all = sst_550_prediction_all.assign_attrs(
    long_name=r"$\theta_O $ $\sigma$ 550m", units=r"$^oC$"
).rename(r"$\theta_O$ $\sigma$ 550m")

sst_1400_groundtruth = sst_1400_groundtruth.assign_attrs(
    long_name=r"$\theta_O$ $\sigma$ 1400m", units=r"$^oC$"
).rename(r"$\theta_O$ $\sigma$ 1400m")
sst_1400_prediction_temp = sst_1400_prediction_temp.assign_attrs(
    long_name=r"$\theta_O $ $\sigma$ 1400m", units=r"$^oC$"
).rename(r"$\theta_O$ $\sigma$ 1400m")
sst_1400_prediction_all = sst_1400_prediction_all.assign_attrs(
    long_name=r"$\theta_O $ $\sigma$ 1400m", units=r"$^oC$"
).rename(r"$\theta_O$ $\sigma$ 1400m")


# In[66]:


def plot_map(anoms, anoms_all, anoms_temp, name):
    new_cmap = cm.cm.thermal

    new_cmap.set_bad("grey", 0.6)

    vmin = 0
    vmax = np.nanmax(anoms.std("time")) / 2
    fig, ax = plt.subplots(
        3,
        1,
        figsize=(5, 8),
        layout="constrained",
        gridspec_kw={
            "width_ratios": [1],
            "height_ratios": [0.9, 0.9, 0.9],
            "wspace": 0.05,
            "hspace": 0.05,
        },
    )

    anoms_std = anoms.std("time").assign_attrs(units="m")
    anoms_std["latitude"] = ohc_truth_map.y.assign_attrs(
        long_name="latitude", units=r"$\degree$"
    )
    anoms_std["longitude"] = ohc_truth_map.x.assign_attrs(
        long_name="longitude", units=r"$\degree$"
    )
    anoms_std.plot(ax=ax[0], add_colorbar=True, cmap=new_cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title(dataset_name)

    anoms_temp.std("time").assign_attrs(units="m").plot(
        ax=ax[1], add_colorbar=True, cmap=new_cmap, vmin=vmin, vmax=vmax
    )
    ax[1].set_title(pred_dict[key1]["name"])

    anoms_all.std("time").assign_attrs(units="m").plot(
        ax=ax[2], add_colorbar=True, cmap=new_cmap, vmin=vmin, vmax=vmax
    )
    ax[2].set_title(pred_dict[key2]["name"])

    plt.savefig(os.path.join(output_path, name), bbox_inches="tight", dpi=600)


# In[67]:


plot_map(ssh_groundtruth, ssh_prediction_all, ssh_prediction_temp, name="SSH_var_Map")


# In[68]:


def plot_3x3_map(anoms_list, anoms_all_list, anoms_temp_list, name):
    new_cmap = cm.cm.thermal

    new_cmap.set_bad("grey", 0.6)

    fig, ax = plt.subplots(
        3,
        3,
        figsize=(12, 8),
        layout="constrained",
        gridspec_kw={
            "width_ratios": [0.9, 0.9, 0.9],
            "height_ratios": [0.9, 0.9, 0.9],
            "wspace": 0.05,
            "hspace": 0.05,
        },
    )
    ax_flat = ax.flatten()

    for i, depth in enumerate(["2.5", "550", "1400"]):
        anoms = anoms_list[i]
        anoms_all = anoms_all_list[i]
        anoms_temp = anoms_temp_list[i]

        vmin = 0
        vmax = np.nanmax(anoms.std("time")) / 2

        anoms_std = anoms.std("time").assign_attrs(units="m")
        anoms_std["latitude"] = ohc_truth_map.y.assign_attrs(
            long_name="latitude", units=r"$\degree$"
        )
        anoms_std["longitude"] = ohc_truth_map.x.assign_attrs(
            long_name="longitude", units=r"$\degree$"
        )

        anoms_std.assign_attrs(
            long_name=r"$\theta_O$ $\sigma$ {0}m".format(depth), units=r"$^oC$"
        ).plot(ax=ax_flat[i], add_colorbar=True, cmap=new_cmap, vmin=vmin, vmax=vmax)
        ax_flat[i].set_title(dataset_name)

        anoms_temp.assign_attrs(
            long_name=r"$\theta_O$ $\sigma$ {0}m".format(depth), units=r"$^oC$"
        ).std("time").plot(
            ax=ax_flat[i + 3], add_colorbar=True, cmap=new_cmap, vmin=vmin, vmax=vmax
        )
        ax_flat[i + 3].set_title(pred_dict[key1]["name"])

        anoms_all.assign_attrs(
            long_name=r"$\theta_O$ $\sigma$ {0}m".format(depth), units=r"$^oC$"
        ).std("time").plot(
            ax=ax_flat[i + 6], add_colorbar=True, cmap=new_cmap, vmin=vmin, vmax=vmax
        )
        ax_flat[i + 6].set_title(pred_dict[key2]["name"])

    plt.savefig(os.path.join(temp_path, name), bbox_inches="tight", dpi=600)


# In[69]:


plot_3x3_map(
    [sst_2_5_groundtruth, sst_550_groundtruth, sst_1400_groundtruth],
    [sst_2_5_prediction_all, sst_550_prediction_all, sst_1400_prediction_all],
    [sst_2_5_prediction_temp, sst_550_prediction_temp, sst_1400_prediction_temp],
    "Temps_var_Map",
)


# ### Movies
#
# This works for two predictions only

# In[70]:


import matplotlib as mpl

mpl.use("Agg")
import gc
import glob
import os
import re
import sys
from subprocess import PIPE, STDOUT, Popen

import cartopy.crs as ccrs
import cmocean as cm
import dask.array as dsa
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dask import compute, delayed
from dask.diagnostics import ProgressBar

try:
    from tqdm.auto import tqdm

    tqdm_avail = True
except:
    warnings.warn(
        "Optional dependency `tqdm` not found. This will make progressbars a lot nicer. \
    Install with `conda install -c conda-forge tqdm`"
    )
    tqdm_avail = False


# In[71]:


keys = list(pred_dict.keys())
# assert len(keys) >= 2, "Maps supported by atleast two keys"
key1 = keys[0]
if len(keys) == 1:
    key2 = keys[0]
elif len(keys) > 2:
    print("Maps only support two models for now!!! Using the first two keys")
    assert False
else:
    key2 = keys[1]


# #### Core

# In[72]:


def _core_plot(ax, data, plotmethod=None, **kwargs):
    """Core plotting functionality"""
    # Deactivate cbar for contours (not sure this should be hardcoded...)
    if plotmethod == "contour":
        kwargs.pop("cbar_kwargs", None)

    # I am probably recoding something from matplotlib...is there a way to get
    # the plot.something functionslity with a keyword?
    # For now do it the hard way
    if plotmethod is None:
        p = data.plot(ax=ax, **kwargs)
    # doesnt work,...i want this for smoother images
    elif plotmethod == "imshow":
        # p = data.plot.imshow(ax=ax, **kwargs)
        # testing interpolation
        p = data.plot.imshow(ax=ax, interpolation="gaussian", **kwargs)
        # print(p.get_interpolation())
    elif plotmethod == "pcolormesh":
        p = data.plot.pcolormesh(ax=ax, **kwargs)
    elif plotmethod == "contour":
        p = data.plot.contour(ax=ax, **kwargs)
    elif plotmethod == "contourf":
        p = data.plot.contourf(ax=ax, **kwargs)
    else:
        raise RuntimeError(
            "Input '%s' not recognized \
        as plotmode"
            % plotmethod
        )
    return p


def _base_plot(
    ax, base_data, timestamp, timestep_value, framedim, plotmethod=None, **kwargs
):
    data = base_data.isel({framedim: timestamp})
    p = _core_plot(ax, data, plotmethod=plotmethod, **kwargs)
    return p


### Presets (should proabably put all others into a submodule)
def basic(
    da,
    fig,
    timestamp,
    timestep_value,
    framedim="time",
    plotmethod=None,
    subplot_kw=None,
    **kwargs,
):
    # create axis
    ax = fig.subplots(subplot_kw=subplot_kw)
    pp = _base_plot(
        ax, da, timestamp, timestep_value, framedim, plotmethod=plotmethod, **kwargs
    )
    return ax, pp


# Data treatment


def _parse_plot_defaults(da, kwargs):
    if isinstance(da, xr.DataArray):
        data = da
    else:
        raise RuntimeError("input of type (%s) not supported yet." % type(da))

    # check these explicitly to avoid any computation if these are set.
    if "vmin" not in kwargs.keys():
        warnings.warn(
            "No `vmin` provided. Data limits are calculated from input. Depending on the input this can take long. Pass `vmin` to avoid this step",
            UserWarning,
        )
        kwargs["vmin"] = data.min().data

    if "vmax" not in kwargs.keys():
        warnings.warn(
            "No `vmax` provided. Data limits are calculated from input. Depending on the input this can take long. Pass `vmax` to avoid this step",
            UserWarning,
        )
        kwargs["vmax"] = data.max().data

    # There is a bug that prevents this from working...Ill have to fix that upstream.
    # defaults["cbar_kwargs"] = dict(extend="neither")
    # This works for now
    kwargs.setdefault("extend", "neither")

    # if any value is dask.array compute them here.
    for k in ["vmin", "vmax"]:
        if isinstance(kwargs[k], dsa.Array):
            kwargs[k] = kwargs[k].compute()

    return kwargs


def _check_plotfunc_output(func, da, framedim="time", **kwargs):
    timestep = 0
    timestep_value = da[framedim].data[timestep]
    fig = plt.figure()
    oargs = func(da, fig, timestep, timestep_value, framedim, **kwargs)
    # I just want the number of output args, delete plot
    plt.close(fig)
    if oargs is None:
        return 0
    else:
        return len(oargs)


def _check_ffmpeg_version():
    p = Popen("ffmpeg -version", stdout=PIPE, shell=True)
    (output, err) = p.communicate()
    p_status = p.wait()
    # Parse version
    if p_status != 0:
        print("No ffmpeg found")
        return None
    else:
        # parse version number
        try:
            found = (
                re.search("ffmpeg version (.+?) Copyright", str(output))
                .group(1)
                .replace(" ", "")
            )
            return found
        except AttributeError:
            # ffmpeg version, Copyright not found in the original string
            found = None
    return found


def _execute_command(
    command, verbose=False, error=True, log_file="output.log", max_lines=10
):
    with open(log_file, "w") as f:
        p = Popen(
            command,
            stdout=PIPE,
            stderr=STDOUT,
            shell=True,
            bufsize=1,
            universal_newlines=True,
        )
        line_count = 0

        for line in iter(p.stdout.readline, ""):
            f.write(line)  # Write to log file

            if verbose and line_count < max_lines:
                sys.stdout.write(line)  # Display line in console
                sys.stdout.flush()
                line_count += 1

        p.stdout.close()
        p.wait()

    # Inform if output was truncated
    if verbose and line_count >= max_lines:
        print(f"\n...Output truncated. Full log saved to {log_file}")

    if error and p.returncode != 0:
        raise RuntimeError(
            f"Command '{command}' failed with return code {p.returncode}"
        )

    return p


def _check_ffmpeg_execute(command, verbose=False):
    if _check_ffmpeg_version() is None:
        raise RuntimeError(
            "Could not find an ffmpeg version on the system. \
        Please install ffmpeg with e.g. `conda install -c conda-forge ffmpeg`"
        )
    else:
        try:
            p = _execute_command(command, verbose=verbose)
            return p
        except RuntimeError:
            raise RuntimeError(
                "Something has gone wrong. Use `verbose=True` to check if ffmpeg displays a problem"
            )


def convert_gif(
    mpath,
    gpath="movie.gif",
    gif_palette=False,
    resolution=[480, 320],
    verbose=False,
    remove_movie=True,
    gif_framerate=5,
):
    if gif_palette:
        palette_filter = (
            '-filter_complex "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse"'
        )
    else:
        palette_filter = ""

    command = "ffmpeg -y -i %s %s -r %i -s %ix%i %s" % (
        mpath,
        palette_filter,
        gif_framerate,
        resolution[0],
        resolution[1],
        gpath,
    )
    p = _check_ffmpeg_execute(command, verbose=verbose)

    print("GIF created at %s" % (gpath))
    if remove_movie:
        if os.path.exists(mpath):
            os.remove(mpath)
    return p


def _combine_ffmpeg_command(
    sourcefolder, moviename, framerate, frame_pattern, ffmpeg_options
):
    # we need `-y` because i can not properly diagnose the errors here...
    command = 'ffmpeg -r %i -i "%s" -y %s -r %i "%s"' % (
        framerate,
        os.path.join(sourcefolder, frame_pattern),
        ffmpeg_options,
        framerate,
        os.path.join(sourcefolder, moviename),
    )
    return command


def write_movie(
    sourcefolder,
    moviename,
    frame_pattern="frame_%05d.png",
    remove_frames=True,
    verbose=False,
    ffmpeg_options="-c:v libvpx-vp9 -b:v 2M -f mp4",
    framerate=20,
):
    command = _combine_ffmpeg_command(
        sourcefolder, moviename, framerate, frame_pattern, ffmpeg_options
    )
    p = _check_ffmpeg_execute(command, verbose=verbose)

    print("Movie created at %s" % (moviename))
    if remove_frames:
        rem_name = frame_pattern.replace("%05d", "*")
        for f in glob.glob(os.path.join(sourcefolder, rem_name)):
            if os.path.exists(f):
                os.remove(f)
    return p


def frame_save(fig, frame, odir=None, frame_pattern="frame_%05d.png", dpi=100):
    fig.savefig(
        os.path.join(odir, frame_pattern % (frame)),
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        transparent=True,
    )
    # I am trying everything to *wipe* this figure, hoping that it could
    # help with the dask glitches I experienced earlier.
    # TBD if this is all needed...how this might affect performance.
    plt.close(fig)
    del fig
    gc.collect(2)


class Movie:
    def __init__(
        self,
        da,
        plotfunc=None,
        framedim="time",
        pixelwidth=1920,
        pixelheight=1080,
        dpi=200,
        frame_pattern="frame_%05d.png",
        fieldname=None,
        input_check=True,
        **kwargs,
    ):
        self.pixelwidth = pixelwidth
        self.pixelheight = pixelheight
        self.dpi = dpi
        self.width = self.pixelwidth / self.dpi
        self.height = self.pixelheight / self.dpi
        self.frame_pattern = frame_pattern
        self.data = da
        self.framedim = framedim
        if plotfunc is None:
            self.plotfunc = basic
        else:
            self.plotfunc = plotfunc
        # set sensible defaults
        self.raw_kwargs = kwargs

        # Check input

        # optional checks (these might need to be deactivated when using custom
        # plot functions.)
        if input_check:
            if isinstance(self.data, xr.Dataset):
                raise ValueError(
                    "xmovie presets do not yet support the input of xr.Datasets. \
                In order to use datasets as inputs, set `input_check` to False. \
                Note that this requires you to manually set colorlimits etc."
                )

            # Set defaults
            self.kwargs = _parse_plot_defaults(self.data, self.raw_kwargs)
        else:
            self.kwargs = self.raw_kwargs

        # Mandatory checks
        # Check if `framedim` exists.
        if self.framedim not in list(self.data.dims):
            raise ValueError("Framedim (%s) not found in input data" % self.framedim)
        # Check the output of plotfunc
        self.plotfunc_n_outargs = _check_plotfunc_output(
            self.plotfunc, self.data, self.framedim, **self.kwargs
        )

    def render_frame(self, timestep, timestep_value):
        """Renders complete figure (frame) for given timestep.

        Parameters
        ----------
        timestep : type
            Description of parameter `timestep`.

        Returns:
        -------
        type
            Description of returned object.

        """
        fig = plt.figure(figsize=[self.width, self.height])
        # create_frame(self.pixelwidth, self.pixelheight, self.dpi)
        # produce dummy output for ax and pp if the plotfunc does not provide them
        if self.plotfunc_n_outargs == 2:
            # this should be the case for all presets provided by xmovie
            ax, pp = self.plotfunc(
                self.data, fig, timestep, timestep_value, self.framedim, **self.kwargs
            )
        else:
            warnings.warn(
                "The provided `plotfunc` does not provide the expected number of output arguments.\
            Expected a function `ax,pp =plotfunc(...)` but got %i output arguments. Inserting dummy values. This should not affect output. ",
                UserWarning,
            )
            _ = self.plotfunc(
                self.data, fig, timestep, timestep_value, self.framedim, **self.kwargs
            )
            ax, pp = None, None
        return fig, ax, pp

    def save_frames(self, odir, progress=False):
        """Save movie frames as picture files.

        Parameters
        ----------
        odir : path
            path to output directory
        progress : type
            Show progress bar. Requires

        """
        # create range of frames
        timesteps = self.data[self.framedim].data
        frame_range = range(len(timesteps))
        if tqdm_avail and progress:
            frame_range = tqdm(frame_range)
        elif ~tqdm_avail and progress:
            warnings.warn("Cant show progess bar at this point. Install tqdm")

        for fi in frame_range:
            fig, ax, pp = self.render_frame(fi, timesteps[fi])
            frame_save(
                fig, fi, odir=odir, frame_pattern=self.frame_pattern, dpi=self.dpi
            )

    # Needs more testing! Slower than the sequential counterpart
    def save_frames_parallel(self, odir, batch_size=10, progress=False):
        """Save frames in parallel batches to reduce I/O overhead."""
        timesteps = self.data[self.framedim].data
        frame_range = range(len(timesteps))

        for i in tqdm(range(0, len(frame_range), batch_size)):
            batch = frame_range[i : i + batch_size]
            tasks = []
            for fi in batch:
                delayed_frame = delayed(self.render_frame)(fi, timesteps[fi])
                tasks.append(
                    delayed(frame_save)(
                        delayed_frame[0],
                        fi,
                        odir=odir,
                        frame_pattern=self.frame_pattern,
                        dpi=self.dpi,
                    )
                )
            compute(*tasks)

    def save(
        self,
        filename,
        remove_frames=True,
        remove_movie=True,
        progress=False,
        verbose=False,
        overwrite_existing=False,
        framerate=15,
        ffmpeg_options="-c:v mjpeg -q:v 2 -pix_fmt yuvj420p",
        gif_palette=False,
        gif_resolution_factor=0.5,
        gif_framerate=10,
        parallel=False,
        batch_size=10,
    ):
        """Save out animation from Movie object.

        Parameters
        ----------
        filename : str
            Pathname to final movie/animation. Output is dependent on filetype:
            Creates movie for `*.mp4` and gif for `*.gif`
        remove_frames : Bool
            Optional removal of frame pictures (the default is True; False will
            leave all picture files in folder).
        remove_movie : Bool
            As `remove_frames` but for movie file. Only applies when filename
            is given as `.gif` (the default is True).
        progress : Bool
            Experimental switch to show progress output. This will be refined
            in future version (the default is False).
        verbose : Bool
            Experimental switch to show output of ffmpeg commands. Useful for
            debugging but can quickly flood your notebook
            (the default is False).
        overwrite_existing : Bool
            Set to overwrite existing files with `filename`
            (the default is False).
        framerate : int
            Frames per second for the output movie file. Only relevant for '.mp4' files.
            (the default is 15)
        ffmpeg_options: str
            Encoding options to pass to ffmpeg call.
            Defaults to : `"-c:v libx264 -preset veryslow -crf 10 -pix_fmt yuv420p"`
        gif_palette : Bool
            Use a gif colorpalette to improve quality. Can lead to artifacts
            in very contrasty situations (the default is False).
        gif_resolution_factor : float
            Factor used to reduce gif resolution compared to movie.
            Use 1.0 to put out the same resolutions for both products.
            (the default is 0.5).
        gif_framerate : int
            As `framerate` but for the gif output file. Only relevant to `.gif` files.
            (the default is 10)
        """
        # parse out directory and filename
        dirname = os.path.dirname(filename)
        filename = os.path.basename(filename)

        # detect gif filename

        isgif = ".gif" in filename
        if isgif:
            giffile = filename
            moviefile = filename.replace("gif", "mp4")
            gpath = os.path.join(dirname, giffile)
        else:
            moviefile = filename

        mpath = os.path.join(dirname, moviefile)

        # check existing files
        if os.path.exists(mpath):
            if not overwrite_existing:
                raise RuntimeError(
                    "File `%s` already exists. Set `overwrite_existing` to True to overwrite."
                    % (mpath)
                )
        if isgif:
            if os.path.exists(gpath):
                if not overwrite_existing:
                    raise RuntimeError(
                        "File `%s` already exists. Set `overwrite_existing` to True to overwrite."
                        % (gpath)
                    )

        # print frames
        if parallel:
            self.save_frames_parallel(dirname, batch_size=batch_size, progress=progress)
        else:
            self.save_frames(dirname, progress=progress)

        # Create movie
        write_movie(
            dirname,
            moviefile,
            frame_pattern=self.frame_pattern,
            remove_frames=remove_frames,
            verbose=verbose,
            framerate=framerate,
            ffmpeg_options=ffmpeg_options,
        )

        # Create gif
        if isgif:
            # if ppath:
            #     create_gif_palette(mpath, ppath=ppath, verbose=verbose)
            convert_gif(
                mpath,
                gpath=gpath,
                resolution=[480, 320],
                gif_palette=gif_palette,
                verbose=verbose,
                remove_movie=remove_movie,
                gif_framerate=gif_framerate,
            )


# #### ENSO

# In[73]:


clim = data["thetao"].sel(lev=slice(0, 500)).groupby("time.dayofyear").mean().compute()
data_surface = data.sel(lev=slice(0, 500))
for k in pred_dict.keys():
    pred_dict[k]["ds_prediction_surface"] = pred_dict[k]["ds_prediction"].sel(
        lev=slice(0, 500)
    )
    pred_dict[k]["clim_pred"] = (
        pred_dict[k]["ds_prediction_surface"]["thetao"]
        .groupby("time.dayofyear")
        .mean()
        .compute()
    )


# In[74]:


def NinoIndexComputeClim2D(T, area, dt=5, window=150):
    T = T.load()
    T_clim = T.copy()
    T_clim = T_clim.sel(x=slice(118, 260), y=slice(-5, 5))
    area = area.sel(x=slice(118, 260), y=slice(-5, 5)).load()
    clim = T_clim.groupby("time.dayofyear").mean("time").compute()
    window = int(window / dt)
    for i, t in enumerate(T_clim.time.values):
        day = int(t.dayofyr)
        T_clim[i] = (T[i] - clim.sel(dayofyear=day)).data

    T_clim = T_clim.rolling(time=window).mean()
    # T_clim = (T_clim*area).sum(["x","y"])/area.sum(["x","y"])
    # T_clim = T_clim.weighted(area).mean(["x", "y"])
    T_clim = T_clim.weighted(area).mean(["y"])

    return T_clim[window:]


# In[75]:


nino_true_compute_clim2D = NinoIndexComputeClim2D(
    data_surface["thetao"], data["areacello"]
)
nino_true_compute_clim2D = nino_true_compute_clim2D.rename("Nino 3.4")
nino_true_compute_clim2D = nino_true_compute_clim2D.assign_attrs(units=r"$\degree C$")

for k in pred_dict.keys():
    pred_dict[k]["nino_pred_compute_clim2D"] = NinoIndexComputeClim2D(
        pred_dict[k]["ds_prediction_surface"]["thetao"],
        pred_dict[k]["ds_prediction"]["areacello"],
    )
    pred_dict[k]["nino_pred_compute_clim2D"] = pred_dict[k][
        "nino_pred_compute_clim2D"
    ].rename("Nino 3.4")
    pred_dict[k]["nino_pred_compute_clim2D"] = pred_dict[k][
        "nino_pred_compute_clim2D"
    ].assign_attrs(units=r"$\degree C$")


# In[76]:


def enso_2d(da, fig, timestamp, timestamp_val, framedim="time", title="", **kwargs):
    da_gt = da[0]
    da_pred1 = da[1]
    da_pred2 = da[2]
    # Plot both one on top of the other
    ax = fig.subplots(3, 1, sharex=True, gridspec_kw={"wspace": 0.02, "hspace": 0.35})
    ax = np.array(ax)
    data_gt = da_gt.isel({framedim: timestamp})
    pp_gt = data_gt.plot.pcolormesh(ax=ax[0], **kwargs)
    ax[0].set_title(dataset_name)
    ax[0].invert_yaxis()
    ax[0].set_xlabel("")

    data_pred1 = da_pred1.isel({framedim: timestamp})
    data_pred1.plot.pcolormesh(ax=ax[1], **kwargs)
    ax[1].set_title(pred_dict[key1]["name"])
    ax[1].invert_yaxis()
    ax[1].set_xlabel("")

    data_pred2 = da_pred2.isel({framedim: timestamp})
    data_pred2.plot.pcolormesh(ax=ax[2], **kwargs)
    ax[2].set_title(pred_dict[key2]["name"])
    ax[2].invert_yaxis()

    # Text for current date based on timestamp
    ax[0].text(
        1.0,
        1.2,
        f"{timestamp_val.year}-{timestamp_val.month}-{timestamp_val.day}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
    )

    return ax, pp_gt


# In[77]:


# combine the two datasets into a single xarray new dimension
da = xr.concat(
    [
        nino_true_compute_clim2D,
        pred_dict[key1]["nino_pred_compute_clim2D"],
        pred_dict[key2]["nino_pred_compute_clim2D"],
    ],
    dim="dummy",
)
mov = Movie(
    da, plotfunc=enso_2d, y="lev", cmap=cm.cm.balance, vmin=-2, vmax=2, title="ENSO"
)
mov.save(
    os.path.join(movie_path, f"enso_2d_movie.mp4"),
    progress=True,
    overwrite_existing=True,
)


# #### Global Map

# In[78]:


import os

import cartopy.crs as ccrs
import cmocean as cm
import matplotlib.pyplot as plt
import numpy as np


def global_surface_map(da, fig, timestamp, timestamp_val, framedim="time", **kwargs):
    var_name = kwargs["var_name"]

    da_gt = da[0]
    da_pred1 = da[1]
    da_pred2 = da[2]

    data_gt = da_gt.isel({framedim: timestamp})
    data_pred1 = da_pred1.isel({framedim: timestamp})
    data_pred2 = da_pred2.isel({framedim: timestamp})

    plt.clf()
    plt.rcParams.update({"font.size": 14})

    # Define colormap
    new_cmap = (
        cm.cm.balance if var_name == "thetao" or var_name == "OHC" else cm.cm.haline
    )
    new_cmap.set_bad(color="grey", alpha=0.0)
    # Set common color range for the colorbar
    vmin, vmax = {
        "thetao": (0, 30),
        "so": (30, 40),
        "uo": (-2, 2),
        "vo": (-2, 2),
        "zos": (-1, 1),
        "OHC": (-0.05, 0.05),
    }[var_name]

    # Create figure with appropriate layout
    ax = fig.subplots(
        1,
        3,
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"wspace": 0.02, "hspace": 0.05},
    )
    ax = np.array(ax)  # Ensure ax is an array for easy indexing

    # Plot Ground Truth (GT)
    im = data_gt.plot(
        ax=ax[0],
        cmap=new_cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax[0].add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.1)
    ax[0].set_title(f"{dataset_name}", fontsize=14)
    gl = ax[0].gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    # Plot Predictions
    im = data_pred1.plot(
        ax=ax[1],
        cmap=new_cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax[1].add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.1)
    ax[1].set_title(pred_dict[key1]["name"], fontsize=14)
    gl = ax[1].gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])
    ax[1].set_yticks([])
    ax[1].set_ylabel("")

    im = data_pred2.plot(
        ax=ax[2],
        cmap=new_cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax[2].add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.1)
    ax[2].set_title(pred_dict[key2]["name"], fontsize=14)
    gl = ax[2].gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])
    ax[2].set_yticks([])
    ax[2].set_ylabel("")

    # Add colorbar for plots
    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(var_list[var_name])

    # Add timestamp text
    ax[0].text(
        1.0,
        1.4,
        f"{timestamp_val.year}-{timestamp_val.month:02d}-{timestamp_val.day:02d}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
    )

    return ax, im


def global_surface_map_bias(
    da, fig, timestamp, timestamp_val, framedim="time", **kwargs
):
    var_name = kwargs["var_name"]

    da_pred1 = da[0]
    da_pred2 = da[1]

    data_pred1 = da_pred1.isel({framedim: timestamp})
    data_pred2 = da_pred2.isel({framedim: timestamp})

    plt.clf()
    plt.rcParams.update({"font.size": 14})

    # Define colormap
    new_cmap = cm.cm.balance
    new_cmap.set_bad(color="grey", alpha=0.0)

    # Set common color range for the colorbar
    vmin, vmax = {
        "thetao": (-5, 5),
        "so": (-1, 1),
        "uo": (-0.01, 0.01),
        "vo": (-0.01, 0.01),
        "zos": (-1, 1),
        "OHC": (-0.1, 0.1),
    }[var_name]

    # Create figure with appropriate layout
    ax = fig.subplots(
        1,
        2,
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"wspace": 0.02, "hspace": 0.05},
    )
    ax = np.array(ax)  # Ensure ax is an array for easy indexing

    # Plot Predictions
    im = data_pred1.plot(
        ax=ax[0],
        cmap=new_cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax[0].add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.1)
    ax[0].set_title(pred_dict[key1]["name"] + " Bias", fontsize=14)
    gl = ax[0].gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])

    im = data_pred2.plot(
        ax=ax[1],
        cmap=new_cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    ax[1].add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=0.1)
    ax[1].set_title(pred_dict[key2]["name"] + " Bias", fontsize=14)
    gl = ax[1].gridlines(draw_labels=True, color="0.4", linestyle="--", alpha=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.xlabel_style = {"size": 14}
    gl.ylabel_style = {"size": 14}
    gl.xlocator = FixedLocator([-120, -60, 0, 60, 120])
    ax[1].set_yticks([])
    ax[1].set_ylabel("")

    # Add colorbar for plots
    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(var_list[var_name])

    # Add timestamp text
    ax[0].text(
        1.0,
        1.4,
        f"{timestamp_val.year}-{timestamp_val.month:02d}-{timestamp_val.day:02d}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
    )

    return ax, im


# In[ ]:


# combine the two datasets into a single xarray new dimension
movie_var_list = ["thetao", "so", "OHC"]

c_p = 3850  # J/(kg C)
rho_0 = 1025  # kg/m^3
zeta_joules_factor = 1e21

for var in movie_var_list:
    if var == "OHC":
        ohc_gt = (
            (data["thetao"] * c_p * rho_0 / zeta_joules_factor)
            .weighted(data["areacello"] * data["dz"])
            .sum(["lev"])
            .compute()
        )
        data_gt = remove_climatology(ohc_gt)

        ohc_pred1 = (
            (
                pred_dict[key1]["ds_prediction"]["thetao"]
                * c_p
                * rho_0
                / zeta_joules_factor
            )
            .weighted(data["areacello"] * data["dz"])
            .sum(["lev"])
            .compute()
        )
        data_pred1 = remove_climatology(ohc_pred1)

        ohc_pred2 = (
            (
                pred_dict[key2]["ds_prediction"]["thetao"]
                * c_p
                * rho_0
                / zeta_joules_factor
            )
            .weighted(data["areacello"] * data["dz"])
            .sum(["lev"])
            .compute()
        )
        data_pred2 = remove_climatology(ohc_pred2)

    else:
        data_gt = data[var].isel(lev=0).compute()
        data_pred1 = pred_dict[key1]["ds_prediction"][var].isel(lev=0).compute()
        data_pred2 = pred_dict[key2]["ds_prediction"][var].isel(lev=0).compute()

    data_gt = data_gt.where(data.wetmask.isel(lev=0)).compute()
    data_pred1 = data_pred1.where(data.wetmask.isel(lev=0)).compute()
    data_pred2 = data_pred2.where(data.wetmask.isel(lev=0)).compute()

    da = xr.concat([data_gt, data_pred1, data_pred2], dim="dummy")
    mov = Movie(da, plotfunc=global_surface_map, var_name=var, input_check=False)
    mov.save(
        os.path.join(movie_path, f"{var}_surface_map_movie.mp4"),
        progress=True,
        overwrite_existing=True,
    )

    # Bias
    da = xr.concat(
        [(data_pred1 - data_gt).compute(), (data_pred2 - data_gt).compute()],
        dim="dummy",
    )
    mov = Movie(da, plotfunc=global_surface_map_bias, var_name=var, input_check=False)
    mov.save(
        os.path.join(movie_path, f"{var}_surface_map_bias_movie.mp4"),
        progress=True,
        overwrite_existing=True,
    )


# #### Global Profile

# In[ ]:


import os

import cmocean as cm
import matplotlib.pyplot as plt
import numpy as np
from xarrayutils.plotting import linear_piecewise_scale


def global_profile(da, fig, timestamp, timestamp_val, framedim="time", **kwargs):
    da_gt = da[0]
    da_pred1 = da[1]
    da_pred2 = da[2]

    data_gt = da_gt.isel({framedim: timestamp})
    data_pred1 = da_pred1.isel({framedim: timestamp})
    data_pred2 = da_pred2.isel({framedim: timestamp})
    var_name = kwargs["var_name"]

    plt.clf()
    plt.rcParams.update({"font.size": 14})

    # Define colormap
    new_cmap = cm.cm.thermal if var_name == "thetao" else cm.cm.haline
    new_cmap.set_bad("grey", 0.6)

    # Create figure with appropriate layout
    ax = fig.subplots(
        3,
        1,
        gridspec_kw={
            "width_ratios": [1] * 1,
            "height_ratios": [1] * 3,
            "wspace": 0.02,
            "hspace": 0.5,
        },
    )
    ax = np.array(ax)  # Ensure ax is an array for easy indexing

    # Set common color range for the colorbar
    vmin, vmax = {
        "thetao": (0, 30),
        "so": (30, 40),
        "uo": (-2, 2),
        "vo": (-2, 2),
        "zos": (-1, 1),
    }[var_name]

    # Plot GT (original data)
    im = data_gt.plot(
        ax=ax[0], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False, y="lev"
    )
    ax[0].invert_yaxis()
    ax[0].set_title(dataset_name, fontsize=14)
    ax[0].set_xticks([])
    ax[0].set_xlabel("")

    # Plot predictions from other models
    im = data_pred1.plot(
        ax=ax[1], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False, y="lev"
    )
    ax[1].invert_yaxis()
    ax[1].set_title(pred_dict[key1]["name"], fontsize=14)
    ax[1].set_xticks([])
    ax[1].set_xlabel("")

    im = data_pred2.plot(
        ax=ax[2], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False, y="lev"
    )
    ax[2].invert_yaxis()
    ax[2].set_title(pred_dict[key2]["name"], fontsize=14)

    # Add shared colorbar for each row
    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(var_list[var_name])

    # Text for current date based on timestamp
    ax[0].text(
        1.0,
        1.2,
        f"{timestamp_val.year}-{timestamp_val.month}-{timestamp_val.day}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
    )

    return ax, im


def global_profile_bias(da, fig, timestamp, timestamp_val, framedim="time", **kwargs):
    da_pred1 = da[0]
    da_pred2 = da[1]

    data_pred1 = da_pred1.isel({framedim: timestamp})
    data_pred2 = da_pred2.isel({framedim: timestamp})
    var_name = kwargs["var_name"]

    plt.clf()
    plt.rcParams.update({"font.size": 14})

    # Define colormap
    new_cmap = cm.cm.balance
    new_cmap.set_bad("grey", 0.6)

    # Create figure with appropriate layout
    ax = fig.subplots(
        2,
        1,
        gridspec_kw={
            "width_ratios": [1] * 1,
            "height_ratios": [1] * 2,
            "wspace": 0.02,
            "hspace": 0.5,
        },
    )
    ax = np.array(ax)  # Ensure ax is an array for easy indexing

    # Set common color range for the colorbar
    vmin, vmax = {
        "thetao": (-2, 2),
        "so": (-0.1, 0.1),
        "uo": (-0.01, 0.01),
        "vo": (-0.01, 0.01),
        "zos": (-1, 1),
    }[var_name]

    # Plot predictions from other models
    im = data_pred1.plot(
        ax=ax[0], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False, y="lev"
    )
    ax[0].invert_yaxis()
    ax[0].set_title(pred_dict[key1]["name"] + " Bias", fontsize=14)
    ax[0].set_xticks([])
    ax[0].set_xlabel("")

    im = data_pred2.plot(
        ax=ax[1], cmap=new_cmap, vmin=vmin, vmax=vmax, add_colorbar=False, y="lev"
    )
    ax[1].invert_yaxis()
    ax[1].set_title(pred_dict[key2]["name"] + " Bias", fontsize=14)

    # Add shared colorbar for each row
    cbar = fig.colorbar(im, ax=ax[:], orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(var_list[var_name])

    # Text for current date based on timestamp
    ax[0].text(
        1.0,
        1.2,
        f"{timestamp_val.year}-{timestamp_val.month}-{timestamp_val.day}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
    )

    return ax, im


# In[ ]:


# combine the two datasets into a single xarray new dimension

movie_var_list = ["thetao", "so"]
for var in movie_var_list:
    # GT
    da_gt = data[var].sel(lev=slice(0, 500))
    section_mask = np.isnan(da_gt).all("x").isel(time=0)
    da_gt_int_x = da_gt.weighted(da_gt["areacello"]).mean(["x"])
    var_gt = da_gt_int_x.where(~section_mask)
    var_gt = var_gt.rename(var_list[var]).assign_attrs(units=var_list[var])
    var_gt["y"] = var_gt.y.assign_attrs(long_name="latitude", units=r"$\degree$")
    var_gt["lev"] = var_gt.lev.assign_attrs(long_name="depth", units="m")
    var_gt = var_gt.compute()

    # Prediction
    da_pred = pred_dict[key1]["ds_prediction"][var].sel(lev=slice(0, 500))
    section_mask = np.isnan(da_pred).all("x").isel(time=0)
    da_pred_int_x = da_pred.weighted(da_pred["areacello"]).mean(["x"])
    var_pred = da_pred_int_x.where(~section_mask)
    var_pred = var_pred.rename(var_list[var]).assign_attrs(units=var_list[var])
    var_pred["y"] = var_pred.y.assign_attrs(long_name="latitude", units=r"$\degree$")
    var_pred["lev"] = var_pred.lev.assign_attrs(long_name="depth", units="m")
    var_pred1 = var_pred.compute()

    da_pred = pred_dict[key2]["ds_prediction"][var].sel(lev=slice(0, 500))
    section_mask = np.isnan(da_pred).all("x").isel(time=0)
    da_pred_int_x = da_pred.weighted(da_pred["areacello"]).mean(["x"])
    var_pred = da_pred_int_x.where(~section_mask)
    var_pred = var_pred.rename(var_list[var]).assign_attrs(units=var_list[var])
    var_pred["y"] = var_pred.y.assign_attrs(long_name="latitude", units=r"$\degree$")
    var_pred["lev"] = var_pred.lev.assign_attrs(long_name="depth", units="m")
    var_pred2 = var_pred.compute()

    # Save
    da = xr.concat([var_gt, var_pred1, var_pred2], dim="dummy")
    mov = Movie(da, plotfunc=global_profile, var_name=var, input_check=False)
    mov.save(
        os.path.join(movie_path, f"{var}_profile_movie.mp4"),
        progress=True,
        overwrite_existing=True,
    )

    # Bias
    da = xr.concat(
        [(var_pred1 - var_gt).compute(), (var_pred2 - var_gt).compute()], dim="dummy"
    )
    mov = Movie(da, plotfunc=global_profile_bias, var_name=var, input_check=False)
    mov.save(
        os.path.join(movie_path, f"{var}_profile_bias_movie.mp4"),
        progress=True,
        overwrite_existing=True,
    )


# In[ ]:


import datetime

datetime.datetime.now()


# In[ ]:


# In[ ]:


# In[ ]:
