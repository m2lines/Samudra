#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8

# In[14]:


import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# from xarrayutils.plotting import linear_piecewise_scale


# In[37]:


years = 400

# 2002 No warming
Pred_path_temp = "/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2024-09-25_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHfTempOnly1975Epochs70Epoch55Years100_10repeat_2002_Train_global_3D_Test_global_3D_all_N_train_0_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_0_rand_seed_1.zarr"  # temp no warming
label_temp = "2002-2012 - Slow"
Pred_path_all = "/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2024-09-25_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHf1975Epochs70Epoch55Years100_10repeat_2002_Train_global_3D_Test_global_3D_all_N_train_0_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_0_rand_seed_1.zarr"  # all no warming
label_all = "2002-2012 - Slow+Fast"
start_year = 2002

# 1998 No warming
# Pred_path_temp = '/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2024-09-26_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHfTempOnly1975Epochs70Epoch55Years100_10repeat_1998_Train_global_3D_Test_global_3D_all_N_train_0_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_0_rand_seed_1.zarr' # temp no warming
# label_temp = '1998-2008 - Slow'
# Pred_path_all = '/pscratch/sd/s/suryad/Ocean_Emulator/Preds/2024-09-25_ConvNextUNetTrain3Dv021Eval3DhfdsanomsNetZeroHf1975Epochs70Epoch55Years100_10repeat_1998_Train_global_3D_Test_global_3D_all_N_train_0_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_0_rand_seed_1.zarr' # all no warming
# label_all = '1998-2008 - Slow+Fast'
# start_year = 1998

warming = False
if warming:
    suffix = "_warming"
else:
    suffix = ""


# In[38]:


suffix = "very_long"


# In[39]:


def post_processor(ds: xr.Dataset, ds_truth: xr.Dataset, ls) -> xr.Dataset:
    """Converts the prediction output to an xarray dataset with the same dimensions/variables as input"""
    # Always run the ds_input_validate in non-deep mode here

    # correct swapped dimensions and warn
    if len(ds.x) == 180 and len(ds.y) == 360:
        ds = ds.rename({"x": "x_i", "y": "y_i"}).rename({"x_i": "y", "y_i": "x"})

    da = ds["__xarray_dataarray_variable__"]
    n_lev = 19
    if set(ls) - {"zos"} == set(["uo", "vo", "thetao", "so"]):
        variables = ["uo", "vo", "thetao", "so"]
    elif set(ls) - {"zos"} == set(["thetao", "so"]):
        variables = ["thetao", "so"]
    elif set(ls) - {"zos"} == set(["uo", "vo"]):
        variables = ["uo", "vo"]
    slices = [slice(i, i + n_lev) for i in range(0, len(variables) * n_lev, n_lev)]
    var_slices = {k: sl for k, sl in zip(variables, slices)}
    variables = {
        k: da.isel(var=sl).rename({"var": "lev"}) for k, sl in var_slices.items()
    }
    variables["zos"] = da.isel(var=-1).squeeze()

    ds_out = xr.Dataset(variables)
    for var in ds_out.data_vars:
        if "lev" in ds_out[var].dims:
            ds_out[var] = ds_out[var].where(ds_truth.wetmask)
        else:
            ds_out[var] = ds_out[var].where(ds_truth.wetmask.isel(lev=0))

    ## attach all coordinates from input
    ds_out = ds_out.assign_coords({co: ds_truth[co] for co in ds_truth.coords})

    return ds_out


levels = 19
emulation_stability = True
smooth = False

# OM4 v0.2.1
ds_input = xr.open_zarr(
    os.path.join("/pscratch/sd/s/suryad/data", "OM4_5daily_v0.2.1.zarr")
)

# Smooth the data
if smooth:
    window = 10
    with ProgressBar():
        ds_input["uo"] = (
            ds_input.uo.rolling(time=window, min_periods=1, center=False)
            .mean()
            .compute()
        )
        ds_input["vo"] = (
            ds_input.vo.rolling(time=window, min_periods=1, center=False)
            .mean()
            .compute()
        )


# our groundtruth is always just a time slice of the training (training is a bad name

if emulation_stability:
    repeats = 3000
    ds_groundtruth = ds_input.isel(lev=slice(None, levels))
    ds_groundtruth = ds_groundtruth.sel(time=slice("1996-01-01", "1996-12-31"))
    new_time = np.arange(ds_groundtruth.time.size * repeats)
    ds_groundtruth = xr.concat([ds_groundtruth] * repeats, dim="time")
    ds_groundtruth["time"] = new_time
    ds_groundtruth = ds_groundtruth.isel(time=slice(3, 30000))

else:
    ds_groundtruth = ds_input.isel(time=slice(2903, 3503)).isel(lev=slice(None, levels))

ls_temp = ["thetao", "so", "zos"]
ls_all = ["uo", "vo", "thetao", "so", "zos"]

output_path = "../outputs/" + label_temp + "_" + label_all
print("Using Output Folder : ", output_path)
if not os.path.isdir(os.path.join(output_path)):
    os.makedirs(os.path.join(output_path))


ds_prediction_raw_temp = xr.open_zarr(Pred_path_temp)
ds_prediction_raw_all = xr.open_zarr(Pred_path_all)

# if emulation_stability:
#     ds_groundtruth = ds_groundtruth.isel(time=slice(0, ds_prediction_all_raw.time.size))

ds_prediction_all = post_processor(
    ds_prediction_raw_all,
    ds_groundtruth.isel(time=slice(0, ds_prediction_raw_all.time.size)),
    ls_all,
)

ds_prediction_temp = post_processor(
    ds_prediction_raw_temp,
    ds_groundtruth.isel(time=slice(0, ds_prediction_raw_temp.time.size)),
    ls_temp,
)


# Run the test to make sure the output is formatted correctly
ds_prediction_all = ds_prediction_all.transpose("time", "lev", ...)
ds_prediction_temp = ds_prediction_temp.transpose("time", "lev", ...)


# In[40]:


from datetime import timedelta

from cftime import DatetimeProlepticGregorian

dates = np.array(range(3, 365 * years, 5))
base = DatetimeProlepticGregorian(start_year, 1, 1)
all_dates = [base + timedelta(days=int(day - 1)) for day in dates]

ds_prediction_temp = ds_prediction_temp.assign_coords(
    time=all_dates[: ds_prediction_temp.time.size]
)
ds_prediction_all = ds_prediction_all.assign_coords(
    time=all_dates[: ds_prediction_all.time.size]
)


# In[41]:


ds_prediction_temp["y"] = ds_prediction_temp.y.assign_attrs(long_name="latitude")
ds_prediction_temp["x"] = ds_prediction_temp.x.assign_attrs(long_name="longitude")
ds_prediction_temp["thetao"] = ds_prediction_temp["thetao"].assign_attrs(
    long_name="Temperature", units=r"${^oC}$"
)

ds_prediction_all["y"] = ds_prediction_all.y.assign_attrs(long_name="latitude")
ds_prediction_all["x"] = ds_prediction_all.x.assign_attrs(long_name="longitude")
ds_prediction_all["thetao"] = ds_prediction_all["thetao"].assign_attrs(
    long_name="Temperature", units=r"${^oC}$"
)


# In[42]:


# ds_prediction_temp = ds_prediction_temp.isel(time=slice(28000, None))


# In[43]:


color_1 = "#DE3A41"
color_2 = "#277DC7"


# In[44]:


# Compute Basin Heat Content Time Series

Days_to_Eq = 0
c_p = 3850  # J/(kg C)
rho_0 = 1025  # kg/m^3

fig, ax = plt.subplots(
    3, 1, figsize=(10, 7.5), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
)

OHC_pred_upper = (
    (
        (ds_prediction_all["thetao"].sel(lev=slice(0, 700)) * c_p * rho_0)
        * ds_prediction_all["areacello"]
        * ds_prediction_all["dz"]
    )
    .sum(["x", "y", "lev"])
    .compute()
)
OHC_pred_upper = OHC_pred_upper.rename("OHC Upper 700m")
OHC_pred_upper = OHC_pred_upper.assign_attrs(units="J")

OHC_pred_upper.plot(ax=ax[0], label=label_all, c=color_1)

OHC_pred_upper_temp = (
    (
        (ds_prediction_temp["thetao"].sel(lev=slice(0, 700)) * c_p * rho_0)
        * ds_prediction_all["areacello"]
        * ds_prediction_all["dz"]
    )
    .sum(["x", "y", "lev"])
    .compute()
)
OHC_pred_upper_temp = OHC_pred_upper_temp.rename("OHC Upper 700m")
OHC_pred_upper_temp = OHC_pred_upper_temp.assign_attrs(units="J")

OHC_pred_upper_temp.plot(ax=ax[0], label=label_temp, c=color_2)

# ax[0].legend()
ax[0].set_title("Ocean Heat Content")
coeffs_OHC_pred_trend = np.polyfit(
    np.arange(OHC_pred_upper[Days_to_Eq:].size), OHC_pred_upper[Days_to_Eq:], 1
)
ax[0].plot(
    OHC_pred_upper[Days_to_Eq:].time.data,
    np.arange(OHC_pred_upper[Days_to_Eq:].size) * coeffs_OHC_pred_trend[0]
    + coeffs_OHC_pred_trend[1],
    c=color_1,
    ls="--",
)
coeffs_OHC_pred_trend_temp = np.polyfit(
    np.arange(OHC_pred_upper_temp[Days_to_Eq:].size),
    OHC_pred_upper_temp[Days_to_Eq:],
    1,
)
ax[0].plot(
    OHC_pred_upper_temp[Days_to_Eq:].time.data,
    np.arange(OHC_pred_upper_temp[Days_to_Eq:].size) * coeffs_OHC_pred_trend_temp[0]
    + coeffs_OHC_pred_trend_temp[1],
    c=color_2,
    ls="--",
)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)
ax[0].legend()

upper_trend = coeffs_OHC_pred_trend[0] * 73
upper_trend_temp = coeffs_OHC_pred_trend_temp[0] * 73


OHC_pred_mid = (
    (
        (ds_prediction_all["thetao"].sel(lev=slice(0, 2000)) * c_p * rho_0)
        * ds_prediction_all["areacello"]
        * ds_prediction_all["dz"]
    )
    .sum(["x", "y", "lev"])
    .compute()
)
OHC_pred_mid = OHC_pred_mid.rename("OHC Upper 2000m")
OHC_pred_mid = OHC_pred_mid.assign_attrs(units="J")

OHC_pred_mid.plot(ax=ax[1], label=label_all, c=color_1)

OHC_pred_mid_temp = (
    (
        (ds_prediction_temp["thetao"].sel(lev=slice(0, 2000)) * c_p * rho_0)
        * ds_prediction_all["areacello"]
        * ds_prediction_all["dz"]
    )
    .sum(["x", "y", "lev"])
    .compute()
)
OHC_pred_mid_temp = OHC_pred_mid_temp.rename("OHC Upper 2000m")
OHC_pred_mid_temp = OHC_pred_mid_temp.assign_attrs(units="J")

OHC_pred_mid_temp.plot(ax=ax[1], label=label_temp, c=color_2)

ax[1].set_title("Ocean Heat Content")
coeffs_OHC_pred_trend = np.polyfit(
    np.arange(OHC_pred_mid[Days_to_Eq:].size), OHC_pred_mid[Days_to_Eq:], 1
)
ax[1].plot(
    OHC_pred_mid[Days_to_Eq:].time.data,
    np.arange(OHC_pred_mid[Days_to_Eq:].size) * coeffs_OHC_pred_trend[0]
    + coeffs_OHC_pred_trend[1],
    c=color_1,
    ls="--",
)
coeffs_OHC_pred_trend_temp = np.polyfit(
    np.arange(OHC_pred_mid_temp[Days_to_Eq:].size), OHC_pred_mid_temp[Days_to_Eq:], 1
)
ax[1].plot(
    OHC_pred_mid_temp[Days_to_Eq:].time.data,
    np.arange(OHC_pred_mid_temp[Days_to_Eq:].size) * coeffs_OHC_pred_trend_temp[0]
    + coeffs_OHC_pred_trend_temp[1],
    c=color_2,
    ls="--",
)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)

mid_trend = coeffs_OHC_pred_trend[0] * 73
mid_trend_temp = coeffs_OHC_pred_trend_temp[0] * 73


OHC_pred_deep = (
    (
        (ds_prediction_all["thetao"].sel(lev=slice(2000, None)) * c_p * rho_0)
        * ds_prediction_all["areacello"]
        * ds_prediction_all["dz"]
    )
    .sum(["x", "y", "lev"])
    .compute()
)
OHC_pred_deep = OHC_pred_deep.rename("OHC 2km to bottom")
OHC_pred_deep = OHC_pred_deep.assign_attrs(units="J")

OHC_pred_deep.plot(ax=ax[2], label=label_all, c=color_1)

OHC_pred_deep_temp = (
    (
        (ds_prediction_temp["thetao"].sel(lev=slice(2000, None)) * c_p * rho_0)
        * ds_prediction_all["areacello"]
        * ds_prediction_all["dz"]
    )
    .sum(["x", "y", "lev"])
    .compute()
)
OHC_pred_deep_temp = OHC_pred_deep_temp.rename("OHC 2km to bottom")
OHC_pred_deep_temp = OHC_pred_deep_temp.assign_attrs(units="J")

OHC_pred_deep_temp.plot(ax=ax[2], label=label_temp, c=color_2)

# ax[0].legend()
ax[2].set_title("Ocean Heat Content")
coeffs_OHC_pred_trend = np.polyfit(
    np.arange(OHC_pred_deep[Days_to_Eq:].size), OHC_pred_deep[Days_to_Eq:], 1
)
ax[2].plot(
    OHC_pred_deep[Days_to_Eq:].time.data,
    np.arange(OHC_pred_deep[Days_to_Eq:].size) * coeffs_OHC_pred_trend[0]
    + coeffs_OHC_pred_trend[1],
    c=color_1,
    ls="--",
)
coeffs_OHC_pred_trend_temp = np.polyfit(
    np.arange(OHC_pred_deep_temp[Days_to_Eq:].size), OHC_pred_deep_temp[Days_to_Eq:], 1
)
ax[2].plot(
    OHC_pred_deep_temp[Days_to_Eq:].time.data,
    np.arange(OHC_pred_deep_temp[Days_to_Eq:].size) * coeffs_OHC_pred_trend_temp[0]
    + coeffs_OHC_pred_trend_temp[1],
    c=color_2,
    ls="--",
)
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=0)

deep_trend = coeffs_OHC_pred_trend[0] * 73
deep_trend_temp = coeffs_OHC_pred_trend_temp[0] * 73

total_trend = upper_trend + mid_trend + deep_trend
total_trend_temp = upper_trend_temp + mid_trend_temp + deep_trend_temp

print(f"OHC portion of upper trend: {upper_trend / total_trend:.2f}")
print(f"OHC portion of mid trend: {mid_trend / total_trend:.2f}")
print(f"OHC portion of deep trend: {deep_trend / total_trend:.2f}")
print(f"OHC portion of upper trend temp: {upper_trend_temp / total_trend_temp:.2f}")
print(f"OHC portion of mid trend temp: {mid_trend_temp / total_trend_temp:.2f}")
print(f"OHC portion of deep trend temp: {deep_trend_temp / total_trend_temp:.2f}")

plt.savefig(
    os.path.join(output_path, "Depth_OHC_" + suffix + ".png"),
    bbox_inches="tight",
    dpi=150,
)
# plt.show()


# In[45]:


Days_to_Eq = 0

c_p = 3850  # J/(kg C)
rho_0 = 1025  # kg/m^3

fig, ax = plt.subplots(
    2, 1, figsize=(10, 5), gridspec_kw={"wspace": 0.25, "hspace": 0.5}
)

OHC_pred = (
    (
        (ds_prediction_all["thetao"][Days_to_Eq:] * c_p * rho_0)
        * ds_prediction_temp["areacello"]
        * ds_prediction_temp["dz"]
    )
    .sum(["x", "y", "lev"])
    .compute()
)
OHC_pred = OHC_pred.rename("Ocean Heat Content")
OHC_pred = OHC_pred.assign_attrs(units="J")

OHC_pred.plot(ax=ax[0], label=label_all, c=color_1)

OHC_pred_temp = (
    (
        (ds_prediction_temp["thetao"][Days_to_Eq:] * c_p * rho_0)
        * ds_prediction_temp["areacello"]
        * ds_prediction_temp["dz"]
    )
    .sum(["x", "y", "lev"])
    .compute()
)
OHC_pred_temp = OHC_pred_temp.rename("Ocean Heat Content")
OHC_pred_temp = OHC_pred_temp.assign_attrs(units="J")

OHC_pred_temp.plot(ax=ax[0], label=label_temp, c=color_2)

# ax[0].legend()
ax[0].set_title("Ocean Heat Content")
coeffs_OHC_pred_trend = np.polyfit(np.arange(OHC_pred[:].size), OHC_pred[:], 1)
ax[0].plot(
    OHC_pred[:].time.data,
    np.arange(OHC_pred[:].size) * coeffs_OHC_pred_trend[0] + coeffs_OHC_pred_trend[1],
    c=color_1,
    ls="--",
)
coeffs_OHC_pred_trend_temp = np.polyfit(
    np.arange(OHC_pred_temp[:].size), OHC_pred_temp[:], 1
)
ax[0].plot(
    OHC_pred_temp[:].time.data,
    np.arange(OHC_pred_temp[:].size) * coeffs_OHC_pred_trend_temp[0]
    + coeffs_OHC_pred_trend_temp[1],
    c=color_2,
    ls="--",
)
ax[0].legend(loc="lower right")

salinity = (
    (
        (ds_prediction_all["so"][Days_to_Eq:] * rho_0)
        * ds_prediction_temp["areacello"]
        * ds_prediction_temp["dz"]
    )
    .sum(["x", "y", "lev"])
    .compute()
)
salinity = salinity.rename("Salinity")
salinity = salinity.assign_attrs(units="g")

salinity.plot(ax=ax[1], label=label_all, c=color_1)

salinity_temp = (
    (
        (ds_prediction_temp["so"][Days_to_Eq:] * rho_0)
        * ds_prediction_temp["areacello"]
        * ds_prediction_temp["dz"]
    )
    .sum(["x", "y", "lev"])
    .compute()
)
salinity_temp = salinity_temp.rename("Salinity")
salinity_temp_temp = salinity_temp.assign_attrs(units="g")

salinity_temp.plot(ax=ax[1], label=label_temp, c=color_2)

# ax[1].legend()
ax[1].set_title("Ocean Total Salinity")
coeffs_salinity_trend = np.polyfit(np.arange(salinity[:].size), salinity[:], 1)
ax[1].plot(
    salinity[:].time.data,
    np.arange(salinity[:].size) * coeffs_salinity_trend[0] + coeffs_salinity_trend[1],
    c=color_1,
    ls="--",
)

coeffs_salinity_trend_temp = np.polyfit(
    np.arange(salinity_temp[:].size), salinity_temp[:], 1
)
ax[1].plot(
    salinity_temp[:].time.data,
    np.arange(salinity_temp[:].size) * coeffs_salinity_trend_temp[0]
    + coeffs_salinity_trend_temp[1],
    c=color_2,
    ls="--",
)

print(coeffs_OHC_pred_trend[0] * 73 / 1e21)

print(coeffs_salinity_trend[0] * 73 / 1e17)

print(coeffs_OHC_pred_trend_temp[0] * 73 / 1e21)

print(coeffs_salinity_trend_temp[0] * 73 / 1e17)

plt.savefig(
    os.path.join(output_path, "OHC_Salinity_" + suffix + ".png"),
    bbox_inches="tight",
    dpi=150,
)
# plt.show()


# In[ ]:


# In[ ]:
