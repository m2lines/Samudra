#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8

# In[13]:


import xarray as xr


# In[14]:


old_data = xr.open_zarr(
    "/pscratch/sd/s/suryad/data/CM4_5daily_v0.4.0_preprocessed.zarr", chunks={}
)


# In[15]:


data = xr.open_zarr(
    "/pscratch/sd/s/suryad/data/cm4_piControl_ocean_200yr_full_chunked.zarr", chunks={}
)
# cm4_piControl_ocean_200yr_full_chunked.zarr
data


# In[16]:


import sys

sys.path.append("/pscratch/sd/s/suryad/Ocean_Emulator/src")

from utils.data import spherical_area_weights


# In[19]:


v = "mask"
levels = old_data.lev.values
level_numbers = [i for i in range(19)]
sorted_vars = [v + "_" + str(lev) for lev in level_numbers]
mask = xr.concat([data[var] for var in sorted_vars], dim="lev")
mask = mask.assign_coords(lev=levels)


# In[17]:


v = "thetao"
levels = old_data.lev.values
level_numbers = [i for i in range(19)]
sorted_vars = [v + "_" + str(lev) for lev in level_numbers]
thetao = xr.concat([data[var] for var in sorted_vars], dim="lev")
thetao = thetao.assign_coords(lev=levels)
thetao = thetao.rename("thetao")
# areacello = spherical_area_weights(thetao)
thetao = thetao.transpose("time", "lev", ...)
thetao["areacello"] = (["lat", "lon"], old_data.areacello.values)
thetao["dz"] = ("lev", old_data.dz.values)


# In[21]:


thetao = thetao.where(mask > 0)


# In[22]:


thetao


# In[23]:


hfds = data["hfds"]


# In[24]:


hfds = hfds.where(mask.isel(lev=0) > 0)


# In[26]:


# thetao


# In[27]:


# def remove_climatology(ds):
#     # Compute the climatology on the detrended data
#     climatology = ds.groupby("time.dayofyear").mean("time").compute()

#     # Remove the seasonal cycle (climatology) from the detrended data
#     day_of_year = ds["time"].dt.dayofyear
#     res = (ds - climatology.sel(dayofyear=day_of_year)).compute()

#     return res


# In[25]:


c_p = 3850  # J/(kg C)
rho_0 = 1025  # kg/m^3

OHC = (
    ((thetao * c_p * rho_0) * thetao["areacello"] * thetao["dz"])
    .sum(["lat", "lon", "lev"])
    .compute()
)
OHC = OHC.rename("OHC")
OHC = OHC.assign_attrs(units="J")


# In[30]:


time = 5 * 24 * 60 * 60
hfds_int = (
    (hfds.cumsum("time") * thetao["areacello"] * time).sum(["lat", "lon"]).compute()
)


# In[ ]:


# In[32]:


time_slice = slice(None, None)


# In[33]:


import matplotlib.pyplot as plt

hfds_int.isel(time=time_slice).plot(label="CM4 HFDS Integrated")
(OHC.isel(time=time_slice) - OHC.isel(time=0)).plot(label="CM4 OHC")
# ((OHC_old.isel(time=time_slice)-OHC_old.isel(time=0))).plot(label='CM4 OHC (old)')
plt.ylabel("")
plt.legend()
plt.show()


# In[3]:


import sys

sys.path.append("/pscratch/sd/s/suryad/Ocean_Emulator")
sys.path.append("/pscratch/sd/s/suryad/Ocean_Emulator/src")

from constants import INPT_VARS, EXTRA_VARS, OUT_VARS
import torch
from utils.data import Normalize
from einops import rearrange


# In[4]:


inputs = INPT_VARS["3D_noFast_all"]
extra_in = EXTRA_VARS["3D_all"]
outputs = OUT_VARS["3D_noFast_all"]


# In[5]:


data_mean = xr.open_dataset(
    "/pscratch/sd/s/suryad/data/2024-11-12-cm4-piControl-ocean-200yr-dataset-stats/centering.nc",
    engine="netcdf4",
)
data_std = xr.open_dataset(
    "/pscratch/sd/s/suryad/data/2024-11-12-cm4-piControl-ocean-200yr-dataset-stats/scaling-full-field.nc",
    engine="netcdf4",
)


# In[6]:


normalize = Normalize.init_instance(
    data_mean,
    data_std,
    inputs,
    extra_in,
    outputs,
)

# normalize = Normalize.get_instance()


# In[7]:


import cftime

t1 = [
    cftime.DatetimeNoLeap(151, 1, 6, 0, 0, 0, 0),
    cftime.DatetimeNoLeap(151, 1, 11, 0, 0, 0, 0),
]
t2 = [
    cftime.DatetimeNoLeap(171, 1, 6, 0, 0, 0, 0),
    cftime.DatetimeNoLeap(171, 1, 11, 0, 0, 0, 0),
]
t3 = [
    cftime.DatetimeNoLeap(191, 1, 6, 0, 0, 0, 0),
    cftime.DatetimeNoLeap(191, 1, 11, 0, 0, 0, 0),
]
t4 = [
    cftime.DatetimeNoLeap(211, 1, 6, 0, 0, 0, 0),
    cftime.DatetimeNoLeap(211, 1, 11, 0, 0, 0, 0),
]
t5 = [
    cftime.DatetimeNoLeap(231, 1, 6, 0, 0, 0, 0),
    cftime.DatetimeNoLeap(231, 1, 11, 0, 0, 0, 0),
]
t6 = [
    cftime.DatetimeNoLeap(251, 1, 6, 0, 0, 0, 0),
    cftime.DatetimeNoLeap(251, 1, 11, 0, 0, 0, 0),
]
t7 = [
    cftime.DatetimeNoLeap(271, 1, 6, 0, 0, 0, 0),
    cftime.DatetimeNoLeap(271, 1, 11, 0, 0, 0, 0),
]
t8 = [
    cftime.DatetimeNoLeap(291, 1, 6, 0, 0, 0, 0),
    cftime.DatetimeNoLeap(291, 1, 11, 0, 0, 0, 0),
]


# In[11]:


files = [
    f"/pscratch/sd/s/suryad/Ocean_Emulator/train_3D/initial_prognostic_{i}.pt"
    for i in range(8)
]


# In[9]:


# Make this a test
for i, file in enumerate(files):
    saved = torch.load(file)
    print(f"\nFile {i}:")
    print("-" * 20)
    for j, t in enumerate([t1, t2, t3, t4, t5, t6, t7, t8]):
        data_in_norm = normalize.normalize_inputs(data[inputs].sel(time=t))
        data_in_norm_tensor = torch.tensor(data_in_norm.to_array().to_numpy())
        data_in_norm_tensor = rearrange(data_in_norm_tensor, "c t h w -> t c h w")
        data_in_norm_tensor = rearrange(data_in_norm_tensor, "t c h w -> (t c) h w")
        diff = (saved - data_in_norm_tensor).max()
        print(f"Time period {j}: {diff:.4f}")


# In[12]:


# Check if all files are the same tensor
all_tensors = [torch.load(file) for file in files]
all_tensors_equal = all(
    torch.allclose(all_tensors[0], tensor) for tensor in all_tensors[1:]
)
print(f"\nAll tensors are the same: {all_tensors_equal}")


# In[4]:


# fs_osn.ls("emulators/ai2_colab/2025-01-23-sample-CM4-piControl-ocean-preprocessed-monthly-data")
# mapper = fs_osn.get_mapper("emulators/ai2_colab/2025-01-23-sample-CM4-piControl-ocean-preprocessed-monthly-data/0151010100.nc")
# data_ai2 = xr.open_dataset(mapper, engine="netcdf4")


# In[5]:


# import numpy as np
# # Check if data_ai and data_m2lines are the same in some fields
# data_m2lines_convert = data_m2lines.sel(time=data_ai.time.values)
# # rename coords
# data_m2lines_convert = data_m2lines_convert.rename({"lon": "lon2", "lat": "lat2"})
# data_m2lines_convert = data_m2lines_convert.rename({"x": "lon", "y": "lat"})
# var_pairs = [["thetao_0", "thetao_lev_2_5"], ["uo_0", "uo_lev_2_5"], ["vo_0", "vo_lev_2_5"], ["so_0", "so_lev_2_5"],
#         ["zos", "zos"], ["tauuo", "tauuo"], ["tauvo", "tauvo"], ["hfds", "hfds"], ["thetao_1", "thetao_lev_10_0"], ["thetao_2", "thetao_lev_22_5"], ["thetao_3", "thetao_lev_40_0"], ["thetao_4", "thetao_lev_65_0"], ["thetao_5", "thetao_lev_105_0"], ["thetao_6", "thetao_lev_165_0"], ["thetao_7", "thetao_lev_250_0"], ["thetao_8", "thetao_lev_375_0"], ["thetao_9", "thetao_lev_550_0"], ["thetao_10", "thetao_lev_775_0"], ["thetao_11", "thetao_lev_1050_0"], ["thetao_12", "thetao_lev_1400_0"], ["thetao_13", "thetao_lev_1850_0"], ["thetao_14", "thetao_lev_2400_0"], ["thetao_15", "thetao_lev_3100_0"], ["thetao_16", "thetao_lev_4000_0"], ["thetao_17", "thetao_lev_5000_0"], ["thetao_18", "thetao_lev_6000_0"],
#         ["uo_1", "uo_lev_10_0"], ["uo_2", "uo_lev_22_5"], ["uo_3", "uo_lev_40_0"], ["uo_4", "uo_lev_65_0"], ["uo_5", "uo_lev_105_0"], ["uo_6", "uo_lev_165_0"], ["uo_7", "uo_lev_250_0"], ["uo_8", "uo_lev_375_0"], ["uo_9", "uo_lev_550_0"], ["uo_10", "uo_lev_775_0"], ["uo_11", "uo_lev_1050_0"], ["uo_12", "uo_lev_1400_0"], ["uo_13", "uo_lev_1850_0"], ["uo_14", "uo_lev_2400_0"], ["uo_15", "uo_lev_3100_0"], ["uo_16", "uo_lev_4000_0"], ["uo_17", "uo_lev_5000_0"], ["uo_18", "uo_lev_6000_0"],
#         ["vo_1", "vo_lev_10_0"], ["vo_2", "vo_lev_22_5"], ["vo_3", "vo_lev_40_0"], ["vo_4", "vo_lev_65_0"], ["vo_5", "vo_lev_105_0"], ["vo_6", "vo_lev_165_0"], ["vo_7", "vo_lev_250_0"], ["vo_8", "vo_lev_375_0"], ["vo_9", "vo_lev_550_0"], ["vo_10", "vo_lev_775_0"], ["vo_11", "vo_lev_1050_0"], ["vo_12", "vo_lev_1400_0"], ["vo_13", "vo_lev_1850_0"], ["vo_14", "vo_lev_2400_0"], ["vo_15", "vo_lev_3100_0"], ["vo_16", "vo_lev_4000_0"], ["vo_17", "vo_lev_5000_0"], ["vo_18", "vo_lev_6000_0"],
#         ["so_1", "so_lev_10_0"], ["so_2", "so_lev_22_5"], ["so_3", "so_lev_40_0"], ["so_4", "so_lev_65_0"], ["so_5", "so_lev_105_0"], ["so_6", "so_lev_165_0"], ["so_7", "so_lev_250_0"], ["so_8", "so_lev_375_0"], ["so_9", "so_lev_550_0"], ["so_10", "so_lev_775_0"], ["so_11", "so_lev_1050_0"], ["so_12", "so_lev_1400_0"], ["so_13", "so_lev_1850_0"], ["so_14", "so_lev_2400_0"], ["so_15", "so_lev_3100_0"], ["so_16", "so_lev_4000_0"], ["so_17", "so_lev_5000_0"], ["so_18", "so_lev_6000_0"]]
# for var_pair in var_pairs:
#     b = np.allclose(data_ai[var_pair[0]].values, data_m2lines_convert[var_pair[1]].values, equal_nan=True)
#     if not b:
#         print(var_pair)
#         # Print the difference
#         # Remove nans
#         data_ai_clean = data_ai[var_pair[0]].values[~np.isnan(data_ai[var_pair[0]].values)]
#         data_m2lines_clean = data_m2lines_convert[var_pair[1]].values[~np.isnan(data_m2lines_convert[var_pair[1]].values)]
#         print((data_ai_clean - data_m2lines_clean).max())


# In[3]:


import fsspec

# CM4
fs_osn = fsspec.filesystem(
    "s3",
    profile="ocean_emulator",  ## This is the profile name you configured above.
)


# In[4]:


import xarray as xr

mapper = fs_osn.get_mapper("emulators/jbusecke/ocean-emulators/CM4_5daily_v0.4.0.zarr")
data = xr.open_zarr(mapper, consolidated=True)


# In[3]:


fluxes = xr.open_zarr(
    fs_osn.get_mapper(
        "emulators/ai2_colab/2024-11-01-CM4-pre-industrial-control-simulation-select-atmosphere-and-coupler-data/gaussian_grid_180_by_360/fluxes_2d.zarr"
    ),
    consolidated=True,
)
fluxes_coarsened = fluxes.coarsen(time=20, coord_func=lambda x, axis: x[:, -1]).mean()
full_state = xr.open_zarr(
    fs_osn.get_mapper(
        "emulators/ai2_colab/2024-11-01-CM4-pre-industrial-control-simulation-select-atmosphere-and-coupler-data/gaussian_grid_180_by_360/full_state.zarr"
    ),
    consolidated=True,
)
full_state_coarsened = full_state.coarsen(
    time=20, coord_func=lambda x, axis: x[:, -1]
).mean()
full_state_coarsened


# In[4]:


fluxes_coarsened = fluxes_coarsened.rename({"lat": "y", "lon": "x"})
full_state_coarsened = full_state_coarsened.rename({"lat": "y", "lon": "x"})

assert (fluxes_coarsened.x == data.x).all(), "X coordinates don't match"
assert (fluxes_coarsened.y == data.y).all(), "Y coordinates don't match"
assert (full_state_coarsened.x == data.x).all(), "X coordinates don't match"
assert (full_state_coarsened.y == data.y).all(), "Y coordinates don't match"


# In[5]:


data["DSWRFtoa"] = fluxes_coarsened["DSWRFtoa"].chunk({"time": 10})
data["air_temperature_at_two_meters"] = full_state_coarsened[
    "air_temperature_at_two_meters"
].chunk({"time": 10})
data["surface_temperature"] = full_state_coarsened["surface_temperature"].chunk(
    {"time": 10}
)
data


# In[5]:


total_time = int(20 * 365 / 5)
ds = data.isel(time=slice(-total_time, None))
window_size = int(2 * 365 / 5)
num_windows = int(total_time / window_size)

ds_new = []
for w_i in range(num_windows):
    print(f"Processing window {w_i + 1} of {num_windows}")
    ds_window = ds.isel(time=slice(w_i * window_size, (w_i + 1) * window_size))
    time_slice = ds_window["time"]
    climatology = ds_window.groupby("time.dayofyear").mean("time").compute()
    day_of_year = ds_window["time"].dt.dayofyear
    res = (ds_window - climatology.sel(dayofyear=day_of_year)).compute()
    ds_new.append(res)

ds_new = xr.concat(ds_new, dim="time")


# In[23]:


import matplotlib.pyplot as plt

ds_new.thetao.isel(x=60, y=60, lev=0).plot(label="Windowed")
remove_climatology(
    ds.isel(x=60, y=60, lev=0).thetao.isel(time=slice(-total_time, None))
).plot(label="Full")
plt.legend()
plt.show()


# In[ ]:


day_of_year = ds["time"].dt.dayofyear
res = (ds - climatology.sel(dayofyear=day_of_year)).compute()
res


# In[13]:


import numpy as np

data = data.isel(time=slice(-3000, None))


def remove_climatology(ds):
    # Compute the climatology on the detrended data
    climatology = ds.groupby("time.dayofyear").mean("time").compute()

    # Remove the seasonal cycle (climatology) from the detrended data
    day_of_year = ds["time"].dt.dayofyear
    res = (ds - climatology.sel(dayofyear=day_of_year)).compute()

    return res


c_p = 3850  # J/(kg C)
rho_0 = 1025  # kg/m^3

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

OHC.plot(ax=ax, label="CM4", c="k")
coeffs_OHC_trend = np.polyfit(np.arange(OHC.size), OHC, 1)
(pos,) = ax.plot(
    OHC.time.data,
    np.arange(OHC.size) * coeffs_OHC_trend[0] + coeffs_OHC_trend[1],
    c="k",
    ls="--",
)

GT_ohc_slope = coeffs_OHC_trend[0]
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.89), ncols=3)
plt.show()


# In[ ]:


# In[6]:


# Convert Temperature from Kelvin to Celsius
data["surface_temperature"] = data["surface_temperature"] - 273.15
data["air_temperature_at_two_meters"] = data["air_temperature_at_two_meters"] - 273.15
data["surface_temperature"].attrs["units"] = r"$\degree C$"
data["air_temperature_at_two_meters"].attrs["units"] = r"$\degree C$"


# In[7]:


# data['DSWRFtoa'].isel(y=60,x=60, time=slice(0, 100)).plot()
data["surface_temperature"].isel(y=60, x=60, time=slice(0, 100)).plot()
data["air_temperature_at_two_meters"].isel(y=60, x=60, time=slice(0, 100)).plot()


# In[8]:


data


# ### Time slice / rearrangement

# In[7]:


# print(data.time.size)
# data = data.sel(time=slice("1975-01-01", None))
# data.time.size

# Rearrange data - None-1982, 1991-None, 1982-1991
# data = xr.concat(
#     [
#         data.sel(time=slice(None, "1982-01-01")),
#         data.sel(time=slice("1990-04-24", None)),
#         data.sel(time=slice("1982-01-01", "1990-04-24")),
#     ],
#     dim="time",
# )


# ### Saving HFDS Anoms

# In[9]:


climatology = data["hfds"].groupby("time.dayofyear").mean("time").compute()
# Remove the seasonal cycle (climatology) from the detrended data
day_of_year = data["hfds"]["time"].dt.dayofyear
data["hfds_anomalies"] = (
    data["hfds"] - climatology.sel(dayofyear=day_of_year)
).compute()


# ### Saving Integrated HC

# In[9]:


# from scipy.integrate import cumulative_trapezoid
# import dask
# a = cumulative_trapezoid(data["hfds"].values, axis=0, initial=0)
# data['cum_integrated_heat'] = (['time', 'y', 'x'], dask.array.from_array(a, chunks=(1,180,360)))


# In[10]:


# (data['cum_integrated_heat']*data['areacello']).sum(['x', 'y']).plot()


# In[11]:


# poly_coeffs = data['cum_integrated_heat'].polyfit(dim='time', deg=1)
# trend = xr.polyval(data['time'], poly_coeffs.polyfit_coefficients).compute()

# # Remove the trend from the original data
# data['cum_integrated_heat'] = data['cum_integrated_heat'] - trend


# In[12]:


# # Compute the climatology on the detrended data
# climatology = data['cum_integrated_heat'].groupby('time.dayofyear').mean('time').compute()

# # Remove the seasonal cycle (climatology) from the detrended data
# day_of_year = data['cum_integrated_heat']['time'].dt.dayofyear
# data['cum_integrated_heat'] = (data['cum_integrated_heat'] - climatology.sel(dayofyear=day_of_year)).compute()


# In[13]:


# data['cum_integrated_heat'] = data['cum_integrated_heat'] + trend


# In[14]:


# cum_hfds_timeseries = (data['cum_integrated_heat']*data['areacello']).sum(['x', 'y'])
# cum_hfds_timeseries.plot()


# In[15]:


# data['cum_integrated_heat'].isel(x=60,y=60).compute()


# In[16]:


# a.lat.item()


# In[17]:


# import matplotlib.pyplot as plt
# a = data['hfds'].isel(x=60,y=60).compute()
# a.plot()
# plt.title(f"lat: {a.lat.item()}, lon:{a.lon.item()}")


# In[18]:


# import matplotlib.pyplot as plt
# a = data['cum_integrated_heat'].isel(x=60,y=60).compute()
# a.plot()
# plt.title(f"lat: {a.lat.item()}, lon:{a.lon.item()}")


# In[19]:


# import numpy as np
# import xarray as xr
# from scipy.spatial import cKDTree

# # Assuming 'data' is your dataset

# # Flatten the lat and lon arrays
# lats = data['lat'].values
# lons = data['lon'].values
# flat_lats = lats.ravel()
# flat_lons = lons.ravel()

# # Build KDTree
# coords = np.column_stack((flat_lats, flat_lons))
# tree = cKDTree(coords)

# # Specify your desired latitude and longitude
# desired_lat = 45.0   # Replace with your desired latitude
# desired_lon = -120.0 # Replace with your desired longitude

# # Query the KDTree for the nearest point
# distance, index = tree.query([desired_lat, desired_lon])

# # Convert flat index back to 2D indices
# iy, ix = np.unravel_index(index, lats.shape)

# # Extract the data point
# data_point = data['cum_integrated_heat'].isel(y=iy, x=ix)

# # Plot the data
# data_point.plot()
# plt.title(f"lat={desired_lat}, lon={ix}")


# ### Convert and save 3D data for training

# Make sure you are using atleast 10 cores!

# In[10]:


from dask.diagnostics import ProgressBar


# In[11]:


ds = data
ds


# In[12]:


assert [str(lev).replace(".", "_") for lev in ds["lev"].values] == [
    "2_5",
    "10_0",
    "22_5",
    "40_0",
    "65_0",
    "105_0",
    "165_0",
    "250_0",
    "375_0",
    "550_0",
    "775_0",
    "1050_0",
    "1400_0",
    "1850_0",
    "2400_0",
    "3100_0",
    "4000_0",
    "5000_0",
    "6000_0",
]


# In[13]:


for lev in ds["lev"].values:
    lev_str = str(lev).replace(".", "_")

    # Create new variables for each original variable with the lev dimension
    ds[f"vo_lev_{lev_str}"] = ds["vo"].sel(lev=lev)
    ds[f"thetao_lev_{lev_str}"] = ds["thetao"].sel(lev=lev)
    ds[f"uo_lev_{lev_str}"] = ds["uo"].sel(lev=lev)
    ds[f"so_lev_{lev_str}"] = ds["so"].sel(lev=lev)

ds = ds.drop_vars(["vo", "thetao", "uo", "so"])
ds


# In[ ]:


# with ProgressBar():
#     scaling_residual = ds.diff("time").std().compute()


# In[ ]:


# scaling_residual.to_zarr("/pscratch/sd/s/suryad/data/3D_data_OM4_5daily_v0.2.1_with_hfds_cuminteg_hfds_1975_scaling_residuals", mode="w")


# In[29]:


with ProgressBar():
    ds_mean = ds.mean().compute()


# In[30]:


ds_mean.to_zarr("/pscratch/sd/s/suryad/data/CM4_5daily_v0.4.0_means", mode="w")


# In[31]:


with ProgressBar():
    ds_std = ds.std().compute()


# In[32]:


ds_std.to_zarr("/pscratch/sd/s/suryad/data/CM4_5daily_v0.4.0_stds", mode="w")


# In[14]:


with ProgressBar():
    ds.to_zarr(
        "/pscratch/sd/s/suryad/data/CM4_5daily_v0.4.0",
        mode="w",
        consolidated=True,
        encoding={v: {"compressor": None} for v in ds.variables},
    )


# In[ ]:


# Save wet mask
wet = data.wetmask
wet.to_zarr("/pscratch/sd/s/suryad/data/CM4_5daily_v0.4.0_wetmask", mode="w")


# In[ ]:


import datetime

import pytz

utc = pytz.utc
utc_dt = datetime.datetime.now(utc)
eastern = pytz.timezone("US/Eastern")
loc_dt = utc_dt.astimezone(eastern)
fmt = "%Y-%m-%d %H:%M:%S %Z%z"
loc_dt.strftime(fmt)


# #### Wet mask

# In[1]:


import xarray as xr


# In[2]:


data = xr.open_zarr("/pscratch/sd/s/suryad/data/OM4_5daily_v0.2.1.zarr")

# import numpy as np
# def manual_v0_fixes(ds_input: xr.Dataset) -> xr.Dataset:
#     """Manual fixes for the already existing data (for now only v0.0). This should not be used in the future"""
#     # fixes that should be checked and fixes on the input data
#     # area = xr.open_dataset(
#     #     "gs://leap-persistent/sd5313/grids_CM2x.zarr", engine="zarr", chunks={}
#     # )["area_C"].rename({"xu_ocean": "x", "yu_ocean": "y"})
#     # print("area", area)
#     # area = xr.open_dataset("/pscratch/sd/s/suryad/data/Grid_New.nc")["area_C"].rename({"xu_ocean": "x", "yu_ocean": "y"})
#     # print("area", area)
#     # from https://github.com/m2lines/ocean_emulators/issues/17
#     dz = xr.DataArray(
#         [
#             5,
#             10,
#             15,
#             20,
#             30,
#             50,
#             70,
#             100,
#             150,
#             200,
#             250,
#             300,
#             400,
#             500,
#             600,
#             800,
#             1000,
#             1000,
#             1000,
#         ],
#         dims=["lev"],
#     )
#     z = xr.DataArray(
#         [
#             2.5,
#             10,
#             22.5,
#             40,
#             65,
#             105,
#             165,
#             250,
#             375,
#             550,
#             775,
#             1050,
#             1400,
#             1850,
#             2400,
#             3100,
#             4000,
#             5000,
#             6000,
#         ],
#         dims="lev",
#     )
#     wetmask = ~np.isnan(ds_input.thetao.isel(time=0).reset_coords(drop=True)).load()
#     # lon = xr.ones_like(ds_input.y) * ds_input.x
#     # lat = ds_input.y * xr.ones_like(ds_input.x)
#     # ds_input = ds_input.assign_coords(
#     #     areacello=area, dz=dz, lev=z, wetmask=wetmask, lon=lon, lat=lat
#     # )
#     ds_input = ds_input.assign_coords(
#         dz=dz, lev=z, wetmask=wetmask
#     )
#     # give a dummy commit hash
#     ds_input.attrs["m2lines/ocean-emulators_git_hash"] = "dummy"
#     return ds_input

# data = manual_v0_fixes(data)

levels = 5
# data = xr.open_zarr(
#     "/pscratch/sd/s/suryad/data/OM4_5daily_v0.2.1.zarr"
# )
# data = xr.open_zarr(
#     "/vast/sd5313/data/m2lines/3D_ocean_data/test_CMIP6_GFDL-CM4.piControl.r1i1p1f1.zarr"
# )
data


# In[3]:


data = data.drop(["tauuo", "tauvo", "hfds"])
# data = data.drop(["tauuo", "tauvo", "hft"])
data


# In[4]:


import numpy as np
import torch


# Outdated!!!
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


# In[5]:


data.lev.values


# In[7]:


wet_stacked = []
for i, lev in enumerate(data["lev"].values[:levels]):
    inputs = []
    inputs.append(data["uo"].sel(lev=lev))
    inputs.append(data["vo"].sel(lev=lev))
    inputs.append(data["thetao"].sel(lev=lev))
    inputs.append(data["so"].sel(lev=lev))
    if i == 0:
        inputs.append(data["zos"])

    inputs = tuple(inputs)
    wet, _ = get_wet_mask(inputs)
    wet_stacked.append(wet)


# In[8]:


wet_3D = torch.stack(wet_stacked)
wet_3D.shape


# In[7]:


import numpy as np
import torch

# Experiment inputs and outputs
DEPTH_LEVELS = [
    "2_5",
    "10_0",
    "22_5",
    "40_0",
    "65_0",
    "105_0",
    "165_0",
    "250_0",
    "375_0",
    "550_0",
    "775_0",
    "1050_0",
    "1400_0",
    "1850_0",
    "2400_0",
    "3100_0",
    "4000_0",
    "5000_0",
    "6000_0",
]

INPT_VARS = {
    "1": ["um", "vm"],
    "2": ["um", "vm", "ur", "vr"],
    "3": ["um", "vm", "Tm"],
    "4": ["um", "vm", "ur", "vr", "Tm", "Tr"],
    "5": ["ur", "vr"],
    "6": ["ur", "vr", "Tr"],
    "7": ["Tm"],
    "8": ["Tm", "Tr"],
    "9": ["u", "v"],
    "10": ["u", "v", "T"],
    "11": ["tau_u", "tau_v"],
    "12": ["tau_u", "tau_v", "t_ref"],
    "3D": ["uo", "vo", "thetao", "so", "zos"],
    "2D": [
        k + DEPTH_LEVELS[0] for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
    ]
    + ["zos"],
    "3D_5": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS[:5]
    ]
    + ["zos"],
    "3D_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_SST_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
        if not (k == "thetao_lev_" and j == DEPTH_LEVELS[0])
    ]
    + ["zos"],
}
EXTRA_VARS = {
    "1": ["ur", "vr"],
    "2": ["ur", "vr", "Tm"],
    "3": ["Tm"],
    "4": ["ur", "vr", "Tm", "Tr"],
    "5": [],
    "6": ["um", "vm"],
    "7": ["um", "vm", "Tm"],
    "8": ["um", "vm", "Tm", "Tr"],
    "9": ["ur", "vr", "tau_u", "tau_v"],
    "10": ["tau_u", "tau_v"],
    "11": ["t_ref"],
    "12": ["tau_u", "tau_v", "t_ref"],
    "13": ["ur", "vr", "Tr", "tau_u", "tau_v", "t_ref"],
    "3D": ["tauuo", "tauvo", "hfds"],
    "2D": ["tauuo", "tauvo", "hfds"],
    "3D_5": ["tauuo", "tauvo", "hfds"],
    "3D_all": ["tauuo", "tauvo", "hfds"],
    "3D_SST_all": ["tauuo", "tauvo", "hfds", "thetao_lev_0"],
}
OUT_VARS = {
    "1": ["um", "vm"],
    "2": ["um", "vm", "Tm"],
    "3": ["ur", "vr"],
    "4": ["ur", "vr", "Tr"],
    "5": ["u", "v"],
    "6": ["u", "v", "T"],
    "3D": ["uo", "vo", "thetao", "so", "zos"],
    "2D": [
        k + DEPTH_LEVELS[0] for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
    ]
    + ["zos"],
    "3D_5": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS[:5]
    ]
    + ["zos"],
    "3D_noFast_5": [
        k + str(j) for k in ["thetao_lev_", "so_lev_"] for j in DEPTH_LEVELS[:5]
    ]
    + ["zos"],
    "3D_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_SST_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
        if not (k == "thetao_lev_" and j == DEPTH_LEVELS[0])
    ]
    + ["zos"],
}


# In[11]:


out_list = INPT_VARS["3D_5"]
lev_map = {str(lev).replace(".", "_"): i for i, lev in enumerate(data["lev"].values)}
# print(out_list[:38] + out_list[39:])


# In[12]:


out_list[0].split("lev_")[-1]


# In[13]:


final_wet = []
for var in out_list:
    try:
        level = lev_map[var.split("lev_")[-1]]
        print(var, level)
    except Exception:
        level = 0
    final_wet.append(wet_3D[level])


# In[14]:


wet = torch.stack(final_wet)
wet.shape


# In[15]:


assert wet.shape[0] == (levels * 4 + 1)


# In[79]:


# wet = torch.cat([wet[:38], wet[39:]], axis=0)
# wet.shape


# In[16]:


torch.save(wet, "/pscratch/sd/s/suryad/data/3D_wet_OM4_v0.2.1_5levels.pt")


# ### Convert and save surface data

# Make sure you are using atleast 8 cores!

# In[81]:


import sys

sys.path.append("../src/")


# In[82]:


from datasets import get_wet_mask


# In[83]:


inputs = []
inputs.append(data["uo"].sel(lev=2.5))
inputs.append(data["vo"].sel(lev=2.5))
inputs.append(data["thetao"].sel(lev=2.5))
inputs.append(data["so"].sel(lev=2.5))
inputs.append(data["zos"])
inputs = tuple(inputs)
wet, _ = get_wet_mask(inputs)
print("Wet resolution:", wet.shape)


# In[84]:


# print("Calculating mask tensors")
# wet, wet_nan = get_wet_mask(inputs, "cpu")
# # wet_bool = np.array(wet.cpu()).astype(bool)
# # wet_lap = compute_laplacian_wet(wet_nan, 4)  # hardcoded
# # wet_lap = xr.where(wet_lap == 0, 1, np.nan)
# # wet_lap = np.nan_to_num(wet_lap)
# print("Wet resolution:", wet.shape)


# In[85]:


import torch


# In[86]:


torch.save(wet, "/pscratch/sd/s/suryad/data/surface_wet_OM4_v0.0.pt")


# #### Extra for surface training

# In[ ]:


from dask.diagnostics import ProgressBar


# In[ ]:


with ProgressBar():
    data_mean = data.mean().compute()


# In[ ]:


data_mean.to_zarr("/vast/sd5313/data/m2lines/3D_ocean_data/surface_data_means")


# In[ ]:


with ProgressBar():
    data_std = data.std().compute()


# In[ ]:


data_std.to_zarr("/vast/sd5313/data/m2lines/3D_ocean_data/surface_data_stds")


# In[ ]:


data.to_zarr("/vast/sd5313/data/m2lines/3D_ocean_data/surface_data")


# ### Test

# In[9]:


import matplotlib.pyplot as plt
import xarray as xr
from dask.diagnostics import ProgressBar


# In[10]:


def profile_mean(ds: xr.Dataset) -> xr.Dataset:
    return ds.weighted(ds.areacello).mean(["x", "y"])


with ProgressBar():
    profile_groundtruth = profile_mean(data).load()


# In[80]:


import numpy as np

# Assuming 'hfds' is your xarray DataArray with time as the coordinate
time = range(len(profile_groundtruth.time.values))
hfds = profile_groundtruth.tauuo.values

# Perform a linear fit (1st degree polynomial)
coefficients = np.polyfit(time, hfds, 1)
trend = np.polyval(coefficients, time)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.plot(time, hfds, label="tauuo", color="blue")

# Plot the linear trend
plt.plot(time, trend, label="Linear Trend", color="red", linestyle="--")

plt.xlabel("Time")
plt.ylabel("tauuo")
plt.title("tauuo with Linear Trend")
plt.legend()
plt.show()


# In[73]:


profile_groundtruth.hfds.plot()


# In[53]:


profile_groundtruth.hfds.plot()


# In[81]:


fig, ax = plt.subplots()
profile_groundtruth.isel(lev=7).isel(
    time=slice(None, 4000)
).thetao.plot()  # .isel(time=slice(-600,None))
# loc = plticker.MultipleLocator(base=4000.0) # this locator puts ticks at regular intervals
# ax.xaxis.set_major_locator(loc)
plt.show()


# In[4]:


data.y


# In[9]:


data_loc = data.sel(x=slice(190, 200), y=slice(75, 85))
location = "Equatorial Pacific"
data_loc_single = data_loc.isel(x=5, y=5).isel(lev=0).isel(time=slice(None, 144))


# In[10]:


data_loc_single.thetao.plot()
plt.title(
    f"{location} (No filter) - x = {data_loc_single.x.item()}, y = {data_loc_single.y.item()}"
)
plt.show()


# ##### Test fast velocities smoothing

# In[12]:


import xarray as xr

data = xr.open_zarr("/pscratch/sd/s/suryad/data/OM4_5daily_v0.2.1.zarr")


# In[34]:


from dask.diagnostics import ProgressBar


# In[35]:


# %matplotlib inline
# import matplotlib.pyplot as plt


# In[36]:


window = 30
with ProgressBar():
    data["uo"] = (
        data.uo.rolling(time=window, min_periods=1, center=False).mean().compute()
    )
    data["vo"] = (
        data.vo.rolling(time=window, min_periods=1, center=False).mean().compute()
    )


# In[11]:


with ProgressBar():
    data.to_zarr(
        "/pscratch/sd/s/suryad/data/OM4_5daily_v0.2.1_fast_smoothed_30.zarr",
        encoding={v: {"compressor": None} for v in data.variables},
        consolidated=True,
        mode="w",
    )


# In[ ]:


# In[35]:


data.uo


# In[93]:


# data.sel(x=slice(195, 196), y=slice(50,51)).isel(lev=17).uo.plot()


# In[ ]:


# In[94]:


data_loc = data.sel(x=slice(190, 200), y=slice(-5, 5))
location = "Equatorial Pacific"
# data_loc = data.sel(x=slice(190, 200), y=slice(45,55))
# location = "Midlatitudes"
data_loc


# In[95]:


data_loc_single = data_loc.isel(x=5, y=5).isel(lev=0)
data_loc_single.uo.plot()
plt.title(
    f"{location} (No filter) - x = {data_loc_single.x.item()}, y = {data_loc_single.y.item()}"
)
plt.show()


# In[96]:


import numpy as np

# Create a DataArray with some data
data = xr.DataArray(range(10), dims="time")
print(data)
# Apply rolling mean with a window of 3 and min_periods=1 to avoid NaNs
rolled_data = data.rolling(time=3, min_periods=1, center=False).mean()
rolled_data


# In[97]:


window = 30
data_loc_single["uo"] = (
    data_loc_single.uo.rolling(time=window, min_periods=1, center=False)
    .mean()
    .compute()
)


# In[98]:


data_loc_single.uo


# In[99]:


data_loc_single.uo.plot()
plt.title(
    f"{location} (With filter {window}) - x = {data_loc_single.x.item()}, y = {data_loc_single.y.item()}"
)
plt.show()


# In[100]:


data_loc_single.thetao.plot()
plt.title(
    f"{location} (No filter) - x = {data_loc_single.x.item()}, y = {data_loc_single.y.item()}"
)
plt.show()


# ##### Test Wet

# In[35]:


wet = xr.open_zarr("/pscratch/sd/s/suryad/data/OM4_5daily_v0.2.1_wetmask")
wet


# In[36]:


wet.lev.values[0]


# In[37]:


depths = [var.split("lev_")[-1].replace("_", ".") for var in OUT_VARS["3D_all"]]
if "zos" in depths:
    zos_index = depths.index("zos")
    depths[zos_index] = "2.5"
depths = [float(depth) for depth in depths]
depths
new_wet = wet.sel(lev=depths)
wet = torch.from_numpy(wet.to_array().to_numpy())
wet = torch.concat([wet] * (1 + 1), dim=1)
wet.shape


# In[30]:


import torch

wettorch = torch.load("/pscratch/sd/s/suryad/data/3D_wet_OM4_5daily_v0.2.1.pt")
(torch.from_numpy(new_wet.to_array().to_numpy()) == wettorch).all()


# In[3]:


data = xr.open_zarr("/pscratch/sd/s/suryad/data/OM4_Horizontal_Regrid_Old.zarr")
data


# In[37]:


data = xr.open_zarr("/pscratch/sd/s/suryad/data/realv00/3D_data_OM4_v0.0")
data


# In[12]:


import numpy as np
import torch

# Experiment inputs and outputs
DEPTH_LEVELS = [
    "2_5",
    "10_0",
    "22_5",
    "40_0",
    "65_0",
    "105_0",
    "165_0",
    "250_0",
    "375_0",
    "550_0",
    "775_0",
    "1050_0",
    "1400_0",
    "1850_0",
    "2400_0",
    "3100_0",
    "4000_0",
    "5000_0",
    "6000_0",
]

INPT_VARS = {
    "1": ["um", "vm"],
    "2": ["um", "vm", "ur", "vr"],
    "3": ["um", "vm", "Tm"],
    "4": ["um", "vm", "ur", "vr", "Tm", "Tr"],
    "5": ["ur", "vr"],
    "6": ["ur", "vr", "Tr"],
    "7": ["Tm"],
    "8": ["Tm", "Tr"],
    "9": ["u", "v"],
    "10": ["u", "v", "T"],
    "11": ["tau_u", "tau_v"],
    "12": ["tau_u", "tau_v", "t_ref"],
    "3D": ["uo", "vo", "thetao", "so", "zos"],
    "3D_5": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS[:5]
    ]
    + ["zos"],
    "3D_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_SST_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
        if not (k == "thetao_lev_" and j == DEPTH_LEVELS[0])
    ]
    + ["zos"],
}
EXTRA_VARS = {
    "1": ["ur", "vr"],
    "2": ["ur", "vr", "Tm"],
    "3": ["Tm"],
    "4": ["ur", "vr", "Tm", "Tr"],
    "5": [],
    "6": ["um", "vm"],
    "7": ["um", "vm", "Tm"],
    "8": ["um", "vm", "Tm", "Tr"],
    "9": ["ur", "vr", "tau_u", "tau_v"],
    "10": ["tau_u", "tau_v"],
    "11": ["t_ref"],
    "12": ["tau_u", "tau_v", "t_ref"],
    "13": ["ur", "vr", "Tr", "tau_u", "tau_v", "t_ref"],
    "3D": ["tauuo", "tauvo", "hfds"],
    "3D_5": ["tauuo", "tauvo", "hfds"],
    "3D_all": ["tauuo", "tauvo", "hfds"],
    "3D_SST_all": ["tauuo", "tauvo", "hfds", "thetao_lev_0"],
}
OUT_VARS = {
    "1": ["um", "vm"],
    "2": ["um", "vm", "Tm"],
    "3": ["ur", "vr"],
    "4": ["ur", "vr", "Tr"],
    "5": ["u", "v"],
    "6": ["u", "v", "T"],
    "3D": ["uo", "vo", "thetao", "so", "zos"],
    "3D_5": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS[:5]
    ]
    + ["zos"],
    "3D_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
    ]
    + ["zos"],
    "3D_SST_all": [
        k + str(j)
        for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
        for j in DEPTH_LEVELS
        if not (k == "thetao_lev_" and j == DEPTH_LEVELS[0])
    ]
    + ["zos"],
}


# In[19]:


data


# In[53]:


data = xr.open_zarr("/pscratch/sd/s/suryad/data/3D_data_OM4_v0.0")
# mean and std
data_mean = xr.open_zarr("/pscratch/sd/s/suryad/data/3D_data_OM4_v0.0_means")
data_std = xr.open_zarr("/pscratch/sd/s/suryad/data/3D_data_OM4_v0.0_stds")

# Extract single timestep
data_single = data[INPT_VARS["3D_all"]].isel(time=0)

# Normalize
data_norm = (data_single - data_mean[INPT_VARS["3D_all"]]) / data_std[
    INPT_VARS["3D_all"]
]

# Unnormalize
data_unnorm = data_norm * data_std[INPT_VARS["3D_all"]] + data_mean[INPT_VARS["3D_all"]]

# Verify that the unnormalized data is the same as the original data
assert (data_single - data_unnorm).sum() == 0


# In[6]:


data2 = xr.open_zarr("/pscratch/sd/s/suryad/data/3D_data_OM4_v0.0_means")
data2


# In[52]:


import torch

d1 = torch.load("/pscratch/sd/s/suryad/data/3D_wet_OM4_v0.0.pt")
d2 = torch.load("/pscratch/sd/s/suryad/data/realv00/3D_wet_OM4_v0.0.pt")
assert (d1 - d2).sum() == 0

d1 = torch.load("/pscratch/sd/s/suryad/data/surface_wet_OM4_v0.0.pt")
d2 = torch.load("/pscratch/sd/s/suryad/data/realv00/surface_wet_OM4_v0.0.pt")
assert (d1 - d2).sum() == 0


# In[16]:


# mean and std
data_mean = xr.open_zarr("/pscratch/sd/s/suryad/data/3D_data_OM4_v0.0_means")
data_std = xr.open_zarr("/pscratch/sd/s/suryad/data/3D_data_OM4_v0.0_stds")

data_mean2 = xr.open_zarr("/pscratch/sd/s/suryad/data/realv00/3D_data_OM4_v0.0_means")
data_std2 = xr.open_zarr("/pscratch/sd/s/suryad/data/realv00/3D_data_OM4_v0.0_stds")


# In[17]:


data_mean.values == data_mean2.values


# In[25]:


data_mean[INPT_VARS["3D_all"]].load()


# In[1]:


ls = [
    k + str(j)
    for k in ["uo_lev_", "vo_lev_", "thetao_lev_", "so_lev_"]
    for j in range(19)
] + ["zos"]
# data_mean2[ls].load()


# In[36]:


assert (
    data_mean[INPT_VARS["3D_all"]].load().to_array().values
    == data_mean2[ls].load().to_array().values
).all()
assert (
    data_std[INPT_VARS["3D_all"]].load().to_array().values
    == data_std2[ls].load().to_array().values
).all()


# In[33]:


import numpy as np

area_weights = np.sqrt(np.cos(np.deg2rad(data.y)))
area_weights


# In[38]:


plt.plot(np.sqrt(np.cos(np.deg2rad(data.y))), label="SQRT")
plt.plot(np.cos(np.deg2rad(data.y)), label="COS")
plt.legend()


# In[51]:


diff_weights = np.array(
    [
        3.6734e-01,
        3.2146e-01,
        2.8054e-01,
        2.3014e-01,
        1.9086e-01,
        1.5179e-01,
        1.3363e-01,
        1.3338e-01,
        1.3123e-01,
        1.3425e-01,
        1.4122e-01,
        1.5607e-01,
        1.8306e-01,
        2.1640e-01,
        2.6909e-01,
        3.4798e-01,
        3.9350e-01,
        3.0327e-01,
        1.5722e-01,
        6.0882e-01,
        5.2569e-01,
        4.6328e-01,
        3.9356e-01,
        3.3616e-01,
        3.0681e-01,
        2.7544e-01,
        2.4836e-01,
        2.4550e-01,
        2.5093e-01,
        2.7280e-01,
        3.1013e-01,
        3.3352e-01,
        3.5167e-01,
        4.0584e-01,
        4.8497e-01,
        4.8098e-01,
        3.2968e-01,
        1.6328e-01,
        1.7627e-02,
        1.6149e-02,
        1.3806e-02,
        1.1896e-02,
        9.5263e-03,
        9.8734e-03,
        8.9447e-03,
        5.9380e-03,
        5.4137e-03,
        5.3000e-03,
        5.1151e-03,
        4.5471e-03,
        3.2893e-03,
        2.3946e-03,
        2.0203e-03,
        1.9423e-03,
        2.0582e-03,
        1.1031e-03,
        3.1544e-04,
        2.2664e-02,
        1.7673e-02,
        1.8266e-02,
        1.2867e-02,
        9.1990e-03,
        6.7433e-03,
        6.5713e-03,
        5.3030e-03,
        4.1866e-03,
        2.8688e-03,
        1.8589e-03,
        1.5699e-03,
        1.3976e-03,
        7.7108e-04,
        4.0685e-04,
        5.9043e-04,
        8.2559e-04,
        7.7413e-04,
        2.1697e-04,
        3.3815e-02,
    ]
)


# In[27]:


temp_weights = np.array(
    [
        3.6734e-01,
        3.2146e-01,
        2.8054e-01,
        2.3014e-01,
        1.9086e-01,
        1.5179e-01,
        1.3363e-01,
        1.3338e-01,
        1.3123e-01,
        1.3425e-01,
        1.4122e-01,
        1.5607e-01,
        1.8306e-01,
        2.1640e-01,
        2.6909e-01,
        3.4798e-01,
        3.9350e-01,
        3.0327e-01,
        1.5722e-01,
    ]
)


# In[6]:


temp_weights = np.array(
    [
        1.7627e-02,
        1.6149e-02,
        1.3806e-02,
        1.1896e-02,
        9.5263e-03,
        9.8734e-03,
        8.9447e-03,
        5.9380e-03,
        5.4137e-03,
        5.3000e-03,
        5.1151e-03,
        4.5471e-03,
        3.2893e-03,
        2.3946e-03,
        2.0203e-03,
        1.9423e-03,
        2.0582e-03,
        1.1031e-03,
        3.1544e-04,
    ]
)


# In[28]:


temp_weights = 1 / temp_weights


# In[30]:


# Copy the weights after 12 till the end (cutoff)
# in numpy
temp_weights[12:] = temp_weights[12]


# In[31]:


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
plt.plot(DEPTH_LEVELS, temp_weights)


# In[ ]:
