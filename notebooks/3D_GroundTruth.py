#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# coding: utf-8

# # Evaluation Plots
#
# Vision:
#
# ```
# from ocean_emulators.plotting import eval_plots
#
# training_url = ...
# prediction_url = ...
#
# eval_plots(training_url, prediction_url)
# ```
#
# Where
#
# ```
# def eval_plots(...):
#     # Run tests on prediction data
#     # Plot all relevant panels in a neat and organized way
# ```
#

# ### Imports

# In[2]:


import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# In[3]:


get_ipython().run_line_magic("matplotlib", "inline")


# In[4]:


import sys

sys.path.append("../ocean_emulators_main/")

from ocean_emulators.dataset_validation import ds_input_validate


# In[5]:


# Rollout details
levels = 19
emulation_stability = False
emulation_stability_cc = False
reorder_82_90 = False


# In[6]:


import pandas as pd

# OM4 v0.2.1
ds_input = xr.open_zarr(
    os.path.join("/pscratch/sd/s/suryad/data", "OM4_5daily_v0.2.1.zarr")
)

ds_input_validate(ds_input)

# our groundtruth is always just a time slice of the training (training is a bad name

if emulation_stability:
    repeats = 100
    ds_groundtruth = ds_input.isel(lev=slice(None, levels))
    ds_groundtruth = ds_groundtruth.sel(time=slice("1996-01-01", "1996-12-31"))
    new_time = pd.date_range(
        start=str(ds_groundtruth.time[0].values),
        periods=repeats * len(ds_groundtruth.time),
        freq="5D",
    )
    ds_groundtruth = xr.concat([ds_groundtruth] * repeats, dim="time")
    ds_groundtruth["time"] = new_time
    ds_groundtruth = ds_groundtruth.isel(time=slice(3, 7003))
elif emulation_stability_cc:
    years = 10
    dates = np.array(range(3, 365 * years, 5))
    repeats = 100 // years

    ds_groundtruth = ds_input.isel(lev=slice(None, levels))
    ds_groundtruth = ds_groundtruth.sel(time=slice("1990-01-01", "1999-12-31"))

    # CC 1
    new_time = np.array(
        [np.datetime64("1990") + np.timedelta64(day - 1, "D") for day in dates]
    )
    for i in range(1, repeats):
        new_time = np.hstack(
            (
                new_time,
                np.array(
                    [
                        np.datetime64(str(1990 + i)) + np.timedelta64(day - 1, "D")
                        for day in dates
                    ]
                ),
            )
        )
    # new_time = pd.date_range(start=str(ds_groundtruth.time[0].values), periods=repeats * len(ds_groundtruth.time), freq="5D")
    ds_groundtruth["hfds"] = ds_groundtruth["hfds"] - 0.12086974
    ds_groundtruth = xr.concat([ds_groundtruth] * repeats, dim="time")
    ds_groundtruth["hfds"] = ds_groundtruth["hfds"] + np.reshape(
        np.arange(ds_groundtruth.time.size) * 4.01369026e-04, (-1, 1, 1)
    )
    ds_groundtruth["time"] = new_time[:7300]
    ds_groundtruth = ds_groundtruth.isel(time=slice(3, 7003))
elif reorder_82_90:
    ds_input = xr.concat(
        [
            ds_input.sel(time=slice(None, "1982-01-01")),
            ds_input.sel(time=slice("1990-04-24", None)),
            ds_input.sel(time=slice("1982-01-01", "1990-04-24")),
        ],
        dim="time",
    )
    ds_groundtruth = ds_input.isel(time=slice(4143, 4743)).isel(lev=slice(None, levels))

else:
    # ds_groundtruth = ds_input.isel(time=slice(4143, 4743)).isel(lev=slice(None, levels))
    #
    # ds_groundtruth = ds_input.isel(time=slice(3, 603)).isel(lev=slice(None, levels))
    #
    # ds_input = ds_input.sel(time=slice("1975-01-01", None))
    # ds_groundtruth = ds_input.isel(time=slice(2903, 3503)).isel(lev=slice(None, levels))

    ds_groundtruth = ds_input

    #
    # ds_groundtruth = ds_input.isel(time=slice(4144, 4744)).isel(lev=slice(None, levels))


ds_groundtruth = ds_groundtruth.astype("float32")


# In[7]:


ds_groundtruth


# In[ ]:


# In[8]:


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


# In[9]:


ds_groundtruth


# In[10]:


# Variables
boundary = xr.open_zarr("/pscratch/sd/a/asubel/Plots_For_Neurips/hfds_anomalies.zarr")
ds_groundtruth["hfds_anomalies"] = boundary["hfds"]
variables = ["hfds", "hfds_anomalies", "tauuo", "tauvo"]
ds_boundary = ds_groundtruth[variables]
# ds_boundary


# In[87]:


# temp_hfds = ds_boundary['hfds'].sel(time=slice('2015-01-01', '2022-12-31')) + masks['Atlantic'].fillna(0) * 30
# ds_boundary['hfds'].loc[dict(time=slice('2015-01-01', '2022-12-31'))] = temp_hfds


# In[88]:


for variable in variables:
    ds_boundary[variable], _, _ = detrend_and_remove_climatology(
        ds_boundary, var=variable
    )


# In[35]:


# Define your time periods
time_periods = [
    ("1975-01-01", "1985-12-31"),
    ("1985-01-01", "1995-12-31"),
    ("1995-01-01", "2005-12-31"),
    ("2005-01-01", "2015-12-31"),
    ("2015-01-01", "2022-12-31"),
]
# time_periods = [('1958-01-01', '1965-12-31'),
#                 ('1965-01-01', '1975-12-31'),
#                 ('1975-01-01', '1985-12-31'),
#                 ('1985-01-01', '1995-12-31'),
#                 ('1995-01-01', '2005-12-31'),
#                 ('2005-01-01', '2015-12-31'),
#                 ('2015-01-01', '2022-12-31')]

# Variables to plot
variables = ["hfds", "hfds_anomalies", "tauuo", "tauvo"]

# Loop through each variable to create a separate figure for each
for variable in variables:
    # Create a figure with subplots for each time period, with mean on top and std dev below
    num_periods = len(time_periods)
    fig, axes = plt.subplots(
        nrows=2, ncols=num_periods, figsize=(30, 12), sharex=True, sharey=True
    )

    # Set color limits based on the variable for mean plots
    if variable == "hfds":
        mean_vmin, mean_vmax = -15, 15
    elif variable == "hfds_anomalies":
        mean_vmin, mean_vmax = -15, 15
    else:
        mean_vmin, mean_vmax = -0.02, 0.02

    # Plot the mean and standard deviation maps for each period
    for i, (start_date, end_date) in enumerate(time_periods):
        # Calculate the mean map for the specified time period
        mean_map = (
            ds_boundary[variable]
            .sel(time=slice(start_date, end_date))
            .mean("time")
            .compute()
        )

        if variable == "hfds" and start_date == "1975-01-01":
            mean_map_one = mean_map
        elif variable == "hfds" and start_date == "2015-01-01":
            mean_map_last = mean_map

        if variable == "hfds_anomalies" and start_date == "1975-01-01":
            mean_map_hfds_anomalies_one = mean_map
        elif variable == "hfds_anomalies" and start_date == "2015-01-01":
            mean_map_hfds_anomalies_last = mean_map

        # Plot the mean map in the top row
        mean_map.plot(
            ax=axes[0, i],
            add_colorbar=(i == num_periods - 1),
            vmin=mean_vmin,
            vmax=mean_vmax,
        )
        axes[0, i].set_title(f"Mean {variable}: {start_date[:4]}-{end_date[:4]}")

        # Calculate the standard deviation map for the specified time period
        std_map = (
            ds_boundary[variable]
            .sel(time=slice(start_date, end_date))
            .std("time")
            .compute()
        )

        # Set color limits for std dev plot
        std_vmin, std_vmax = 0, std_map.max() / 2

        # Plot the standard deviation map in the bottom row
        std_map.plot(
            ax=axes[1, i],
            add_colorbar=(i == num_periods - 1),
            vmin=std_vmin,
            vmax=std_vmax,
        )
        axes[1, i].set_title(
            f"Standard Deviation {variable}: {start_date[:4]}-{end_date[:4]}"
        )

    # Adjust layout to fit all subplots
    plt.tight_layout()
    fig.suptitle(
        f"Mean and Standard Deviation Maps of {variable} Across Different Periods",
        fontsize=16,
        y=1.02,
    )

    # Save the plot to a file
    plt.savefig(f"mean_std_maps_{variable}.png", bbox_inches="tight", dpi=150)

    # Display the plot
    # plt.show()


# In[33]:


(mean_map_last - mean_map_one).plot(vmax=75)


# In[24]:


(mean_map_last - mean_map_one).weighted(ds_groundtruth.areacello).mean(
    ["x", "y"]
).compute()


# In[25]:


(mean_map_hfds_anomalies_last - mean_map_hfds_anomalies_one).weighted(
    ds_groundtruth.areacello
).mean(["x", "y"]).compute()


# In[34]:


per_5day_increase = 1 * 0.71 * 510 * pow(10, 6) * pow(10, 6) * (60 * 60 * 24 * 5)
per_5day_increase
# per_5day_increase_per_cell = per_5day_increase / (180*360)
# per_5day_increase_per_cell


# In[21]:


hfds_timeseries = (ds_boundary["hfds"] * ds_boundary["areacello"]).mean(
    ["x", "y"]
)  # .plot()


# In[23]:


hfds_timeseries.plot()


# In[32]:


multiplier = 1 / 73
hfds_timeseries = (
    ds_boundary["hfds"].isel(time=slice(-600, None)) * ds_boundary["areacello"]
).mean(["x", "y"])  # .plot()
hfds_timeseries_1wm2 = ds_boundary["hfds"].isel(time=slice(-600, None)) + np.reshape(
    np.arange(ds_boundary.isel(time=slice(-600, None)).time.size) * (multiplier),
    (-1, 1, 1),
)
hfds_timeseries_1wm2 = (hfds_timeseries_1wm2 * ds_boundary["areacello"]).mean(
    ["x", "y"]
)  # .plot()
# plt.ylabel('hfds')


# In[33]:


yearly_averaged_hfds = hfds_timeseries.groupby("time.year").mean("time")
yearly_averaged_hfds.plot(c="k", label="No trend added")
poly_coeffs = yearly_averaged_hfds.polyfit(dim="year", deg=1)
trend = xr.polyval(
    yearly_averaged_hfds["year"], poly_coeffs.polyfit_coefficients
).compute()
trend.plot(c="k")
yearly_averaged_hfds = hfds_timeseries_1wm2.groupby("time.year").mean("time")
yearly_averaged_hfds.plot(c="b", label="trend added")
poly_coeffs = yearly_averaged_hfds.polyfit(dim="year", deg=1)
trend = xr.polyval(
    yearly_averaged_hfds["year"], poly_coeffs.polyfit_coefficients
).compute()
trend.plot(c="b")
plt.ylabel("hfds yearly averaged")


# In[ ]:


# In[48]:


import matplotlib.pyplot as plt
import regionmask
from xmip.regionmask import merged_mask

basins = regionmask.defined_regions.natural_earth_v4_1_0.ocean_basins_50

mask = merged_mask(basins, ds_groundtruth)

atlantic_mask = xr.where(np.logical_or(mask == 0, mask == 1), 1.0, np.nan)
pacific_mask = xr.where(np.logical_or(mask == 2, mask == 3), 1.0, np.nan)
southern_ocean_mask = xr.where(mask == 7, 1.0, np.nan)
indian_ocean_mask = xr.where(mask == 5, 1.0, np.nan)
all_ocean_mask = ds_groundtruth.wetmask.isel(lev=0)

masks = atlantic_mask.to_dataset(name="Atlantic")
masks["Pacific"] = pacific_mask
masks["Southern"] = southern_ocean_mask
masks["Indian"] = indian_ocean_mask
masks["All"] = all_ocean_mask

# Define your time periods
time_periods = [
    ("1958-01-01", "1965-12-31"),
    ("1965-01-01", "1975-12-31"),
    ("1975-01-01", "1985-12-31"),
    ("1985-01-01", "1995-12-31"),
    ("1995-01-01", "2005-12-31"),
    ("2005-01-01", "2015-12-31"),
    ("2015-01-01", "2022-12-31"),
]


# Loop through each variable to create a separate figure for each
for variable in variables:
    # Number of regions
    num_regions = len(masks)

    # Create a figure with subplots for each region
    fig, axes = plt.subplots(
        nrows=num_regions, ncols=1, figsize=(14, 4 * num_regions), sharex=True
    )

    # Loop through each mask (region) and plot in a separate subplot
    for ax, (mask_name, mask) in zip(axes, masks.items()):
        for start_date, end_date in time_periods:
            # Remove the seasonal cycle by subtracting the climatology from the original data
            deseasonalized_data = ds_boundary[variable].sel(
                time=slice(start_date, end_date)
            )

            # Calculate the timeseries for the specified region and time period
            timeseries = (
                (deseasonalized_data * mask)
                .weighted(ds_boundary.areacello)
                .mean(["x", "y"])
                .groupby("time.year")
                .mean("time")
                .compute()
            )

            # Plot the timeseries on the current subplot
            label = f"{mask_name}: {start_date[:4]}-{end_date[:4]}"
            timeseries.plot(ax=ax, label=label)

        # Customize each subplot
        ax.set_title(f"Timeseries of {variable} in {mask_name} (Yearly Averages)")
        ax.set_xlabel("Time")
        ax.set_ylabel(variable)
        ax.grid(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), title="Regions & Periods")

    # Adjust layout to fit all subplots and legends
    plt.tight_layout()
    fig.suptitle(
        f"Timeseries of {variable} Across Different Regions (Yearly Averages)",
        fontsize=16,
        y=1.02,
    )
    print("Saved ", variable)

    plt.savefig(f"ground_{variable}", bbox_inches="tight", dpi=150)
    # Display the plot
    # plt.show()


# In[ ]:


# In[ ]:


# In[11]:


import matplotlib.pyplot as plt
import xarray as xr


# In[25]:


import matplotlib.pyplot as plt

# Create a figure with two subplots, arranged vertically, with shared x-axis
fig, axes = plt.subplots(
    nrows=2, figsize=(7, 12), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
)

# Plot the area plot (2D map) for lev=0 on the first subplot (top) without the colorbar
ds_groundtruth.isel(time=10, lev=0).wetmask.plot(ax=axes[0], add_colorbar=False)
axes[0].set_title("Area Plot of Wetmask at lev=0")
axes[0].set_xlabel("")  # Hide x-axis label for the top plot to keep things clean
axes[0].set_ylabel("Latitude")

# Plot the cross-section (latitude slice) on the second subplot (bottom) without the colorbar
ds_groundtruth.sel(y=20, method="nearest").isel(time=10).wetmask.plot(
    ax=axes[1], add_colorbar=False
)
axes[1].set_title("Cross-section of Wetmask at Latitude 0")
axes[1].invert_yaxis()  # Invert the y-axis for the cross-section plot
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Depth (lev)")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()


# In[ ]:


ds_std = xr.open_zarr(
    "/pscratch/sd/s/suryad/data/3D_data_OM4_5daily_v0.2.1_with_hfds_anom_1975_stds"
)
ds_std.load()


# In[ ]:


scaling_residual = xr.open_zarr(
    "/pscratch/sd/s/suryad/data/3D_data_OM4_5daily_v0.2.1_with_hfds_anom_1975_scaling_residuals"
)
scaling_residual.load()


# In[ ]:


(ds_std / scaling_residual).compute()


# In[ ]:


outs = [
    k + str(j)
    for k in ["thetao_lev_", "so_lev_"]
    for j in [str(lev).replace(".", "_") for lev in ds["lev"].values]
] + ["zos"]


# In[ ]:


import matplotlib.pyplot as plt

x = [
    2.5,
    10.0,
    22.5,
    40.0,
    65.0,
    105.0,
    165.0,
    250.0,
    375.0,
    550.0,
    775.0,
    1050.0,
    1400.0,
    1850.0,
    2400.0,
    3100.0,
    4000.0,
    5000.0,
    6000.0,
]
y = [
    36.99,
    47.96,
    54.28,
    73.49,
    92.2,
    123.6,
    132.8,
    163.6,
    205.3,
    279.1,
    414.0,
    486.7,
    573.1,
    963.9,
    1788.0,
    1410.0,
    866.2,
    611.1,
    1205.0,
]
plt.plot(x, y)
plt.xlabel("Depth")
plt.ylabel("Salinity Loss Scaled")


# In[ ]:


import matplotlib.pyplot as plt

x = [
    2.5,
    10.0,
    22.5,
    40.0,
    65.0,
    105.0,
    165.0,
    250.0,
    375.0,
    550.0,
    775.0,
    1050.0,
    1400.0,
    1850.0,
    2400.0,
    3100.0,
    4000.0,
    5000.0,
    6000.0,
]
y = [
    45.63,
    49.43,
    59.52,
    71.43,
    83.41,
    83.57,
    97.04,
    139.7,
    154.8,
    152.9,
    152.8,
    170.5,
    238.2,
    322.3,
    386.5,
    387.9,
    336.9,
    432.6,
    764.6,
]
plt.plot(x, y)
plt.xlabel("Depth")
plt.ylabel("Temperature Loss Scaled")
