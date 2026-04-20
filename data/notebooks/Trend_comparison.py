#!/usr/bin/env python

# # Evaluating the prediction performance in space
#
# Looking at two error metrics:
#
# - The differenc in the slope of a linear regression in time at every grid point (as an indicator of the ability to reproduce long term trends, one of the main goals of our project)
# - The RMSE over the entire rollout. This is more similar to the training, but at no point in the training are we looking at such long rollouts.
#
#
# >[!WARNING]
# >Make sure to install the ocean_emulator package, or these imports below will fail

# In[1]:


get_ipython().system("pip install -e ../")


# In[1]:


import matplotlib.pyplot as plt
import xarray as xr
from dask.diagnostics import ProgressBar
from xarrayutils.plotting import linear_piecewise_scale

from ocean_emulators.postprocessing import post_processor

get_ipython().run_line_magic("matplotlib", "inline")


# In[2]:


with ProgressBar():
    url_input = "gs://leap-persistent/jbusecke/ocean-emulators/OM4_5daily_v0.2.1.zarr"
    ds = xr.open_dataset(url_input, engine="zarr", chunks={})
    ds_truth = (
        ds.isel(time=slice(4143, 4743))
        .astype("float32")
        .chunk({"time": 10, "x": -1, "y": -1, "lev": -1})
    )

    url_prediction = (
        "gs://leap-persistent/sd5313/convnext_epoch-70_train-OM4v0.2.1_eval-OM4v0.2.1"
    )
    ds_pred_raw = xr.open_dataset(url_prediction, engine="zarr")
    ds_pred = (
        post_processor(ds_pred_raw, ds_truth)
        .astype("float32")
        .chunk({"time": 10, "x": -1, "y": -1, "lev": -1})
    )


# >[!NOTE]
# >For Surya: Lets figure out the chunking issue so you can upload the fully postprocessed data to the cloud

# In[5]:


# combine truth and prediction into a single dataset for convenience.

ds_combined = xr.concat(
    [
        ds_truth[[*list(ds_pred.data_vars)]].assign_coords(model="truth"),
        ds_pred.assign_coords(model="prediction"),
    ],
    dim="model",
)
ds_combined


# In[6]:


# Getting the trend by fitting a 2nd deg poly, and only retaining the second coefficient (the first is the mean).
trend = ds_combined.polyfit("time", deg=1).isel(degree=1)
with ProgressBar():
    trend = trend.load()
trend_diff = trend.diff("model")
trend_diff


# In[7]:


# Compute the RMSE
rmse = (ds_combined.diff("model") ** 2).mean("time")
with ProgressBar():
    rmse = rmse.load()
rmse


# In[8]:


# Prototype plots
kwargs = {"robust": True}

for var in ["uo", "vo", "so", "thetao"]:
    da = trend_diff[f"{var}_polyfit_coefficients"].squeeze().reset_coords(drop=True)
    da.name = var

    plt.figure(figsize=[12, 8])
    plt.subplot(3, 2, 1)
    da.mean("y").plot(yincrease=False, **kwargs)
    ax = plt.gca()
    linear_piecewise_scale(1000, 5)
    # indicate the point between the different scalings
    ax.axhline(1000, color="0.5", ls="--")
    # Rearange the yticks
    ax.set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
    ax.set_title("Trend Slope Lat Mean [pred-truth]")

    plt.subplot(3, 2, 3)
    da.mean("x").plot(yincrease=False, **kwargs)
    ax = plt.gca()
    linear_piecewise_scale(1000, 5)
    # indicate the point between the different scalings
    ax.axhline(1000, color="0.5", ls="--")
    # Rearange the yticks
    ax.set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
    ax.set_title("Trend Slope Lon Mean [pred-truth]")

    plt.subplot(3, 2, 5)
    ax = plt.gca()
    da.mean("lev").plot(**kwargs)
    ax.set_title("Trend Slope Depth Mean [pred-truth]")

    da = rmse[var].squeeze().reset_coords(drop=True)

    plt.subplot(3, 2, 2)
    da.mean("y").plot(yincrease=False, **kwargs)
    ax = plt.gca()
    linear_piecewise_scale(1000, 5)
    # indicate the point between the different scalings
    ax.axhline(1000, color="0.5", ls="--")
    # Rearange the yticks
    ax.set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
    ax.set_title("RMSE Lat Mean [pred-truth]")

    plt.subplot(3, 2, 4)
    da.mean("x").plot(yincrease=False, **kwargs)
    ax = plt.gca()
    linear_piecewise_scale(1000, 5)
    # indicate the point between the different scalings
    ax.axhline(1000, color="0.5", ls="--")
    # Rearange the yticks
    ax.set_yticks([0, 250, 500, 750, 1000, 3000, 5000])
    ax.set_title("RMSE Lon Mean [pred-truth]")

    plt.subplot(3, 2, 6)
    ax = plt.gca()
    da.mean("lev").plot(**kwargs)
    ax.set_title("RMSE Depth Mean [pred-truth]")
    plt.tight_layout()


# In[ ]:
