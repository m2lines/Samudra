#!/usr/bin/env python

# In[1]:


import fsspec

fs_osn = fsspec.filesystem(
    "s3",
    profile="ocean_emulator_write",  ## This is the profile name you configured above.
)


# In[2]:


fs_osn.ls("emulators/sd5313/Samudra")


# #### Data check

# In[2]:


mapper = fs_osn.get_mapper("emulators/jbusecke/ocean-emulators/CM4_5daily_v0.4.0.zarr")


# In[3]:


import xarray as xr

ds = xr.open_zarr(mapper, consolidated=True)
ds


# In[ ]:


# check data size
print("Total data size of CM4: ", ds.nbytes / 1e9, "GB")


# In[14]:


import cftime

# Example DatetimeNoLeap object
no_leap_date = cftime.DatetimeNoLeap(150, 3, 29, 12, 0, 0)  # Replace with your own date

# Convert to DatetimeJulian
julian_date = cftime.DatetimeJulian(
    no_leap_date.year,
    no_leap_date.month,
    no_leap_date.day,
    no_leap_date.hour,
    no_leap_date.minute,
    no_leap_date.second,
)

julian_date


# In[4]:


ds.thetao.time.values


# In[9]:


data2.time.values


# In[10]:


get_ipython().run_cell_magic(
    "time",
    "",
    '# All Sets - Depth - Thetao\nimport matplotlib.pyplot as plt\n\nlevel_slices = [slice(None, 700), slice(700, 2000), slice(2000, None)]\n# level_slices = [slice(None, 300), slice(300, 600), slice(600,)]\ntitles = [\n    r"$\\theta_O$ Upper 0.7km",\n    r"$\\theta_O$ 0.7-2.0km",\n    r"$\\theta_O$ 2.0km to bottom",\n]\n\ndata = ds\ndata2\n\nds_input = xr.open_zarr(\n    os.path.join("/pscratch/sd/s/suryad/data", "OM4_5daily_v0.2.1.zarr")\n)\nds_full_groundtruth = ds_input.astype("float32")\ndata2 = ds_full_groundtruth\n\nplt.rcdefaults()\nplt.rcParams.update({"font.size": 14})\nfig, axs = plt.subplots(\n    3, 1, figsize=(16, 9), gridspec_kw={"wspace": 0.25, "hspace": 0.5}\n)\n\nfor i, lev_slice in enumerate(level_slices):\n    thetao = (\n        (data["thetao"].sel(lev=lev_slice))\n        .weighted(data["areacello"] * data["dz"])\n        .mean(["x", "y", "lev"])\n    )\n    thetao = thetao.rename(r"$\\theta_O$")\n    thetao = thetao.assign_attrs(units=r"$\\degree C$")\n    thetao.plot(ax=axs[i], label="CM4", color="k")\n\n    #     thetao = (data2[\'thetao\'].sel(lev=lev_slice)).weighted(data2[\'areacello\'] * data2[\'dz\']).mean([\'x\', \'y\', \'lev\'])\n    #     thetao = thetao.rename(r\'$\\theta_O$\')\n    #     thetao = thetao.assign_attrs(units=r\'$\\degree C$\')\n    #     thetao.plot(ax=axs[i], label=\'OM4\', color=\'red\')\n\n    axs[i].set_title(titles[i], fontsize=14)\n    if i == 0:\n        axs[i].legend(fontsize=13)\n\nplt.show()\n',
)


# Weighted Tests

# In[1]:


import xarray as xr

# In[10]:


# Create a dataset
ds = xr.Dataset(
    {
        "temperature": (["x", "y"], [[2, 2], [2, 2]]),
        "precipitation": (["x", "y"], [[0, 1], [2, 3]]),
    },
    coords={"x": [0, 1], "y": [0, 1]},
)
vol = xr.DataArray([[3, 2], [3, 2]], dims=["x", "y"], coords={"x": [0, 1], "y": [0, 1]})


# In[17]:


ds.temperature.weighted(vol).mean()


# In[23]:


((ds.temperature) * (vol / vol.sum())).sum()


# In[ ]:
