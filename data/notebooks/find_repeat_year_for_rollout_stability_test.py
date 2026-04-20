#!/usr/bin/env python

# # Lets identify a single year that has very little net heatflux.

# In[1]:


# !pip install -e /home/jovyan/PROJECTS/ocean_emulators/


# In[2]:


import hvplot.pandas  # noqa
import xarray as xr
from distributed import Client

from ocean_emulators.preprocessing import manual_v0_fixes

# In[3]:


client = Client()
client


# In[4]:


path = "gs://leap-persistent/sd5313/input_OM4v0.0"


# In[5]:


ds = manual_v0_fixes(xr.open_dataset(path, engine="zarr", chunks={}))
ds_years = ds.resample(time="1YS").mean("time")
hf = ds_years.hfds.load()


# In[6]:


df = xr.Dataset(
    {
        "mean": hf.weighted(hf.areacello).mean(["x", "y"]),
        "std": hf.weighted(hf.areacello).std(["x", "y"]),
        "year": xr.DataArray([t.year for t in hf.time.data], dims="time"),
    }
).to_dataframe()


# In[7]:


df.hvplot.scatter(x="std", y="mean", color="year")
