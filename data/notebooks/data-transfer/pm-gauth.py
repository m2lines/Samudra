#!/usr/bin/env python

# In[1]:


get_ipython().system("pip install google-auth gcsfs jupyterlab xarray zarr dask")


# In[1]:


import gcsfs
import xarray as xr
from google.oauth2.credentials import Credentials

# import an access token
# - option 1: read an access token from a file
with open("token.txt") as f:
    access_token = f.read().strip()

# setup a storage client using credentials
credentials = Credentials(access_token)
fs = gcsfs.GCSFileSystem(token=credentials)


# In[3]:


ds = xr.open_dataset(
    fs.get_mapper(
        "gs://leap-scratch/jbusecke/ocean-emulators/test_CMIP6_GFDL-CM4.piControl.r1i1p1f1.zarr"
    ),
    engine="zarr",
    chunks={},
)


# In[4]:


ds


# In[9]:


ds.isel(time=slice(0, 3)).to_zarr(
    "/pscratch/sd/s/shubhamg/test.zarr",
    encoding={v: {"compressor": None} for v in ds.variables},
    consolidated=True,
    mode="w",
)


# In[10]:


from dask.diagnostics import ProgressBar

# In[12]:


with ProgressBar():
    ds.to_zarr(
        "/pscratch/sd/s/shubhamg/test_CMIP6_GFDL-CM4.piControl.r1i1p1f1.zarr",
        encoding={v: {"compressor": None} for v in ds.variables},
        consolidated=True,
        mode="w",
    )


# In[ ]:
