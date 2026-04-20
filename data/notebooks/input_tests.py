#!/usr/bin/env python

# # Checking all input datasets systematically
#
# ## TODO
# - [ ] Use dask-gateway
# - [ ] Get rid of manual fixes

# In[1]:


import xarray as xr
from distributed import Client

from ocean_emulators.dataset_validation import ds_input_validate

# In[2]:


client = Client()
client


# In[3]:


input_paths = [
    # "gs://leap-persistent/sd5313/input_OM4v0.0",
    "gs://leap-persistent/jbusecke/ocean-emulators/OM4_5daily_v0.2.1.zarr",
    # "gs://leap-persistent/jbusecke/ocean-emulators/CMIP6_GFDL-CM4.piControl.r1i1p1f1_v0.1.zarr"
]
input_dict = {
    path.split("/")[-1]: xr.open_dataset(path, engine="zarr", chunks={})
    for path in input_paths
}


# ## Manual fixes
# There are issues with the current datasets, which can be fixed manually for now.
# Should be seen as a continously updated to-do list for the preprocessing.

# In[4]:


# from ocean_emulators.preprocessing import manual_v0_fixes
# input_dict['input_OM4v0.0'] = manual_v0_fixes(input_dict['input_OM4v0.0'])

# # add wetmask to CM4
# import numpy as np
# ds = input_dict['CMIP6_GFDL-CM4.piControl.r1i1p1f1_v0.1.zarr']
# wetmask = ~np.isnan(ds.thetao.isel(time=0)).reset_coords(drop=True)
# ds = ds.assign_coords(wetmask=wetmask)
# input_dict['CMIP6_GFDL-CM4.piControl.r1i1p1f1_v0.1.zarr'] = ds


# In[5]:


# this takes VERY LONG, even on a 64 core machine!
for name, ds in input_dict.items():
    try:
        ds_input_validate(ds, deep=True)
        print(f"{name} ✅")
    except Exception as e:
        print(f"{name} ❌")
        print(f"Failed with {e}")


# In[ ]:
