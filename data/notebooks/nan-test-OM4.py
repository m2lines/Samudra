#!/usr/bin/env python

# # OM4 nan test

# In[1]:


# In[2]:
import glob

import xarray as xr

from ocean_emulators.preprocessing import test_nan_consistency

directory = "/scratch/aa9537/OM4-5daily/"
files = sorted(glob.glob(f"{directory}*ocean_5daily.nc"))
# grid_path =
# files
pick_vars = [
    # 'average_DT',
    # 'average_T1',
    # 'average_T2',
    "hfds",
    "so",
    "tauuo",
    "tauvo",
    "thetao",
    "time_bnds",
    "uo",
    "vo",
    "zos",
]


# In[3]:


ds_raw = xr.open_mfdataset(
    files,
    use_cftime=True,
    parallel=True,
    chunks={"time": 1},
    combine="nested",
    concat_dim="time",
)
ds = ds_raw[pick_vars]
ds = ds.rename({"z_l": "lev"})
ds = ds.astype("float32")
ds


# In[4]:


ds_raw


# This dataset still has staggered grid variables, and we cannot apply the nan-consistency logic as is.
# Instead we compare all variables on each grid location seperately.

# In[5]:


ds_u = ds[[v for v in ds.data_vars if set(["xq", "yh"]).issubset(set(ds[v].dims))]]
ds_v = ds[[v for v in ds.data_vars if set(["xh", "yq"]).issubset(set(ds[v].dims))]]
ds_t = ds[[v for v in ds.data_vars if set(["xh", "yh"]).issubset(set(ds[v].dims))]]


# In[6]:


from dask.diagnostics import ProgressBar

test_dict = {}
for test_ds, id in [(ds_t, "tracer"), (ds_u, "u"), (ds_v, "v")]:
    with ProgressBar():
        try:
            test_nan_consistency(test_ds)
            test_dict[id] = "passed"
            print(id, "passed")
        except Exception as e:
            test_dict = str(e)
            print(id, f"failed with {e}")


# In[7]:


test_dict
