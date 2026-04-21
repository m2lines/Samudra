#!/usr/bin/env python

# # Preprocessing OM4 data
#
# v0.0 indicates Adams preprocessing
#
# - I have used https://github.com/m2lines/ocean_emulators/blob/main/notebooks/transfer-om4-from-greene.ipynb to move a subset of variables from Greene to the cloud.
#
# ## TODO + next steps
# - TODO: With more details of the grid metrics I can try a more sophisticated filter, but lets wait for how much data we are loosin
# - Get the original ocean/land fraction? That might be important for AI2
# - Get the additional variables (flux components, sea ice)

# In[1]:


# import dask
# from dask_gateway import Gateway

# gateway = Gateway()

# # close existing clusters
# open_clusters = gateway.list_clusters()
# print(list(open_clusters))
# if len(open_clusters)>0:
#     for c in open_clusters:
#         cluster = gateway.connect(c.name)
#         cluster.shutdown()

# options = gateway.cluster_options()
# # options.worker_resource_allocation = '4CPU, 28.9Gi'
# options.worker_resource_allocation = '16CPU, 115.8Gi'
# options.idle_timeout_minutes = 30

# # get more mem per worker for the shitty scaling of the flux calc
# # Did not get this to work, so back to full utilization and `to_zarr_split`.
# display(options)


# In[2]:


# # Create a cluster with those options
# cluster = gateway.new_cluster(options)
# display(cluster)
# client = cluster.get_client()
# # cluster.scale(2)
# cluster.scale(50)
# client


# In[3]:


# # notes for xarray-schema
# - option to evaluate existence of dims, but not the order?
# - CoordsSchema not implemented?
# - # can I check that this is actually cftime?


# In[4]:


# !pip install -e /home/jovyan/PROJECTS/ocean_emulators/


# In[5]:


# !pip install xarray-schema


# In[6]:


# just for the split zarr writing
# !pip install git+https://github.com/ocean-transport/scale-aware-air-sea@kwargs-for-open-close


# In[37]:


from distributed import Client, LocalCluster

cluster = LocalCluster(threads_per_worker=8, n_workers=8)
# cluster = LocalCluster(threads_per_worker=2, n_workers=32)
client = Client(cluster)
client


# In[8]:


import cartopy.crs as ccrs
import fsspec
import matplotlib.pyplot as plt
import xarray as xr
from ocean_preprocessing.dataset_validation import (
    ds_input_validate,
    ds_processed_validate,
)
from ocean_preprocessing.plotting import rotated_vectors_qc_plots
from ocean_preprocessing.preprocessing import (
    horizontal_regrid,
    rotate_vectors,
    spatially_filter,
)
from ocean_preprocessing.simulation_preprocessing.gfdl_om4 import om4_preprocessing
from ocean_preprocessing.utils import get_git_url_hash

# In[9]:
from scale_aware_air_sea.utils import to_zarr_split

get_ipython().run_line_magic("matplotlib", "inline")


# ## Creating and Validating `ds_processed`

# In[ ]:


zarr_data_path = "gs://leap-persistent/jbusecke/ocean_emulators/OM4/OM4_raw_test.zarr"
nc_grid_path = "gs://leap-persistent/sd5313/OM4-5daily/ocean_static_no_mask_table.nc"
nc_mosaic_path = "gs://leap-persistent/sd5313/OM4-5daily/ocean_hgrid.nc"
ds_processed = om4_preprocessing(zarr_data_path, nc_grid_path, nc_mosaic_path)

# only check a few time steps for now to save time
ds_processed_validate(ds_processed.isel(time=slice(0, 10)), deep=True)

ds_processed


# ## Rotate velocities (this is not specific to the model)

# In[ ]:


u_rotated, v_rotated = rotate_vectors(
    ds_processed.uo, ds_processed.vo, ds_processed.angle
)
rotated_vectors_qc_plots(ds_processed.uo, ds_processed.vo, u_rotated, v_rotated)


# In[ ]:


ds_processed["uo"] = u_rotated
ds_processed["vo"] = v_rotated
# TODO: make sure rotation is noted in an attr?


# ### Smooth the output
#
# - I saw no difference when filtering each level independently as long as I provided a 3d wetmask

# In[ ]:


ds_filtered = spatially_filter(ds_processed, ds_processed.wetmask)
ds_filtered


# In[14]:


for l in [0, 5, 16]:
    filtered = ds_filtered["so"].isel(time=10, lev=l).load()
    unfiltered = ds_processed["so"].isel(time=10, lev=l).load()
    diff = unfiltered - filtered

    plt.figure(figsize=[15, 8])
    unfiltered.plot(robust=True)

    plt.figure(figsize=[15, 8])
    filtered.plot(robust=True)

    plt.figure(figsize=[15, 8])
    diff.plot(robust=True)

    fig, ax = plt.subplots(
        figsize=[7, 7],
        subplot_kw={"projection": ccrs.NorthPolarStereo(central_longitude=-20.0)},
    )
    diff.plot(ax=ax, vmax=0.05, transform=ccrs.PlateCarree(), x="lon", y="lat")
    ax.set_extent([-20, 340, 78, 90], ccrs.PlateCarree())


# >[!WARNING]
# >We are not properly filtering across the tripolar fold. I think it is minor, but you can see some artifacts above.

# ## Regrid onto gaussian grid
#
# I spent way too much time with this today in order to get the closest in terms of preserving the global integral of temp/salinity, but I think I got it sufficiently close.
#
# Some notes that helped me with the thinking [here](https://xesmf.readthedocs.io/en/latest/notebooks/Masking.html) and [here](https://xesmf.readthedocs.io/en/latest/notebooks/Compare_algorithms.html). The key was to recalculate the new `ocean area fraction` as part of the horizontal regridding.

# In[15]:


target_grid_path = "gs://leap-persistent/m2lines/ai2_colab/2024-08-01-sample-raw-CM4-data/gaussian_grid_180_by_360.nc"

with fsspec.open(target_grid_path) as f:
    ds_target_grid = xr.open_dataset(f).load()
ds_target_grid = ds_target_grid.rename(
    {
        "grid_x": "x_b",
        "grid_y": "y_b",
        "grid_xt": "x",
        "grid_yt": "y",
        "grid_lon": "lon_b",
        "grid_lat": "lat_b",
        "grid_lont": "lon",
        "grid_latt": "lat",
    }
)
ds_target_grid


# In[16]:


ds_regridded = horizontal_regrid(ds_filtered, ds_target_grid)
ds_regridded


# In[17]:


ds_regridded.so.isel(time=100, lev=3).plot()


# >[!NOTE]
# >We are loosing thin land bridges like panama!

# In[18]:


ds_regridded.ocean_fraction.isel(lev=3).plot()


# #### Compare weighted mean of the different 'steps'
#
# >[!NOTE]
# >I am using the mean here, since it does not rely on the absolute values of the area, which are slightly biased compared to the values from GFDL (see `./notebooks/xesmf_area_comparison.ipynb` for details
#
# >[!WARNING]
# >I am currently not extracting the original ocean area fraction from OM4 and am thus using the wetmask (wet=all area is ocean).

# In[26]:


ds_filtered = ds_filtered.assign_coords(
    ocean_fraction=ds_filtered.wetmask.astype("float64")
)
ds_processed = ds_processed.assign_coords(
    ocean_fraction=ds_processed.wetmask.astype("float64")
)


# In[27]:


def int_ds(ds):
    # this works better than `.sum()`, probably since the area
    # from the original dataset is not equal to the recomputed area
    w = ds.dz * ds.areacello * ds.ocean_fraction
    return ds.weighted(w).mean(["x", "y", "lev"]).isel(time=slice(0, 10)).load()


from dask.diagnostics import ProgressBar

with ProgressBar():
    int_test = int_ds(ds_regridded[["thetao", "so"]])
    int_filtered = int_ds(ds_filtered[["thetao", "so"]])
    int_raw = int_ds(ds_processed[["thetao", "so"]])


# In[28]:


int_raw.thetao.plot(label="baseline")
int_filtered.thetao.plot(ls="--", label="filtered")
int_test.thetao.plot(ls="-.", label="regridded")
plt.legend()
plt.title("Comparison of global weighted mean thetao")


# In[29]:


((int_test - int_raw) / int_raw * 100).thetao.plot(label="thetao")
((int_test - int_raw) / int_raw * 100).so.plot(label="so")
plt.ylabel("relative error [%]")
plt.legend()
plt.title("Relative error of regridded global mean")
plt.axhline(0, color="0.3", ls="--")


# ## Reduce precision and validate output

# In[30]:


ds_input = ds_regridded.astype("float32")
ds_input.attrs["m2lines/ocean-emulators_git_hash"] = get_git_url_hash()


# ## Write this bad boy to the bucket

# In[42]:


# short deep validation (full validation will be done on the written dataset)
ds_input_validate(ds_input.isel(time=slice(0, 5)).load(), deep=True)


# In[32]:


ds_input


# In[ ]:


# ds_input.to_zarr("gs://leap-persistent/jbusecke/ocean-emulators/OM4_5daily_v0.2.zarr")

import gcsfs

fs = gcsfs.GCSFileSystem()
# new version.
path = "gs://leap-persistent/jbusecke/ocean-emulators/OM4_5daily_v0.3.0.zarr"
# fs.rm(path, recursive=True)
mapper = fs.get_mapper(path)
to_zarr_split(
    ds_input, mapper, split_interval=30
)  # this is hella slow on a 64 core machine.
# TODO increase interval + get coiled or dask-gateway cluster.


# In[44]:


ds_rewrite = xr.open_zarr(
    "gs://leap-persistent/jbusecke/ocean-emulators/OM4_5daily_v0.2.zarr", chunks={}
)
ds_rewrite


# In[ ]:


# manual fix of the lon_b, lat_b coords (they ended up as variables).

"""
I should write a test for this...
"""


# In[46]:


# rewrite with new bugfix increment
ds_rewrite_fixed = ds_rewrite.set_coords(["lon_b", "lat_b"])
ds_rewrite_fixed.to_zarr(
    "gs://leap-persistent/jbusecke/ocean-emulators/OM4_5daily_v0.2.1.zarr"
)


# In[ ]:
