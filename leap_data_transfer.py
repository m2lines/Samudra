################################################################################
#### Data to Leap

import xarray as xr
import gcsfs
import json
from dask.diagnostics import ProgressBar


with open(
    "/home/sd5313/.config/gcloud/application_default_credentials.json"
) as f:  # 🚨 make sure to enter the `.json` file from step 7
    token = json.load(f)
fs = gcsfs.GCSFileSystem(token=token)

### Prediction
mapper = fs.get_mapper(
    "gs://leap-persistent/sd5313/convnext100MAddedSSTB_epoch-60_300_train-OM4v0.0_eval-OM4v0.0"
)

ds = xr.open_zarr(
    "/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/ConvNext UNet Train3DEval3D100M_SSTB_Epoch60_300_Train_global_3D_Test_global_3D_all_N_train_4000_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_4000_rand_seed_1.zarr"
)

ds1 = xr.open_zarr(
    "/vast/sd5313/data/m2lines/3D_ocean_data/3D_data", 
)

ds1 = ds1.rename({'x':'x_i', 'y':'y_i'}).rename({'x_i':'y', 'y_i':'x'})
ds1 = ds1['thetao_lev_0'].isel(time=slice(4441, 4741)) # Last 300 / 600
ds1 = ds1.expand_dims('var', axis=-1)
ds1['var'] = [38]

import numpy as np
before = ds.isel(var=slice(0, 38)).to_array().squeeze().to_numpy()
after = ds.isel(var=slice(38, None)).to_array().squeeze().to_numpy()
middle = ds1.to_numpy()

final = np.concatenate([before, middle, after], axis=-1)

final_ds = xr.DataArray(final, dims=['time', 'x', 'y', 'var'])

with ProgressBar():
    final_ds.to_zarr(mapper)

# mapper = fs.get_mapper("gs://leap-persistent/sd5313/OM4_train_data_stds")
# ds = xr.open_zarr("/vast/sd5313/data/m2lines/3D_ocean_data/3D_data_stds")

# with ProgressBar():
#     ds.to_zarr(mapper)

### Test data
# mapper = fs.get_mapper(
#     "gs://leap-persistent/sd5313/OM4_Horizontal_Regrid_Old.zarr"
# )
# ds = xr.open_zarr('/vast/sd5313/data/m2lines/3D_ocean_data/OM4_Horizontal_Regrid_Old.zarr')

# with ProgressBar():
#     ds.to_zarr(mapper)

################################################################################
#### Data from Leap
# import gcsfs
# import xarray as xr
# from google.cloud import storage
# from google.oauth2.credentials import Credentials
# from dask.diagnostics import ProgressBar

# # import an access token
# # - option 1: read an access token from a file
# with open("token.txt") as f:
#     access_token = f.read().strip()

# # setup a storage client using credentials
# credentials = Credentials(access_token)
# fs = gcsfs.GCSFileSystem(token=credentials)
# # CM4 data
# ds = xr.open_dataset(fs.get_mapper('gs://leap-scratch/jbusecke/ocean-emulators/test_CMIP6_GFDL-CM4.piControl.r1i1p1f1.zarr'), engine='zarr', chunks={})
# print(ds)

# with ProgressBar():
#     ds.to_zarr('/vast/sd5313/data/m2lines/3D_ocean_data/test_CMIP6_GFDL-CM4.piControl.r1i1p1f1.zarr',
#                                  encoding = {v:{'compressor':None} for v in ds.variables},
#                                  consolidated=True, mode='w')
