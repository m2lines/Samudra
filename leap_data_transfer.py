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

# Prediction
mapper = fs.get_mapper(
    "gs://leap-persistent/sd5313/convnextpredepoch50.zarr"
)
ds = xr.open_zarr('/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/ConvNext UNet Train3DSurfaceEval3DEpoch50_Train_global_3D_Test_global_3D_all_N_train_4000_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_4000_rand_seed_1.zarr')

# Test data
mapper = fs.get_mapper(
    "gs://leap-persistent/sd5313/OM4_Horizontal_Regrid_Old.zarr"
)
ds = xr.open_zarr('/vast/sd5313/data/m2lines/3D_ocean_data/OM4_Horizontal_Regrid_Old.zarr')

with ProgressBar():
    ds.to_zarr(mapper)

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


