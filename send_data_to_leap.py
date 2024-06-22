### OPTION 1
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
# mapper = fs.get_mapper(
#     "gs://leap-persistent/sd5313/convnextpredepoch50.zarr"
# )
# ds = xr.open_zarr('/scratch/sd5313/M2Lines/emulator/Ocean_Emulator/Preds/ConvNext UNet Train3DSurfaceEval3DEpoch50_Train_global_3D_Test_global_3D_all_N_train_4000_Lateral_Data_025_no_smooth/Pred_lateral_Fast_Data_025_global_3D_all_N_samples_4000_rand_seed_1.zarr')

# Test data
mapper = fs.get_mapper("gs://leap-persistent/sd5313/OM4_Horizontal_Regrid_Old.zarr")
ds = xr.open_zarr(
    "/vast/sd5313/data/m2lines/3D_ocean_data/OM4_Horizontal_Regrid_Old.zarr"
)

with ProgressBar():
    ds.to_zarr(mapper)
