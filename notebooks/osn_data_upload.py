import fsspec
import xarray as xr

fs_osn = fsspec.filesystem(
    's3',
    profile='ocean_emulator_write',  ## This is the profile name you configured above.
)

local_path = "/pscratch/sd/s/suryad/data/3D_data_OM4_5daily_v0.2.1_with_hfds_anom_1975"
osn_path = "/emulators/sd5313/Samudra/OM4"

ds = xr.open_zarr(local_path)
mapper = fs_osn.get_mapper(osn_path)

import time
start = time.time()
ds.to_zarr(mapper)
print(f'Time taken: {time.time() - start} seconds')