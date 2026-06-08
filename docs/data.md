<!--
SPDX-FileCopyrightText: 2026 Ocean Emulator Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Data

## What is OM4?

[OM4](https://www.gfdl.noaa.gov/om4-0/) is a physical Ocean simulation produced by [NOAA's Geophysical Fluid Dynamics Laboratory](https://www.gfdl.noaa.gov/).
It is the ocean and sea-ice component of the [CM4 Global Climate Model](https://www.gfdl.noaa.gov/coupled-physical-model-cm4/), which is a coupled climate model that
includes the atmosphere, ocean, sea-ice, and land. OM4 and CM4 are part of [CMIP6, the Coupled Model Intercomparison
Project Phase 6](https://pcmdi.llnl.gov/CMIP6/), which is a large, multi-institutional coupled climate model.

OM4 is natively stored in a tripolar grid in the NetCDF format. We have taken steps to make the data easier to work with
in the machine learning context.

## Taking a look at each dataset

Our data is stored in [Zarr](https://zarr.dev) and is canonically opened with [Xarray](https://xarray.dev). Here is a
quick demonstration on how to open each processed dataset we make available:

```shell
>>> import xarray as xr
>>> # One degree data with guassian filtering applied
>>> # > NOTE: We recommend using the non-filtered data (see the next dataset)
>>> ds = xr.open_zarr('https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2025-11/om4_onedeg_filter/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 98GB
Dimensions:    (time: 4745, y: 180, x: 360)
Coordinates:
  * time       (time) object 38kB 1958-01-03 12:00:00 ... 2022-12-29 12:00:00
  * x          (x) float64 3kB 0.5 1.5 2.5 3.5 4.5 ... 356.5 357.5 358.5 359.5
  * y          (y) float64 1kB -89.24 -88.25 -87.25 -86.26 ... 87.25 88.25 89.24
Data variables: (12/99)
    hfds       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    mask_0     (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_1     (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_10    (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_11    (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_12    (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    ...         ...
    vo_5       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_6       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_7       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_8       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_9       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    zos        (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
Attributes:
    hfds:                              {'cell_measures': 'area: areacello', '...
    m2lines/cli_args:                  /Users/alxmrs/git/ocean_emulators/ocea...
    m2lines/date_created:              2025-12-03T10:10:28.215668
    m2lines/ocean_emulators_git_hash:  https://github.com/m2lines/ocean_emula...
    regrid_method:                     conservative
    so:                                {'cell_measures': 'area: areacello', '...
    tauuo:                             {'cell_methods': 'yh:mean xq:point tim...
    tauvo:                             {'cell_methods': 'yq:point xh:mean tim...
    thetao:                            {'cell_measures': 'area: areacello', '...
    uo:                                {'cell_methods': 'z_l:mean yh:mean xq:...
    vo:                                {'cell_methods': 'z_l:mean yq:point xh...
    zos:                               {'cell_measures': 'area: areacello', '...
>>> # One degree data with _no_ gaussian filtering (no filter).
>>> ds = xr.open_zarr('https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2025-11/om4_onedeg/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 98GB
Dimensions:    (time: 4745, y: 180, x: 360)
Coordinates:
  * time       (time) object 38kB 1958-01-03 12:00:00 ... 2022-12-29 12:00:00
  * x          (x) float64 3kB 0.5 1.5 2.5 3.5 4.5 ... 356.5 357.5 358.5 359.5
  * y          (y) float64 1kB -89.24 -88.25 -87.25 -86.26 ... 87.25 88.25 89.24
Data variables: (12/99)
    hfds       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    mask_0     (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_1     (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_10    (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_11    (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    mask_12    (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    ...         ...
    vo_5       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_6       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_7       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_8       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    vo_9       (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    zos        (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
Attributes:
    m2lines/cli_args:                  /Users/alxmrs/git/ocean_emulators/ocea...
    m2lines/date_created:              2025-11-26T12:51:52.411906
    m2lines/ocean_emulators_git_hash:  https://github.com/m2lines/ocean_emula...
    regrid_method:                     conservative
>>> # Half degree data with no gaussian filtering
>>> ds = xr.open_zarr('https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2025-11/om4_halfdeg/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 394GB
Dimensions:    (time: 4745, y: 360, x: 720)
Coordinates:
  * time       (time) object 38kB 1958-01-03 12:00:00 ... 2022-12-29 12:00:00
  * x          (x) float64 6kB 0.25 0.75 1.25 1.75 ... 358.2 358.8 359.2 359.8
  * y          (y) float64 3kB -89.62 -89.12 -88.62 -88.13 ... 88.62 89.12 89.62
Data variables: (12/99)
    hfds       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    mask_0     (y, x) bool 259kB dask.array<chunksize=(360, 720), meta=np.ndarray>
    mask_1     (y, x) bool 259kB dask.array<chunksize=(360, 720), meta=np.ndarray>
    mask_10    (y, x) bool 259kB dask.array<chunksize=(360, 720), meta=np.ndarray>
    mask_11    (y, x) bool 259kB dask.array<chunksize=(360, 720), meta=np.ndarray>
    mask_12    (y, x) bool 259kB dask.array<chunksize=(360, 720), meta=np.ndarray>
    ...         ...
    vo_5       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    vo_6       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    vo_7       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    vo_8       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    vo_9       (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    zos        (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
Attributes:
    m2lines/cli_args:                  /Users/alxmrs/git/ocean_emulators/ocea...
    m2lines/date_created:              2025-11-26T11:46:51.855769
    m2lines/ocean_emulators_git_hash:  https://github.com/m2lines/ocean_emula...
    regrid_method:                     conservative
>>> # Quarter degree data with no gaussian filtering.
>>> ds = xr.open_zarr('https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2025-11/om4_quarterdeg/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 2TB
Dimensions:    (time: 4745, y: 720, x: 1440)
Coordinates:
  * time       (time) object 38kB 1958-01-03 12:00:00 ... 2022-12-29 12:00:00
  * x          (x) float64 12kB 0.125 0.375 0.625 0.875 ... 359.4 359.6 359.9
  * y          (y) float64 6kB -89.81 -89.56 -89.31 -89.06 ... 89.31 89.56 89.81
Data variables: (12/99)
    hfds       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    mask_0     (y, x) bool 1MB dask.array<chunksize=(720, 1440), meta=np.ndarray>
    mask_1     (y, x) bool 1MB dask.array<chunksize=(720, 1440), meta=np.ndarray>
    mask_10    (y, x) bool 1MB dask.array<chunksize=(720, 1440), meta=np.ndarray>
    mask_11    (y, x) bool 1MB dask.array<chunksize=(720, 1440), meta=np.ndarray>
    mask_12    (y, x) bool 1MB dask.array<chunksize=(720, 1440), meta=np.ndarray>
    ...         ...
    vo_5       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    vo_6       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    vo_7       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    vo_8       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    vo_9       (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    zos        (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
Attributes:
    hfds:                              {'cell_measures': 'area: areacello', '...
    m2lines/cli_args:                  /Users/alxmrs/git/ocean_emulators/ocea...
    m2lines/date_created:              2025-12-01T15:34:44.338655
    m2lines/ocean_emulators_git_hash:  https://github.com/m2lines/ocean_emula...
    regrid_method:                     conservative
    so:                                {'cell_measures': 'area: areacello', '...
    tauuo:                             {'cell_methods': 'yh:mean xq:point tim...
    tauvo:                             {'cell_methods': 'yq:point xh:mean tim...
    thetao:                            {'cell_measures': 'area: areacello', '...
    uo:                                {'cell_methods': 'z_l:mean yh:mean xq:...
    vo:                                {'cell_methods': 'z_l:mean yq:point xh...
    zos:                               {'cell_measures': 'area: areacello', '...
```

## Snapshot datasets

The datasets above store the ocean state as 5-day **averages**. We also provide **snapshot**
datasets, where the prognostic state variables (`thetao`, `so`, `uo`, `vo`, `zos`) are
*instantaneous* values sampled at 00:00 UTC every 5 days, rather than 5-day means. The
forcing variables (`hfds`, `tauuo`, `tauvo`, `wfo`) remain 5-day means, since the ocean
integrates these fluxes in time (a snapshot of the forcing would not conserve the heat,
momentum, and freshwater taken up between states).

These datasets also carry the grid-metadata coordinates needed for physical analysis:
`areacello` (geometric cell area), `ocean_fraction` (the per-level land-aware wet fraction;
use `areacello * ocean_fraction` for area/volume weighting), the cell-corner bounds
`lon_b`/`lat_b`, and the depth axis `lev`/`dz`. They are available at 1°, 1/2°, and 1/4°
(regridded to a regular grid) and on the native 1/4° tripolar grid (no regridding).

```shell
>>> import xarray as xr
>>> # 1 degree snapshots (no gaussian filtering)
>>> ds = xr.open_zarr('https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2026-06/om4_onedeg_snapshots/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 100GB
Dimensions:         (time: 4745, y: 180, x: 360, lev: 19, y_b: 181, x_b: 361)
Coordinates:
  * time            (time) object 38kB 1958-01-06 00:00:00 ... 2023-01-01 00:...
  * y               (y) float64 1kB -89.24 -88.25 -87.25 ... 87.25 88.25 89.24
  * x               (x) float64 3kB 0.5 1.5 2.5 3.5 ... 356.5 357.5 358.5 359.5
    areacello       (y, x) float64 518kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    lat             (y, x) float64 518kB dask.array<chunksize=(90, 360), meta=np.ndarray>
    lon             (y, x) float64 518kB dask.array<chunksize=(90, 360), meta=np.ndarray>
    ocean_fraction  (lev, y, x) float64 10MB dask.array<chunksize=(19, 180, 360), meta=np.ndarray>
  * lev             (lev) float64 152B 2.5 10.0 22.5 40.0 ... 4e+03 5e+03 6e+03
    dz              (lev) float64 152B dask.array<chunksize=(19,), meta=np.ndarray>
    lat_b           (y_b, x_b) float64 523kB dask.array<chunksize=(91, 361), meta=np.ndarray>
    lon_b           (y_b, x_b) float64 523kB dask.array<chunksize=(91, 361), meta=np.ndarray>
Dimensions without coordinates: y_b, x_b
Data variables: (12/100)
    hfds            (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    mask_0          (y, x) bool 65kB dask.array<chunksize=(180, 360), meta=np.ndarray>
    ...              ...
    wfo             (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
    zos             (time, y, x) float32 1GB dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
Attributes: (12/13)
    hfds:                              {'cell_measures': 'area: areacello', '...
    m2lines/cli_args:                  /Users/alxmrs/git/Ocean_Emulator/data/...
    m2lines/date_created:              2026-06-04T17:41:39.340848
    m2lines/ocean_emulators_git_hash:  https://github.com/Open-Athena/Ocean_E...
    regrid_method:                     conservative
    wfo:                               {'cell_measures': 'area: areacello', '...
>>> # 1/2 degree snapshots
>>> ds = xr.open_zarr('https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2026-06/om4_halfdeg_snapshots/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 399GB
Dimensions:         (time: 4745, y: 360, x: 720, lev: 19, y_b: 361, x_b: 721)
Coordinates:
  * time            (time) object 38kB 1958-01-06 00:00:00 ... 2023-01-01 00:...
  * y               (y) float64 3kB -89.62 -89.12 -88.62 ... 88.62 89.12 89.62
  * x               (x) float64 6kB 0.25 0.75 1.25 1.75 ... 358.8 359.2 359.8
    areacello       (y, x) float64 2MB dask.array<chunksize=(360, 720), meta=np.ndarray>
    ocean_fraction  (lev, y, x) float64 39MB dask.array<chunksize=(19, 360, 720), meta=np.ndarray>
  * lev             (lev) float64 152B 2.5 10.0 22.5 40.0 ... 4e+03 5e+03 6e+03
    lon_b           (y_b, x_b) float64 2MB dask.array<chunksize=(91, 361), meta=np.ndarray>
Dimensions without coordinates: y_b, x_b
Data variables: (12/100)
    hfds            (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
    ...              ...
    zos             (time, y, x) float32 5GB dask.array<chunksize=(1, 360, 720), meta=np.ndarray>
>>> # 1/4 degree snapshots
>>> ds = xr.open_zarr('https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2026-06/om4_quarterdeg_snapshots/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 2TB
Dimensions:         (time: 4745, y: 720, x: 1440, lev: 19, y_b: 721, x_b: 1441)
Coordinates:
  * time            (time) object 38kB 1958-01-06 00:00:00 ... 2023-01-01 00:...
  * y               (y) float64 6kB -89.81 -89.56 -89.31 ... 89.31 89.56 89.81
  * x               (x) float64 12kB 0.125 0.375 0.625 ... 359.4 359.6 359.9
    areacello       (y, x) float64 8MB dask.array<chunksize=(720, 1440), meta=np.ndarray>
    ocean_fraction  (lev, y, x) float64 158MB dask.array<chunksize=(19, 720, 1440), meta=np.ndarray>
  * lev             (lev) float64 152B 2.5 10.0 22.5 40.0 ... 4e+03 5e+03 6e+03
Dimensions without coordinates: y_b, x_b
Data variables: (12/100)
    hfds            (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
    ...              ...
    zos             (time, y, x) float32 20GB dask.array<chunksize=(1, 720, 1440), meta=np.ndarray>
>>> # Native 1/4 degree tripolar grid snapshots (no regridding; x/y are curvilinear indices,
>>> # with 2D lon/lat coordinates giving the geographic location of each cell)
>>> ds = xr.open_zarr('https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2026-06/om4_tripolar_snapshots/OM4.zarr')
>>> ds
<xarray.Dataset> Size: 2TB
Dimensions:         (time: 4745, y: 1080, x: 1440, lev: 19, y_b: 1081, x_b: 1441)
Coordinates:
  * time            (time) object 38kB 1958-01-06 00:00:00 ... 2023-01-01 00:...
  * y               (y) float64 9kB -80.39 -80.31 -80.23 ... 89.73 89.84 89.95
  * x               (x) float64 12kB -299.7 -299.5 -299.2 ... 59.53 59.78 60.03
    areacello       (y, x) float64 12MB dask.array<chunksize=(1080, 1440), meta=np.ndarray>
    lat             (y, x) float64 12MB dask.array<chunksize=(271, 361), meta=np.ndarray>
    lon             (y, x) float64 12MB dask.array<chunksize=(271, 361), meta=np.ndarray>
    ocean_fraction  (lev, y, x) float64 236MB dask.array<chunksize=(19, 1080, 1440), meta=np.ndarray>
  * lev             (lev) float64 152B 2.5 10.0 22.5 40.0 ... 4e+03 5e+03 6e+03
    lat_b           (y_b, x_b) float64 12MB dask.array<chunksize=(271, 361), meta=np.ndarray>
    lon_b           (y_b, x_b) float64 12MB dask.array<chunksize=(271, 361), meta=np.ndarray>
Dimensions without coordinates: y_b, x_b
Data variables: (12/100)
    hfds            (time, y, x) float32 30GB dask.array<chunksize=(1, 1080, 1440), meta=np.ndarray>
    ...              ...
    zos             (time, y, x) float32 30GB dask.array<chunksize=(1, 1080, 1440), meta=np.ndarray>
```

Each snapshot store ships with companion `OM4_means.zarr` and `OM4_stds.zarr` normalization
datasets in the same directory (see the normalization script below).

## How to get the data

We recommend two methods for acquiring the dataset locally.

1. If you would like to copy a small slice of the data across time, please use our cloning script:

   ```shell
   $ # last updated: 2026-04-28
   $ uv run scripts/clone_data.py --help
   #    usage: clone_data [-h] [--source SOURCE] [--time_start TIME_START] [--time_end TIME_END] [--write_time_chunks WRITE_TIME_CHUNKS] [--compact_variables] [--local_cluster] dest
   #
   # Make a copy of the OM4 dataset (~100 GiBs - ~2 TiBs).
   #
   # positional arguments:
   #   dest                  Root directory for the copy of datasets.
   #
   # options:
   #   -h, --help            show this help message and exit
   #   --source SOURCE       Alternative source root directory to copy data from. Defaults to 1° OM4 without Gaussian filtering.
   #   --time_start TIME_START
   #                         start index for data.isel() along time dimension.
   #   --time_end TIME_END   end index for data.isel() along time dimension.
   #   --write_time_chunks WRITE_TIME_CHUNKS
   #                         The number of chunks to write in each time dimension. Default=1.
   #   --compact_variables   Turn on a 'compact' data representation. This is now Zarr is more traditionally stored, but is sub-optimal in our data loader.
   #   --local_cluster       Run pipeline on a local dask cluster. This should be faster due to multi-processing.

   $ # Get 10 time steps worth of data to your local directory ./data_cache/
   $ uv run scripts/clone_data.py --time_end 10 ./data_cache/om4_onedeg_10/
   # Copies OM4.zarr, OM4_means.zarr and OM4_stds.zarr datasets into ./data_cache/om4_onedeg_10/
   ```

2. If you would like to get whole datasets for training, we recommend copying them with [`rclone`](https://rclone.org/):

   ```shell
    rclone config create nyu-osn-public s3 provider=Other anonymous=true endpoint=https://nyu1.osn.mghpcc.org/
    rclone copy --progress nyu-osn-public:m2lines-pubs/FOMO/v2025-11/om4_onedeg ./data_cache/ --ignore-existing --transfers=32
   ```

Of course, you are more than welcome to access our Zarr datasets directly from cloud storage without downloading as they
are in a cloud native data format.

## How was each dataset produced? How can I reproduce the data?

The code for pre-processing our datasets and the raw sources of data are all made available in this project. To access
our pre-processing code, please see the [`data/`](../data/) subproject README.

Each of the above datasets includes metadata that explains the exact process used to engineer it. To check the command
used to produce the data, lookup the following attributes:
```python
ds = ...
ds.attrs["m2lines/cli_args"]                  # Exact arguments used to process this dataset
ds.attrs["m2lines/ocean_emulators_git_hash"]  # The version of code of the pre-processing codebase, now rebased into data/
```

All of our data engineering (for the v2025-11 datasets) was tracked in this GitHub issue: [#450](https://github.com/Open-Athena/Ocean_Emulator/issues/450).
Below are updated versions of these data engineering scripts that make use of the publically available data.

For each script, we use coiled to manage our dask clusters. To get set up with a coiled cluster, please do the following:

1. Make sure coiled in installed. If you've installed all the dev dependencies, then it should be in your local env.
   ```shell
   uv pip install coiled
   ```

2. Log in to coiled. This may require creating an account (recommended: sign in with a Google or Github SSO).
   ```shell
   coiled login
   ```

3. Connect to a cloud provider. Here are a few commands to set up the top three cloud providers:
   ```shell
   coiled setup aws
   coiled setup gcp
   coiled setup azure
   ```

**1° OM4**

```shell
deactivate
mamba activate ocean_preprocessing
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
# These keys are only needed to _write_ to the bucket.
# Check with M2LInES project management for how to get the OSN Access keys.
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
python -m ocean_preprocessing om4 \
   "s3://m2lines-pubs/FOMO/raw/om4_5daily.zarr" \
   "s3://m2lines-pubs/FOMO/raw/ocean_static_no_mask_table.zarr" \
   "s3://m2lines-pubs/FOMO/raw/grids/ocean_hgrid.zarr" \
   "s3://m2lines-pubs/FOMO/raw/grids/gaussian_grid_180_by_360.zarr" \
    --output_path="s3://m2lines-pubs/FOMO/v$(date "+%Y-%m")/om4_ondeg/OM4.zarr" \
    --skip_spatial_filtering \
    --skip_validation \
    --cluster="coiled" \
    --n_workers=40 \
    --wait_for_workers=True
```

**1° filtered OM4**

```shell
deactivate
mamba activate ocean_preprocessing
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
# These keys are only needed to _write_ to the bucket.
# Check with M2LInES project management for how to get the OSN Access keys.
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
python -m ocean_preprocessing om4 \
   "s3://m2lines-pubs/FOMO/raw/om4_5daily.zarr" \
   "s3://m2lines-pubs/FOMO/raw/ocean_static_no_mask_table.zarr" \
   "s3://m2lines-pubs/FOMO/raw/grids/ocean_hgrid.zarr" \
   "s3://m2lines-pubs/FOMO/raw/grids/gaussian_grid_180_by_360.zarr" \
    --output_path="s3://m2lines-pubs/FOMO/v$(date '+%Y-%m')/om4_onedeg_filter/OM4.zarr" \
    --skip_validation \
    --cluster="coiled" \
    --n_workers=40 \
    --wait_for_workers=True
```

**1/2° OM4**
```shell
deactivate
mamba activate ocean_preprocessing
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
# These keys are only needed to _write_ to the bucket.
# Check with M2LInES project management for how to get the OSN Access keys.
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
python -m ocean_preprocessing om4 \
   "s3://m2lines-pubs/FOMO/raw/om4_5daily.zarr" \
   "s3://m2lines-pubs/FOMO/raw/ocean_static_no_mask_table.zarr" \
   "s3://m2lines-pubs/FOMO/raw/grids/ocean_hgrid.zarr" \
   "s3://m2lines-pubs/FOMO/raw/grids/gaussian_grid_360_by_720.zarr" \
    --output_path="s3://m2lines-pubs/FOMO/v$(date '+%Y-%m')/om4_halfdeg/OM4.zarr" \
    --skip_spatial_filtering \
    --skip_validation \
    --cluster="coiled" \
    --n_workers=40 \
    --wait_for_workers=True
```

**1/4° OM4**
```shell
deactivate
mamba activate ocean_preprocessing
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
# These keys are only needed to _write_ to the bucket.
# Check with M2LInES project management for how to get the OSN Access keys.
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
python -m ocean_preprocessing om4 \
   "s3://m2lines-pubs/FOMO/raw/om4_5daily.zarr" \
   "s3://m2lines-pubs/FOMO/raw/ocean_static_no_mask_table.zarr" \
   "s3://m2lines-pubs/FOMO/raw/grids/ocean_hgrid.zarr" \
   "s3://m2lines-pubs/FOMO/raw/grids/gaussian_grid_720_by_1440.zarr" \
    --output_path="s3://m2lines-pubs/FOMO/v$(date '+%Y-%m')/om4_quarterdeg/OM4.zarr" \
    --skip_spatial_filtering \
    --skip_validation \
    --cluster="coiled" \
    --n_workers=40 \
    --wait_for_workers=True
```

**OM4 snapshots**

The snapshot datasets are produced with the same `om4` pipeline, pointed at the 5-daily
*snapshot* source (instantaneous state at 00:00 UTC every 5 days, with 5-day-mean forcings)
instead of the averaged source. The state variables come straight from the simulation on the
native 1/4° tripolar grid — ask M2LInES project management for access to the snapshot source
store. One run per output grid:

```shell
deactivate
mamba activate ocean_preprocessing
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
# These keys are only needed to _write_ to the bucket.
# Check with M2LInES project management for how to get the OSN Access keys.
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
# 1° snapshots (see below for the 1/2°, 1/4°, and native tripolar variants)
python -m ocean_preprocessing om4 \
   "s3://emulators/wg2437/1958-2022.OM4p25_5daily_snapshots" \
   "s3://m2lines-pubs/FOMO/raw/ocean_static_no_mask_table.zarr" \
   "s3://m2lines-pubs/FOMO/raw/grids/ocean_hgrid.zarr" \
   "s3://m2lines-pubs/FOMO/raw/grids/gaussian_grid_180_by_360.zarr" \
    --output_path="s3://m2lines-pubs/FOMO/v$(date '+%Y-%m')/om4_onedeg_snapshots/OM4.zarr" \
    --skip_spatial_filtering \
    --skip_validation \
    --cluster="coiled" \
    --n_workers=40 \
    --wait_for_workers=True
```

The other resolutions use the same command with a different target grid and output name:

- **1/2°**: target `gaussian_grid_360_by_720.zarr` → `om4_halfdeg_snapshots`
- **1/4°**: target `gaussian_grid_720_by_1440.zarr` → `om4_quarterdeg_snapshots`
- **native tripolar**: add `--skip_regridding` and write to `om4_tripolar_snapshots`. The
  target-grid argument is unused when regridding is skipped but is still required
  positionally, so pass any grid (e.g. `gaussian_grid_720_by_1440.zarr`).

After each of these datasets are computed, one can produce complementary mean and std datasets by running the following
script:

```shell
export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
# These keys are only needed to _write_ to the bucket.
# Check with M2LInES project management for how to get the OSN Access keys.
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
python ocean_preprocessing/make_norm_datasets.py "s3://m2lines-pubs/FOMO/v$(date "+%Y-%m")/om4_ondeg/OM4.zarr"
# This creates OM4_means.zarr and OM4_stds.zarr in the same directory as the input OM4.zarr.
```

This script is designed to run on a large VM in the cloud. It is simple to modify the script to point to a coiled cluster
or other dask cluster system.