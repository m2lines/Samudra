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