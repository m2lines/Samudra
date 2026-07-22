<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

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

All of our data engineering (for the v2025-11 datasets) was tracked in this GitHub issue: [#450](https://github.com/m2lines/Samudra/issues/450).
Below are updated versions of these data engineering scripts that make use of the publically available data.

There are two ways to run the pipeline:

- **[NYU Torch HPC (CPU, local Dask)](#reproducing-on-nyu-torch-hpc-cpu-no-cloud-egress)** — *recommended.* Runs on a CPU
  node whose network is local to the NYU OSN pod, so streaming raw data in and writing processed Zarr back incurs no
  cloud egress cost.
- **[Coiled (cloud)](#coiled-cloud)** — convenient managed Dask clusters, but the cluster lives in a cloud region, so
  moving the multi-TB datasets to/from OSN incurs egress charges.

### Coiled (cloud)

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
   "s3://m2lines-pubs/Samudra/raw/om4_5daily.zarr" \
   "s3://m2lines-pubs/Samudra/raw/ocean_static_no_mask_table.zarr" \
   "s3://m2lines-pubs/Samudra/raw/grids/ocean_hgrid.zarr" \
   "s3://m2lines-pubs/Samudra/raw/grids/gaussian_grid_180_by_360.zarr" \
    --output_path="s3://m2lines-pubs/Samudra/v$(date "+%Y-%m")/om4_onedeg/OM4.zarr" \
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
   "s3://m2lines-pubs/Samudra/raw/om4_5daily.zarr" \
   "s3://m2lines-pubs/Samudra/raw/ocean_static_no_mask_table.zarr" \
   "s3://m2lines-pubs/Samudra/raw/grids/ocean_hgrid.zarr" \
   "s3://m2lines-pubs/Samudra/raw/grids/gaussian_grid_180_by_360.zarr" \
    --output_path="s3://m2lines-pubs/Samudra/v$(date '+%Y-%m')/om4_onedeg_filter/OM4.zarr" \
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
   "s3://m2lines-pubs/Samudra/raw/om4_5daily.zarr" \
   "s3://m2lines-pubs/Samudra/raw/ocean_static_no_mask_table.zarr" \
   "s3://m2lines-pubs/Samudra/raw/grids/ocean_hgrid.zarr" \
   "s3://m2lines-pubs/Samudra/raw/grids/gaussian_grid_360_by_720.zarr" \
    --output_path="s3://m2lines-pubs/Samudra/v$(date '+%Y-%m')/om4_halfdeg/OM4.zarr" \
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
   "s3://m2lines-pubs/Samudra/raw/om4_5daily.zarr" \
   "s3://m2lines-pubs/Samudra/raw/ocean_static_no_mask_table.zarr" \
   "s3://m2lines-pubs/Samudra/raw/grids/ocean_hgrid.zarr" \
   "s3://m2lines-pubs/Samudra/raw/grids/gaussian_grid_720_by_1440.zarr" \
    --output_path="s3://m2lines-pubs/Samudra/v$(date '+%Y-%m')/om4_quarterdeg/OM4.zarr" \
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
python ocean_preprocessing/make_norm_datasets.py "s3://m2lines-pubs/Samudra/v$(date "+%Y-%m")/om4_onedeg/OM4.zarr"
# This creates OM4_means.zarr and OM4_stds.zarr in the same directory as the input OM4.zarr.
```

This script is designed to run on a large VM in the cloud. It is simple to modify the script to point to a coiled cluster
or other dask cluster system.

## Reproducing on NYU Torch HPC (CPU, no cloud egress)

This is the recommended way to regenerate the datasets. It runs the exact same `ocean_preprocessing` pipeline, but on a
Torch CPU node with a **local Dask cluster** (`--cluster=local`) instead of coiled. Because Torch's network is local to
the NYU OSN pod, reading the raw OM4 data and writing the processed Zarr back stays on the internal network — there is no
cloud egress charge.

Two Slurm harness scripts drive this:

- [`scripts/slurm_preprocess_om4.sbatch`](../scripts/slurm_preprocess_om4.sbatch) — runs the OM4 pipeline for one
  resolution.
- [`scripts/slurm_make_norm_om4.sbatch`](../scripts/slurm_make_norm_om4.sbatch) — builds `OM4_means.zarr` /
  `OM4_stds.zarr` alongside a processed `OM4.zarr` (used by the metric suite).

### One-time environment setup on `/scratch`

The pipeline needs the conda-only `ocean_preprocessing` env (xESMF is not on PyPI). Install
[miniforge](https://github.com/conda-forge/miniforge) on `/scratch` once, then create the env from the checked-in
`data/mamba_env.yaml`:

```bash
# On a Torch login node:
cd /scratch/$USER
curl -L -o miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash miniforge.sh -b -p /scratch/$USER/miniforge3
source /scratch/$USER/miniforge3/etc/profile.d/conda.sh

# The '-e .' in the env file resolves relative to the current dir, so create from data/:
cd /scratch/$USER/Samudra/data      # your repo checkout on scratch
mamba env create -f mamba_env.yaml  # creates env 'ocean_preprocessing'
```

The harness autodetects conda from `$CONDA_EXE`, `~/miniforge3`, or `/scratch/$USER/miniforge3`; otherwise set
`CONDA_SH=<miniforge>/etc/profile.d/conda.sh`.

### Credentials

```bash
export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
# Write keys for the OSN pod. Check with M2LInES project management to obtain them.
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### Submitting the jobs

Both the raw **inputs** and the processed **outputs** live under the `Samudra` prefix on the OSN pod: inputs are read
from `s3://m2lines-pubs/Samudra/raw/...` (`RAW_ROOT`) and results are written under `OUTPUT_BASE`. The raw inputs were
mirrored there from `FOMO/raw` with a server-side `rclone` copy on the data transfer node.

```bash
export OUTPUT_BASE="s3://m2lines-pubs/Samudra/v$(date +%Y-%m)"

# Smoke test first: reads real OSN data, writes nothing (--dry_run) on 10 steps.
RESOLUTION=onedeg EXTRA_ARGS="--small_run --dry_run" \
  sbatch --cpus-per-task=16 --mem=64G --time=00:30:00 scripts/slurm_preprocess_om4.sbatch

# Then each dataset. All fit on the `cs` partition (128-core, 513 GB) because
# the pipeline is chunked one timestep at a time.
RESOLUTION=twodeg        sbatch --cpus-per-task=32  --mem=240G --time=08:00:00   scripts/slurm_preprocess_om4.sbatch
RESOLUTION=onedeg        sbatch --cpus-per-task=64  --mem=480G --time=12:00:00   scripts/slurm_preprocess_om4.sbatch
RESOLUTION=onedeg_filter sbatch --cpus-per-task=64  --mem=480G --time=12:00:00   scripts/slurm_preprocess_om4.sbatch
RESOLUTION=halfdeg       sbatch --cpus-per-task=128 --mem=490G --time=1-00:00:00 scripts/slurm_preprocess_om4.sbatch
RESOLUTION=quarterdeg N_WORKERS=16 \
                         sbatch --cpus-per-task=128 --mem=490G --time=2-00:00:00 scripts/slurm_preprocess_om4.sbatch

# Normalization datasets, after each OM4.zarr lands:
for R in twodeg onedeg onedeg_filter halfdeg quarterdeg; do
  RESOLUTION=$R sbatch scripts/slurm_make_norm_om4.sbatch
done
```

`RESOLUTION` selects the target grid and whether spatial filtering is applied:

| `RESOLUTION`    | target grid                 | spatial filter | output subdir      |
| --------------- | --------------------------- | -------------- | ------------------ |
| `twodeg`        | `gaussian_grid_90_by_180`   | skipped        | `om4_twodeg`        |
| `twodeg_filter` | `gaussian_grid_90_by_180`   | on (scale 36)  | `om4_twodeg_filter` |
| `onedeg`        | `gaussian_grid_180_by_360`  | skipped        | `om4_onedeg`        |
| `onedeg_filter` | `gaussian_grid_180_by_360`  | on (scale 18)  | `om4_onedeg_filter` |
| `halfdeg`       | `gaussian_grid_360_by_720`  | skipped        | `om4_halfdeg`       |
| `quarterdeg`    | `gaussian_grid_720_by_1440` | skipped        | `om4_quarterdeg`    |

### Partitions

The harness defaults to `--account=torch_pr_347_lzanna --partition=cs`. `cs` nodes (128-core, 513 GB, 184 of them) start
promptly and handle every resolution. The big-memory `cl` partition (7 nodes, 3 TB) is a **fallback only** for the 0.25°
run if it hits memory pressure — it is heavily contended and can queue for days, so prefer capping `N_WORKERS` on `cs`
first.

### Monitoring

```bash
squeue --me -o '%.10i %.14j %.2t %.10M %.20R'
tail -f slurm-<jobid>.out
```

