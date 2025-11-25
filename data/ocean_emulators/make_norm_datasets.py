#!/usr/bin/env python3
"""A quick utility script to create *_means.zarr and *_stds.zarr from a dataset."""

import os
import fsspec
import xarray as xr
import sys
from dask.distributed import LocalCluster

if __name__ == "__main__":
    client = LocalCluster().get_client()
    # This could be the OSN pod, it just has to be set via environment variable like so:
    # ```
    # export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
    # export AWS_ACCESS_KEY_ID=...
    # export AWS_SECRET_ACCESS_KEY=...
    # ```
    fs_src = fsspec.filesystem("s3")

    src = sys.argv[1]
    assert src.startswith("s3://"), f"{src=} must start with `s3://`"

    path, base = os.path.dirname(src), os.path.basename(src)
    ds_name, ds_ext = os.path.splitext(base)

    print(f"Reading Zarr dataset at {src=}.")
    ds = xr.open_zarr(src, chunks={})

    print("Computing mean and std across space and time.")
    ds_means = ds.mean()
    ds_stds = ds.std()

    mean_path = os.path.join(path, f"{ds_name}_means{ds_ext}")
    print(f"Writing mean Zarr to {mean_path!r}.")
    ds_means.to_zarr(mean_path, zarr_format=2, consolidated=True)
    stds_path = os.path.join(path, f"{ds_name}_stds{ds_ext}")
    print(f"Writing std Zarr to {stds_path!r}.")
    ds_stds.to_zarr(stds_path, zarr_format=2, consolidated=True)

    print("done.")
