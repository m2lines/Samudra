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

    print("Apply masks to source dataset.")
    # Apply masks to variables to exclude land values (set to 0) before computing statistics
    # This ensures mean/std calculations only consider ocean points
    ds_masked = ds.copy()

    # Variables that should not be masked (masks themselves, indices, static variables)
    skip_vars = set()
    for var in ds.data_vars:
        if (
            var.startswith("mask_")
            or var.startswith("idepth_")
            or var in ["sea_surface_fraction", "hfgeou"]
        ):
            skip_vars.add(var)

    for var in ds.data_vars:
        if var in skip_vars:
            continue

        # Extract depth level from variable name if it exists (e.g., "so_0" -> 0)
        if "_" in var:
            parts = var.rsplit("_", 1)
            if parts[1].isdigit():
                level = int(parts[1])
                mask_name = f"mask_{level}"

                if mask_name in ds:
                    # Apply the mask: set land values to 0
                    ds_masked[var] = ds[var].where(ds[mask_name] > 0, 0)
                    continue

        # For 2D ocean variables without depth index, use surface mask (mask_0)
        if "mask_0" in ds and set(ds[var].dims) == {"time", "y", "x"}:
            ds_masked[var] = ds[var].where(ds["mask_0"] > 0, 0)

    ds = ds_masked

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
