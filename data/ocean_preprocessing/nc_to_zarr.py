#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""A quick utility script to convert NetCDF files from the OSN pod to Zarr."""

import os
import sys

import fsspec
import xarray as xr

if __name__ == "__main__":
    fs_src = fsspec.filesystem(
        "s3",
        key=os.environ["OSN_BUCKET_ACCESS_KEY"],
        secret=os.environ["OSN_BUCKET_SECRET_KEY"],
        endpoint_url="https://nyu1.osn.mghpcc.org/",
    )
    # This could also be the OSN pod, it just has to be set via environment variable like so:
    # ```
    # export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
    # export AWS_ACCESS_KEY_ID=...
    # export AWS_SECRET_ACCESS_KEY=...
    # ```
    fs_dst = fsspec.filesystem("s3")

    src, dst = sys.argv[1:3]
    assert src.startswith("s3://"), f"{src=} must start with `s3://`"
    assert dst.startswith("s3://"), f"{dst=} must start with `s3://`"

    fs_dst.mkdirs(os.path.dirname(dst), exist_ok=True)
    zstore = fs_dst.get_mapper(dst)

    with fs_src.open(src) as f:
        ds = xr.open_dataset(f)
        ds.to_zarr(zstore, zarr_format=2, consolidated=True)
