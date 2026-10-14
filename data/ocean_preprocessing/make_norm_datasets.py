#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Create *_means.zarr and *_stds.zarr from a dataset."""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from dask.distributed import Client


def write_zarr_with_retries(
    ds: xr.Dataset, path: str, client: Client, write_retries: int
) -> None:
    """Write a derived dataset, retrying failed Dask tasks in place."""
    delayed = ds.to_zarr(
        path,
        mode="w",
        zarr_format=2,
        consolidated=True,
        compute=False,
    )
    client.compute(delayed, retries=write_retries, sync=True)


def normalization_fields(ds: xr.Dataset) -> xr.Dataset:
    """Mask flat or depth-stacked ocean fields without reducing their depth axis."""
    fields = {}
    for name, field in ds.data_vars.items():
        if "time" not in field.dims:
            continue
        if "lev" in field.dims:
            mask = ds["wetmask"]
        else:
            _, _, suffix = str(name).rpartition("_")
            level = int(suffix) if suffix.isdigit() else 0
            if "wetmask" in ds:
                mask = ds.wetmask.isel(lev=level, drop=True)
            else:
                mask = ds[f"mask_{level}"]
        fields[name] = field.where(mask > 0)
    return xr.Dataset(fields, attrs=ds.attrs)


def main(
    src: str,
    write_retries: int = 5,
    *,
    start: str | None = None,
    end: str | None = None,
) -> None:
    """Calculate and write normalization statistics for ``src``."""
    from dask.distributed import LocalCluster

    if write_retries < 0:
        raise ValueError("write_retries must be non-negative")

    client = LocalCluster().get_client()
    # This could be the OSN pod, it just has to be set via environment variable like so:
    # ```
    # export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
    # export AWS_ACCESS_KEY_ID=...
    # export AWS_SECRET_ACCESS_KEY=...
    # ```
    path, base = os.path.dirname(src), os.path.basename(src)
    ds_name, ds_ext = os.path.splitext(base)

    print(f"Reading Zarr dataset at {src=}.")
    ds = xr.open_zarr(src, chunks={})

    if start is not None or end is not None:
        ds = ds.sel(time=slice(start, end))
    if not ds.sizes.get("time", 0):
        raise ValueError("Normalization interval selects no timestamps")
    ds = normalization_fields(ds)
    dims = [dim for dim in ("time", "y", "x") if dim in ds.dims]
    print("Computing mean and std over wet space and time, retaining depth.")
    ds_means = ds.mean(dims, skipna=True)
    ds_stds = ds.std(dims, skipna=True)
    attrs = {
        "normalization_source": src,
        "normalization_start": str(ds.time.values[0]),
        "normalization_end": str(ds.time.values[-1]),
        "normalization_steps": ds.sizes["time"],
        "normalization_method": "population mean/std over finite wet-ocean samples",
    }
    ds_means.attrs.update(attrs)
    ds_stds.attrs.update(attrs)

    mean_path = os.path.join(path, f"{ds_name}_means{ds_ext}")
    print(f"Writing mean Zarr to {mean_path!r}.")
    # These stores are derived entirely from ``src``. Replace a partial store
    # left by an interrupted/requeued attempt so that restarting is safe.
    write_zarr_with_retries(ds_means, mean_path, client, write_retries)
    stds_path = os.path.join(path, f"{ds_name}_stds{ds_ext}")
    print(f"Writing std Zarr to {stds_path!r}.")
    write_zarr_with_retries(ds_stds, stds_path, client, write_retries)

    print("done.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="local or s3:// path to the source Zarr dataset")
    parser.add_argument(
        "--write-retries",
        "--write_retries",
        type=int,
        default=5,
        help="number of Dask retries for each failed write task (default: 5)",
    )
    parser.add_argument(
        "--start", help="First training date to include (source calendar)"
    )
    parser.add_argument("--end", help="Last training date to include (source calendar)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.src, args.write_retries, start=args.start, end=args.end)
