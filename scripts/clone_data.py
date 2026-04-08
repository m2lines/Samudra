# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Script to clone and compact Zarr ocean data."""
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "xarray[io]",
#   "zarr<3",
#   "dask",
#   "requests",
#   "aiohttp",
#   "gcsfs",
#   "numcodecs>=0.15",
#   "distributed",
#   "tenacity",
#   "ocean-emulators",
# ]
#
# [tool.uv.sources]
# ocean-emulators = { path = "../" }
# ///

import argparse
import os
import pathlib

import dask
import dask.diagnostics
import xarray as xr
from dask.distributed import LocalCluster
from tenacity import retry

from ocean_emulators.utils.data import compact_dataset

DEFAULT_DATA_ROOT = "https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2025-11/om4_onedeg/"


@retry
def robust_open_dataset(target: str, **kwargs) -> xr.Dataset:
    return xr.open_dataset(target, **kwargs)


def main(args: argparse.Namespace) -> None:
    """Clones slice of Samudra data at the `dest_root` directory."""
    if args.local_cluster:
        cluster = LocalCluster()
        client = cluster.get_client()  # noqa: F841

    # Ensure the path/to/dest exists
    if not args.dest.startswith("gs://") and not args.dest.startswith("s3://"):
        pathlib.Path(args.dest).mkdir(parents=True, exist_ok=True)

    time_slice = slice(args.time_start, args.time_end)
    output_chunks = dict(time=args.write_time_chunks, lev=19)

    for name, dest_fmt in [
        ("OM4", "zarr"),
        ("OM4_means", "zarr"),
        ("OM4_stds", "zarr"),
    ]:
        dest = os.path.join(args.dest, name)
        source = (args.source or DEFAULT_DATA_ROOT) + name

        # Open Xarray Datasets with retries + exponential backoff.
        if name == "OM4":
            data = robust_open_dataset(source, engine="zarr", chunks={"time": 700})
            data = data.isel(time=time_slice)
        else:
            data = robust_open_dataset(source, engine="zarr", chunks={})

        if args.compact_variables:
            data = compact_dataset(data)

        with dask.diagnostics.ProgressBar():
            if dest_fmt.lower() == "zarr":
                if name == "OM4":
                    data = data.chunk(output_chunks)
                data.to_zarr(dest + ".zarr")
            else:
                data.to_netcdf(dest + ".nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "clone_data",
        description="Make a copy of the OM4 dataset (~100 GiBs - ~2 TiBs).",
    )
    parser.add_argument(
        "dest", type=str, help="Root directory for the copy of datasets."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="Alternative source root directory to copy data from. Defaults to 1° OM4 without Gaussian filtering.",
    )
    parser.add_argument(
        "--time_start",
        type=int,
        default=0,
        help="start index for data.isel() along time dimension.",
    )
    parser.add_argument(
        "--time_end",
        type=int,
        default=None,
        help="end index for data.isel() along time dimension.",
    )
    parser.add_argument(
        "--write_time_chunks",
        type=int,
        default=1,
        help="The number of chunks to write in each time dimension. Default=1.",
    )
    parser.add_argument(
        "--compact_variables",
        action="store_true",
        help="Turn on a 'compact' data representation. This is now Zarr is more traditionally stored, but is sub-optimal in our data loader.",
    )
    parser.add_argument(
        "--local_cluster",
        action="store_true",
        help="Run pipeline on a local dask cluster. This should be faster due to multi-processing.",
    )

    main(parser.parse_args())
