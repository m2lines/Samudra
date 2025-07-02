"""Script to clone remote Oceans Emulator data locally."""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray[io]",
#   "zarr>=3",
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

DATA_ROOT = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/"


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
        source = DATA_ROOT + name

        # Open Xarray Datasets with retries + exponential backoff.
        if name == "OM4":
            data = robust_open_dataset(source, engine="zarr", chunks={"time": 700})
            data = data.isel(time=time_slice)
        else:
            data = robust_open_dataset(source, engine="zarr", chunks={})

        if args.compact_variables:
            data = compact_dataset(data)

        # Turn off compression
        encoding = {var: {"compressors": None} for var in data.data_vars.keys()}

        with dask.diagnostics.ProgressBar():
            if dest_fmt.lower() == "zarr":
                if name == "OM4":
                    data = data.chunk(output_chunks)
                data.to_zarr(dest + ".zarr", encoding=encoding, zarr_format=3)
            else:
                data.to_netcdf(dest + ".nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "clone_data", description="Make a copy of the Samudra dataset (~70 GiBs)."
    )
    parser.add_argument("dest", type=str, help="Root directory for copy of datasets.")
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
    parser.add_argument("--write_time_chunks", type=int, default=1)
    parser.add_argument("--compact_variables", action="store_true")
    parser.add_argument("--local_cluster", action="store_true")

    main(parser.parse_args())
