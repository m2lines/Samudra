"""Script to clone remote Samudra data locally."""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray[io]",
#   "zarr<3",
#   "dask",
#   "requests",
#   "aiohttp",
#   "gcsfs",
#   "numcodecs>=0.15",
#   "tenacity",
# ]
# ///

import argparse
import os
import pathlib

import dask
import dask.diagnostics
import xarray as xr
from tenacity import retry

DATA_ROOT = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/"


@retry
def robust_open_dataset(target: str, **kwargs) -> xr.Dataset:
    return xr.open_dataset(target, **kwargs)


def compact_dataset(ds: xr.Dataset) -> xr.Dataset:
    return xr.Dataset()


def main(args: argparse.Namespace) -> None:
    """Clones slice of Samudra data at the `dest_root` directory."""
    # Ensure the path/to/dest exists
    if not args.dest.startswith("gs://"):
        pathlib.Path(args.dest).mkdir(parents=True, exist_ok=True)

    time_slice = slice(args.time_start, args.time_end)
    output_chunks = dict(time=args.write_time_chunks)

    for name, dest_fmt in [
        ("OM4", "zarr"),
        ("OM4_means", "netcdf"),
        ("OM4_stds", "netcdf"),
    ]:
        dest = os.path.join(args.dest, name)
        source = DATA_ROOT + name

        # Open Xarray Datasets with retries + exponential backoff.
        if name == "OM4":
            data = robust_open_dataset(source, engine="zarr", chunks={"time": 700})
            data = data.isel(time=time_slice)
        else:
            data = robust_open_dataset(source, engine="zarr", chunks={})

        with dask.diagnostics.ProgressBar():
            if dest_fmt.lower() == "zarr":
                data.chunk(output_chunks).to_zarr(dest + ".zarr")
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
    parser.add_argument("--write_time_chunks", type=int, default=10)
    parser.add_argument("--compact_variables", action="store_true")

    main(parser.parse_args())
