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
# ]
# ///

import argparse
import os
import pathlib

import dask
import dask.diagnostics
import xarray as xr
from xarray.backends.common import robust_getitem

DATA_ROOT = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/"


class DatasetOpener:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getitem__(self, target):
        return xr.open_dataset(target, **self.kwargs)


def main(dest_root: str, time_slice: slice, write_time_chunks: int) -> None:
    """Clones slice of Samudra data at the `dest_root` directory."""
    # Ensure the path/to/dest exists
    if not dest_root.startswith("gs://"):
        pathlib.Path(dest_root).mkdir(parents=True, exist_ok=True)

    output_chunks = dict(time=write_time_chunks)

    chunked_opener = DatasetOpener(engine="zarr", chunks={"time": 700})
    zarr_opener = DatasetOpener(engine="zarr", chunks={})

    for name, dest_fmt in [
        ("OM4", "zarr"),
        ("OM4_means", "netcdf"),
        ("OM4_stds", "netcdf"),
    ]:
        dest = os.path.join(dest_root, name)
        target = DATA_ROOT + name

        # Open Xarray Datasets with retries + exponential backoff.
        if name == "OM4":
            data = robust_getitem(chunked_opener, target)
            data = data.isel(time=time_slice)
        else:
            data = robust_getitem(zarr_opener, target)

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
    parser.add_argument("--write_time_chunks", type=int, default=1)
    args = parser.parse_args()

    time_range = slice(args.time_start, args.time_end)
    main(args.dest, time_slice=time_range, write_time_chunks=args.write_time_chunks)
