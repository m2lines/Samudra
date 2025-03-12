"""Script to clone remote Samudra data locally."""
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
# ]
# ///

import argparse
import os
import pathlib
from typing import Any

import dask
import dask.diagnostics
import numcodecs
import numcodecs.zarr3
import xarray as xr

DATA_ROOT = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/"


def main(dest_root: str, time_slice: slice, upgrade_zarr: bool = False) -> None:
    """Clones slice of Samudra data at the `dest_root` directory."""
    # Ensure the path/to/dest exists
    if not dest_root.startswith("gs://"):
        pathlib.Path(dest_root).mkdir(parents=True, exist_ok=True)

    for name, dest_fmt in [
        ("OM4", "zarr"),
        ("OM4_means", "netcdf"),
        ("OM4_stds", "netcdf"),
    ]:
        dest = os.path.join(dest_root, name)
        target = DATA_ROOT + name

        if name == "OM4":
            data = xr.open_dataset(target, engine="zarr", chunks={"time": 700})
            data = data.isel(time=time_slice)
        else:
            data = xr.open_dataset(target, engine="zarr", chunks={})

        kwargs: dict[str, Any] = {}
        if upgrade_zarr:
            kwargs["zarr_format"] = 3
            # Bug in Zarr v3 Codecs; using a workaround:
            # https://github.com/pydata/xarray/issues/9987#issuecomment-2631471771
            kwargs["encoding"] = {
                v: {"compressors": [numcodecs.zarr3.Blosc()]}
                for v in data.data_vars.keys()
            }

        with dask.diagnostics.ProgressBar():
            if dest_fmt.lower() == "zarr":
                data.chunk(dict(time=1)).to_zarr(
                    dest + ".zarr", consolidated=False, **kwargs
                )
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
    parser.add_argument("--migrate_to_zarr_v3", action="store_true")
    args = parser.parse_args()

    time_range = slice(args.time_start, args.time_end)
    main(args.dest, time_slice=time_range, upgrade_zarr=args.migrate_to_zarr_v3)
