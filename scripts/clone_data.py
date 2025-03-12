"""Script to clone remote Samudra data locally."""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "xarray[io]",
#     "dask",
# ]
# ///

import argparse
import pathlib

import dask
import dask.diagnostics
import xarray as xr

DATA_ROOT = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/"


def main(dest_root: pathlib.Path, time_slice: slice) -> None:
    """Clones slice of Samudra data at the `dest_root` directory."""
    # Ensure the path/to/dest exists
    dest_root.mkdir(parents=True, exist_ok=True)

    for name, dest_fmt in [
        ("OM4", "zarr"),
        ("OM4_means", "netcdf"),
        ("OM4_stds", "netcdf"),
    ]:
        dest = dest_root / name

        if name == "OM4":
            data = xr.open_dataset(
                DATA_ROOT + name, engine="zarr", chunks={"time": 700}
            )
            data = data.isel(time=time_slice)
            data = data.chunk({"time": 1})
        else:
            data = xr.open_dataset(DATA_ROOT + name, engine="zarr", chunks={})

        with dask.diagnostics.ProgressBar():
            if dest_fmt.lower() == "zarr":
                data.to_zarr(str(dest) + ".zarr")
            else:
                data.to_netcdf(str(dest) + ".nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "clone_data", description="Make a local copy of the Samudra dataset (~70 GiBs)."
    )
    parser.add_argument(
        "dest", type=pathlib.Path, help="Root directory for local copy of datasets."
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
    args = parser.parse_args()

    time_range = slice(args.time_start, args.time_end)
    main(args.dest, time_slice=time_range)
