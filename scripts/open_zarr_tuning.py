# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray[io]",
#   "zarr>=3",
#   "dask",
#   "requests",
#   "aiohttp",
# ]
# ///
"""Experimenting with optimal ways to open the OM4 Zarr.

Using techniques from this blog post:
- https://earthmover.io/blog/xarray-open-zarr-improvements

How to run experiments:
- Change the Zarr version (above) to compare ZarrV2 vs ZarrV3.
- Configure: `uv run scripts/open_zarr_tuning.py --help`
"""

import argparse
import pathlib
import sys
import time

import xarray as xr
import zarr

REMOTE_DATA = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4"


def main(parsed_args: argparse.Namespace):
    if zcon := parsed_args.zarr_concurrency:
        zarr.config.set({"async": {"concurrency": zcon, "timeout": None}})

    chunks = {}
    if tc := parsed_args.time_chunks:
        chunks = dict(time=tc)

    target = parsed_args.target or REMOTE_DATA

    start_time = time.perf_counter()
    for _ in range(8):
        xr.open_zarr(target, chunks=chunks, consolidated=False)
    end_time = time.perf_counter()

    print(f"ELAPSED: {end_time - start_time:.4f} (8 iter). config: {parsed_args}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments to tune OpenZarr")
    parser.add_argument("--target", type=pathlib.Path, default=None)
    parser.add_argument("--zarr_concurrency", type=int, default=None)
    parser.add_argument("--time_chunks", type=int, default=None)

    print(f"python-version={sys.version}")
    print(f"zarr-version={zarr.__version__}")
    main(parser.parse_args())
