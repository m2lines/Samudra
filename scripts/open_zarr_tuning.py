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


def main(args: argparse.Namespace) -> float:
    """Calculates elapsed time to open Zarr target over several iterations."""
    chunks = {}
    if tc := args.time_chunks:
        chunks["time"] = tc

    target = args.target or REMOTE_DATA

    start_time = time.perf_counter()
    for _ in range(args.n_iters):
        # Zarr v3 has a runtime config contextmanager.
        if zc := args.zarr_concurrency:
            with zarr.config.set({"async.concurrency": zc}):
                xr.open_zarr(target, chunks=chunks)
        # Zarr v2 does not.
        else:
            xr.open_zarr(target, chunks=chunks)
    end_time = time.perf_counter()

    return end_time - start_time


def Target(candidate: str) -> pathlib.Path | str:
    """Target can either be a local file or a remote URL."""
    if "://" in candidate:
        return candidate
    return pathlib.Path(candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments to tune OpenZarr")
    parser.add_argument("--target", type=Target, default=None)
    parser.add_argument("--n_iters", type=int, default=8)
    parser.add_argument(
        "--zarr_concurrency",
        type=int,
        default=getattr(zarr, "config", {}).get("async.concurrency"),
    )
    parser.add_argument("--time_chunks", type=int, default=None)
    args = parser.parse_args()

    print(sys.version)
    print(f"zarr-version={zarr.__version__},xarray-version={xr.__version__}")
    elapsed = main(args)
    print(f"ELAPSED: {elapsed:.4f}s. Config: {args}")
