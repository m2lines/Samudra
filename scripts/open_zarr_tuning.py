# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray[io]",
#   "zarr<3",   # Zarr v2 --> change to `zarr<3`; Zarr v3 --> change to `zarr>=3`.
#   "dask",
#   "requests",
#   "aiohttp",
#   "numcodecs>=0.15",
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
import tempfile
import time
from typing import Any

import xarray as xr
import zarr  # type: ignore

REMOTE_DATA = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/OM4"


def main(args: argparse.Namespace) -> float:
    """Calculates elapsed time to open Zarr source over several iterations."""
    source = args.source or REMOTE_DATA

    chunks = {}
    if tc := args.time_chunks:
        chunks["time"] = tc

    write_kwargs: dict[str, Any] = dict(consolidated=False)
    if zarr.__version__.startswith("3"):
        import numcodecs  # type: ignore
        import numcodecs.zarr3  # type: ignore

        # Bug in Zarr v3 Codecs; using a workaround:
        # https://github.com/pydata/xarray/issues/9987#issuecomment-2631471771
        write_kwargs["encoding"] = {"zos": {"compressors": [numcodecs.zarr3.Blosc()]}}

    start_time = time.perf_counter()
    for _ in range(args.n_iters):
        # Zarr v3 has a runtime config contextmanager.
        if zc := args.zarr_concurrency:
            with zarr.config.set({"async.concurrency": zc}):
                ds = xr.open_zarr(source, chunks=chunks, consolidated=True)
        # Zarr v2 does not.
        else:
            ds = xr.open_zarr(source, chunks=chunks, consolidated=True)

        if args.write_test_data:
            with tempfile.TemporaryDirectory() as tmpdir:
                ds.zos.isel(time=slice(0, 1024)).to_zarr(
                    tmpdir + "OM4.zarr", **write_kwargs
                )
    end_time = time.perf_counter()

    return end_time - start_time


def Source(candidate: str) -> pathlib.Path | str:
    """Data Source can either be a local file or a remote URL."""
    if "://" in candidate:
        return candidate
    return pathlib.Path(candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments to tune OpenZarr")
    parser.add_argument("--source", type=Source, default=None)
    parser.add_argument("--n_iters", type=int, default=8)
    parser.add_argument(
        "--zarr_concurrency",
        type=int,
        default=getattr(zarr, "config", {}).get("async.concurrency"),
    )
    parser.add_argument("--time_chunks", type=int, default=None)
    parser.add_argument("--write_test_data", action="store_true")
    args = parser.parse_args()

    print(sys.version)
    print(f"zarr-version={zarr.__version__},xarray-version={xr.__version__}")
    elapsed = main(args)
    print(f"ELAPSED: {elapsed:.4f}s. Config: {args}")
