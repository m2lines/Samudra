# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray[io]",
#   "zarr>=3",   # Zarr v2 --> change to `zarr<3`; Zarr v3 --> change to `zarr>=3`.
#   "dask",
#   "requests",
#   "aiohttp",
#   "numcodecs>=0.15",
#   "zarrs",
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

    chunks: Any = {}  # For some reason `dict[str, int] | None` doesn't pass type check.
    if tc := args.time_chunks:
        assert not args.disable_dask, "Dask must be enabled for time chunks to be set."
        chunks["time"] = tc

    if args.disable_dask:
        chunks = None

    write_kwargs: dict[str, Any] = dict(consolidated=False)
    if use_zarr_v3 := zarr.__version__.startswith("3"):
        from numcodecs.zarr3 import Blosc  # type: ignore

        # Bug in Zarr v3 Codecs; using a workaround:
        # https://github.com/pydata/xarray/issues/9987
        write_kwargs["encoding"] = {"zos": {"compressors": [Blosc()]}}

    zarr3_config = {}
    if args.use_zarrs:
        import zarrs  # noqa: F401

        assert use_zarr_v3, "Zarrs (the rust backend) only supports Zarr v3!"
        zarr3_config.update({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

    if zc := args.zarr_concurrency:
        zarr3_config.update({"async.concurrency": zc})

    start_time = time.perf_counter()
    for _ in range(args.n_iters):
        # Zarr v3 has a runtime config contextmanager.
        if use_zarr_v3:
            with zarr.config.set(zarr3_config):
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
    parser.add_argument(
        "--use_zarrs",
        action="store_true",
        help="`zarrs` is a rust backend for zarr-python (requires zarr v3+)",
    )

    parser.add_argument(
        "--disable_dask",
        action="store_true",
        help="Turns off dask by setting `chunks=None`.",
    )
    args = parser.parse_args()

    print(sys.version)
    print(f"zarr-version={zarr.__version__},xarray-version={xr.__version__}")
    elapsed = main(args)
    print(f"ELAPSED: {elapsed:.4f}s. Config: {args}")
