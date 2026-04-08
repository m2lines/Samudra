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

DATA_ROOT = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/"

# All scale directories and their zarr stores used by FOMO multiscale training.
MULTISCALE_SOURCES = [
    "om4_quarterdeg_v2/OM4.zarr",
    "om4_quarterdeg_v2/OM4_means.zarr",
    "om4_quarterdeg_v2/OM4_stds.zarr",
    "om4_halfdeg_v4/OM4.zarr",
    "om4_halfdeg_v4/OM4_means.zarr",
    "om4_halfdeg_v4/OM4_stds.zarr",
    "om4_onedeg_v3/OM4.zarr",
    "om4_onedeg_v3/OM4_means.zarr",
    "om4_onedeg_v3/OM4_stds.zarr",
]


@retry
def robust_open_dataset(target: str, **kwargs) -> xr.Dataset:
    return xr.open_dataset(target, **kwargs)


def clone_one(
    source: str,
    dest: str,
    *,
    compact: bool,
    time_slice: slice,
    read_time_chunks: int,
    write_time_chunks: int,
    is_main_data: bool,
) -> None:
    """Clone a single zarr store, optionally compacting variables."""
    print(f"\n{'=' * 60}")
    print(f"Source: {source}")
    print(f"Dest:   {dest}")

    if is_main_data:
        data = robust_open_dataset(
            source, engine="zarr", chunks={"time": read_time_chunks}
        )
        data = data.isel(time=time_slice)
    else:
        data = robust_open_dataset(source, engine="zarr", chunks={})

    if compact:
        data = compact_dataset(data)

    output_chunks = dict(time=write_time_chunks)
    if "lev" in data.dims:
        output_chunks["lev"] = 19

    with dask.diagnostics.ProgressBar():
        if is_main_data:
            data = data.chunk(output_chunks)
        data.to_zarr(dest, mode="w")


def main(args: argparse.Namespace) -> None:
    """Clones and optionally compacts Zarr ocean data."""
    if args.local_cluster:
        cluster = LocalCluster(
            n_workers=args.n_workers,
            threads_per_worker=args.threads_per_worker,
        )
        client = cluster.get_client()  # noqa: F841
        print(f"Dask dashboard: {client.dashboard_link}")

    time_slice = slice(args.time_start, args.time_end)

    if args.src_root:
        # Local-to-local mode: read from src_root, write to dest.
        sources = args.sources if args.sources else MULTISCALE_SOURCES
        for rel_path in sources:
            source = os.path.join(args.src_root, rel_path)
            dest = os.path.join(args.dest, rel_path)
            pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)

            # OM4.zarr is the main time-series data; means/stds are small.
            is_main = os.path.basename(rel_path) == "OM4.zarr"

            clone_one(
                source=source,
                dest=dest,
                compact=args.compact_variables,
                time_slice=time_slice,
                read_time_chunks=args.read_time_chunks,
                write_time_chunks=args.write_time_chunks,
                is_main_data=is_main,
            )
    else:
        # Legacy remote mode: read from OSN HTTP endpoint.
        if not args.dest.startswith("gs://") and not args.dest.startswith("s3://"):
            pathlib.Path(args.dest).mkdir(parents=True, exist_ok=True)

        for name in ["OM4", "OM4_means", "OM4_stds"]:
            source = DATA_ROOT + name
            dest = os.path.join(args.dest, name + ".zarr")
            is_main = name == "OM4"

            clone_one(
                source=source,
                dest=dest,
                compact=args.compact_variables,
                time_slice=time_slice,
                read_time_chunks=args.read_time_chunks,
                write_time_chunks=args.write_time_chunks,
                is_main_data=is_main,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "clone_data",
        description="Clone and compact Zarr ocean data. "
        "Use --src_root for local-to-local copies (e.g. on HPC).",
    )
    parser.add_argument("dest", type=str, help="Root directory for output datasets.")
    parser.add_argument(
        "--src_root",
        type=str,
        default=None,
        help="Root directory for source zarr data (local-to-local mode). "
        "If unset, reads from the remote OSN endpoint.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Relative zarr paths under src_root to process. "
        "Defaults to all FOMO multiscale sources.",
    )
    parser.add_argument(
        "--time_start",
        type=int,
        default=0,
        help="Start index for data.isel() along time dimension.",
    )
    parser.add_argument(
        "--time_end",
        type=int,
        default=None,
        help="End index for data.isel() along time dimension.",
    )
    parser.add_argument("--read_time_chunks", type=int, default=50)
    parser.add_argument("--write_time_chunks", type=int, default=1)
    parser.add_argument("--compact_variables", action="store_true")
    parser.add_argument("--local_cluster", action="store_true")
    parser.add_argument(
        "--n_workers",
        type=int,
        default=None,
        help="Number of Dask workers (only with --local_cluster).",
    )
    parser.add_argument(
        "--threads_per_worker",
        type=int,
        default=1,
        help="Threads per Dask worker (only with --local_cluster).",
    )

    main(parser.parse_args())
