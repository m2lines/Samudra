#!/usr/bin/env python3
"""Stage zarr data to node-local storage for faster I/O during training.

Copies zarr stores from a (potentially slow) network filesystem to fast
node-local NVMe, optionally time-slicing to reduce the amount of data copied.
Means/stds zarr stores are copied in full (they're tiny).

Usage inside a SLURM job:
    python scripts/stage_data.py \
        --src-root /scratch/user/data \
        --dst-root /state/partition1/data \
        --time-start 1975-01-03 \
        --time-end 2014-10-05 \
        om4_quarterdeg_v2 om4_halfdeg_v4 om4_onedeg_v3
"""

import argparse
import shutil
import time
from pathlib import Path

import xarray as xr


def stage_zarr(
    src: Path,
    dst: Path,
    time_start: str | None = None,
    time_end: str | None = None,
) -> None:
    """Copy a zarr store, optionally selecting a time range."""
    if dst.exists():
        print(f"  Already staged: {dst}")
        return

    ds = xr.open_dataset(src, engine="zarr", chunks={})

    if "time" in ds.dims and time_start and time_end:
        n_before = ds.sizes["time"]
        ds = ds.sel(time=slice(time_start, time_end))
        n_after = ds.sizes["time"]
        print(f"  Time-sliced: {n_before} → {n_after} timesteps")

    ds.to_zarr(dst, mode="w")
    print(f"  Wrote: {dst}")


def stage_source(
    src_root: Path,
    dst_root: Path,
    source_dir: str,
    time_start: str | None,
    time_end: str | None,
) -> None:
    """Stage one data source (OM4.zarr + means + stds)."""
    src_dir = src_root / source_dir
    dst_dir = dst_root / source_dir

    if not src_dir.exists():
        print(f"WARNING: source not found, skipping: {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Main data zarr — time-slice to reduce copy size.
    print(f"Staging {source_dir}/OM4.zarr ...")
    t0 = time.monotonic()
    stage_zarr(src_dir / "OM4.zarr", dst_dir / "OM4.zarr", time_start, time_end)
    print(f"  Took {time.monotonic() - t0:.1f}s")

    # Means and stds — tiny, copy in full.
    for name in ["OM4_means.zarr", "OM4_stds.zarr"]:
        src_path = src_dir / name
        dst_path = dst_dir / name
        if dst_path.exists():
            print(f"  Already staged: {dst_path}")
        elif src_path.exists():
            shutil.copytree(src_path, dst_path)
            print(f"  Copied: {dst_path}")
        else:
            print(f"  WARNING: {src_path} not found, skipping")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="Source subdirectories to stage")
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument(
        "--time-start", default=None, help="Start of time range to keep"
    )
    parser.add_argument("--time-end", default=None, help="End of time range to keep")
    args = parser.parse_args()

    args.dst_root.mkdir(parents=True, exist_ok=True)

    print(f"=== Data Staging ===")
    print(f"Source:     {args.src_root}")
    print(f"Dest:       {args.dst_root}")
    if args.time_start and args.time_end:
        print(f"Time range: {args.time_start} to {args.time_end}")
    print()

    t0 = time.monotonic()
    for source_dir in args.sources:
        stage_source(
            args.src_root,
            args.dst_root,
            source_dir,
            args.time_start,
            args.time_end,
        )
        print()

    print(f"Total staging time: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
