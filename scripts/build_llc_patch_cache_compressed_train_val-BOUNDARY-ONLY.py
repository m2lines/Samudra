#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
#   "xarray[io]>=2025.1.2",
#   "dask[array]>=2025.2",
#   "zarr<3",
#   "numcodecs",
# ]
# ///
"""Build a compressed train+val-only LLC cache with ONLY the boundary channels.

Boundary-only variant of build_llc_patch_cache_compressed_train_val.py, for
tests where you only need to load the boundary tensor from the cache. Nothing
prognostic is loaded, computed, or written. The single packed array is:

- `boundary[time, boundary_channel, y, x]`

compressed with Blosc (default LZ4 + byte shuffle).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
import zarr

import xarray as xr
from numcodecs import Blosc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_llc_patch_cache_uncompressed_train_val import (  # noqa: E402
    DEFAULT_BOUNDARY_CHANNELS,
    DEFAULT_FLOAT_TYPE,
    DEFAULT_MEANS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PROGNOSTIC_CHANNELS,
    DEFAULT_SOURCE,
    DEFAULT_STDS,
    SUPPORTED_FLOAT_TYPES,
    build_flat_masks,
    build_flat_stats,
    build_packed_data_array,
    extract_xy_coords,
    remove_store,
    select_train_val_times,
    slice_patch,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def build_boundary_only_dataset(
    data: xr.Dataset,
    means: xr.Dataset,
    stds: xr.Dataset,
    args: argparse.Namespace,
) -> xr.Dataset:
    """Boundary-only sibling of build_training_ready_dataset.

    Mirrors the upstream builder but only constructs the boundary array, its
    stats, and its mask. No prognostic variables are read from `data`, so this
    works even when the source has been trimmed to boundary fields only.
    """
    data, train_count, val_count = select_train_val_times(data, args)
    y_coords, x_coords = extract_xy_coords(data)

    boundary = build_packed_data_array(
        data,
        DEFAULT_BOUNDARY_CHANNELS,
        dim_name="boundary_channel",
        y_coords=y_coords,
        x_coords=x_coords,
        time_chunk=args.time_chunk,
        float_type=args.float_type,
    )
    boundary_mean = build_flat_stats(
        means,
        DEFAULT_BOUNDARY_CHANNELS,
        dim_name="boundary_channel",
        float_type=args.float_type,
    )
    boundary_std = build_flat_stats(
        stds,
        DEFAULT_BOUNDARY_CHANNELS,
        dim_name="boundary_channel",
        float_type=args.float_type,
    )
    # build_flat_masks returns (prognostic_mask, boundary_mask); we pass an empty
    # prognostic channel list so no prognostic source vars are needed, and keep
    # only the boundary mask.
    _, boundary_mask = build_flat_masks(
        data,
        DEFAULT_PROGNOSTIC_CHANNELS,   # discarded result; keeps helper happy
        DEFAULT_BOUNDARY_CHANNELS,
        y_coords=y_coords,
        x_coords=x_coords,
    )

    ds_out = xr.Dataset(
        data_vars={
            "boundary": boundary,
            "boundary_mean": boundary_mean,
            "boundary_std": boundary_std,
            "boundary_mask": boundary_mask,
        },
        coords={
            "time": boundary.time,
            "boundary_channel": boundary.boundary_channel,
            "y": boundary.y,
            "x": boundary.x,
        },
        attrs={
            "cache_format": "llc-train-ready-v1-boundaryonly",
            "source_path": str(args.source),
            "means_path": str(args.means),
            "stds_path": str(args.stds),
            "train_start": args.train_start,
            "train_end": args.train_end,
            "val_start": args.val_start,
            "val_end": args.val_end,
            "train_time_count": train_count,
            "val_time_count": val_count,
            "time_chunk": args.time_chunk,
            "float_type": args.float_type,
            "boundary_channel_names_json": json.dumps(DEFAULT_BOUNDARY_CHANNELS),
        },
    )

    logger.info(
        "Selected %d train times + %d val times = %d total times",
        train_count,
        val_count,
        int(ds_out.sizes["time"]),
    )
    logger.info(
        "Packed boundary shape=%s",
        tuple(int(ds_out.sizes[d]) for d in ("time", "boundary_channel", "y", "x")),
    )
    logger.info("Using float type: %s", args.float_type)
    return ds_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--means", type=Path, default=DEFAULT_MEANS)
    parser.add_argument("--stds", type=Path, default=DEFAULT_STDS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-name", default=None)

    parser.add_argument("--face", type=int, default=1)
    parser.add_argument("--i-start", type=int, default=2880)
    parser.add_argument("--i-end", type=int, default=3600)
    parser.add_argument("--j-start", type=int, default=720)
    parser.add_argument("--j-end", type=int, default=1440)

    parser.add_argument("--train-start", default="2011-09-13")
    parser.add_argument("--train-end", default="2012-09-13")
    parser.add_argument("--val-start", default="2012-09-14")
    parser.add_argument("--val-end", default="2012-10-14")

    parser.add_argument(
        "--float-type",
        default=DEFAULT_FLOAT_TYPE,
        choices=SUPPORTED_FLOAT_TYPES,
        help="Floating point precision for boundary data and stats.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=1,
        help="Chunk size along time for the packed boundary array.",
    )
    parser.add_argument(
        "--compressor",
        choices=("lz4", "zstd", "none"),
        default="lz4",
        help="Blosc codec for boundary chunks. Use none for no compression.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="Blosc compression level, 0-9. LZ4 usually wants 1-5; zstd often wants 1-3.",
    )
    parser.add_argument(
        "--shuffle",
        choices=("shuffle", "bitshuffle", "noshuffle"),
        default="shuffle",
        help="Blosc shuffle mode.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--time-batch",
        type=int,
        default=744,
        help=(
            "Number of time steps written per Zarr region write. Smaller batches "
            "keep the Dask graph small so scheduling never OOMs on misaligned "
            "patches. 744 ~= one month at hourly resolution."
        ),
    )
    return parser.parse_args()


def default_output_name(args: argparse.Namespace) -> str:
    start = args.train_start.replace("-", "")
    end = args.val_end.replace("-", "")
    compression_tag = (
        "uncompressed"
        if args.compressor == "none"
        else f"{args.compressor}_c{args.compression_level}_{args.shuffle}"
    )
    return (
        f"LLC4320_face{args.face}_"
        f"i{args.i_start}-{args.i_end}_"
        f"j{args.j_start}-{args.j_end}_"
        f"trainval_boundaryonly_{start}_{end}_"
        f"t{args.time_chunk}_{args.float_type}_{compression_tag}.zarr"
    )


def build_output_path(args: argparse.Namespace) -> Path:
    output_name = args.output_name or default_output_name(args)
    if not output_name.endswith(".zarr"):
        output_name = f"{output_name}.zarr"
    return args.output_root / output_name


def build_compressor(args: argparse.Namespace) -> Blosc | None:
    if args.compressor == "none":
        return None
    if not 0 <= args.compression_level <= 9:
        raise ValueError("compression-level must be between 0 and 9")

    shuffle_by_name = {
        "shuffle": Blosc.SHUFFLE,
        "bitshuffle": Blosc.BITSHUFFLE,
        "noshuffle": Blosc.NOSHUFFLE,
    }
    return Blosc(
        cname=args.compressor,
        clevel=args.compression_level,
        shuffle=shuffle_by_name[args.shuffle],
    )


def build_encoding(
    ds_out: xr.Dataset,
    args: argparse.Namespace,
) -> dict[str, dict[str, object]]:
    y_size = int(ds_out.sizes["y"])
    x_size = int(ds_out.sizes["x"])
    bound_channels = int(ds_out.sizes["boundary_channel"])
    time_size = int(ds_out.sizes["time"])
    compressor = build_compressor(args)

    return {
        "boundary": {
            "compressor": compressor,
            "chunks": (args.time_chunk, bound_channels, y_size, x_size),
        },
        "boundary_mean": {"compressor": None, "chunks": (bound_channels,)},
        "boundary_std": {"compressor": None, "chunks": (bound_channels,)},
        "boundary_mask": {
            "compressor": None,
            "chunks": (bound_channels, y_size, x_size),
        },
        "time": {"compressor": None, "chunks": (min(time_size, 1024),)},
        "boundary_channel": {"compressor": None, "chunks": (bound_channels,)},
        "y": {"compressor": None, "chunks": (y_size,)},
        "x": {"compressor": None, "chunks": (x_size,)},
    }


def write_training_ready_in_batches(
    ds_out: xr.Dataset,
    tmp_path: Path,
    encoding: dict[str, dict[str, object]],
    time_batch: int,
) -> None:
    """Write the store in temporal batches using Zarr region writes.

    Rationale: a single ds_out.to_zarr() over the whole year builds one enormous
    Dask graph. dask.order() must sort that entire graph in memory before any
    work runs, and misaligned patches (extra concatenate layers per timestep)
    inflate it until ordering itself OOMs. Writing one batch at a time keeps each
    graph ~(time_batch / n_times) of the size, so scheduling stays cheap and the
    peak memory is bounded regardless of how many source chunks a patch touches.
    """
    time_size = int(ds_out.sizes["time"])
    time_vars = [name for name, var in ds_out.data_vars.items() if "time" in var.dims]
    static_ds = ds_out.drop_vars(time_vars)
    time_ds = ds_out[time_vars]

    # Drop coords from the time dataset up front: all coordinates (time, y, x,
    # channel labels) are written once in phase 1 as part of static_ds. The array
    # dims are still recorded via _ARRAY_DIMENSIONS, so this is safe.
    time_skeleton = time_ds.drop_vars(list(time_ds.coords))

    # IMPORTANT: build each encoding dict from the variables actually written in
    # that phase. time_ds.variables still contains coordinate variables
    # (boundary_channel, ...) which already exist in the store after phase 1;
    # passing encoding for them triggers a "variable already exists, but encoding
    # was provided" error on the append write.
    static_encoding = {k: v for k, v in encoding.items() if k in static_ds.variables}
    time_encoding = {k: v for k, v in encoding.items() if k in time_skeleton.variables}

    # Phase 1: write coords + static vars (mask/mean/std) with real values.
    logger.info("Writing coordinates and static variables")
    static_ds.to_zarr(
        tmp_path, mode="w", encoding=static_encoding, consolidated=False
    )

    # Phase 2: create the big time-varying array as a metadata-only skeleton.
    # compute=False creates the zarr array (correct shape/chunks/compressor)
    # but defers the data write, which we intentionally never trigger.
    logger.info("Creating time-varying array skeletons")
    time_skeleton.to_zarr(
        tmp_path,
        mode="a",
        encoding=time_encoding,
        compute=False,
        consolidated=False,
    )

    # Phase 3: fill the skeleton one temporal batch at a time via region writes.
    n_batches = (time_size + time_batch - 1) // time_batch
    for b, start in enumerate(range(0, time_size, time_batch)):
        stop = min(start + time_batch, time_size)
        logger.info(
            "Writing time batch %d/%d: time[%d:%d)", b + 1, n_batches, start, stop
        )
        batch = time_ds.isel(time=slice(start, stop)).drop_vars(
            list(time_ds.coords)
        )
        batch.to_zarr(
            tmp_path,
            region={"time": slice(start, stop)},
            consolidated=False,
        )

    logger.info("Consolidating metadata")
    zarr.consolidate_metadata(str(tmp_path))


def main() -> None:
    args = parse_args()
    if args.time_chunk <= 0:
        raise ValueError("time-chunk must be positive")
    if args.i_end <= args.i_start:
        raise ValueError("i-end must be greater than i-start")
    if args.j_end <= args.j_start:
        raise ValueError("j-end must be greater than j-start")

    output_path = build_output_path(args)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )
    if tmp_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{tmp_path} already exists. Pass --overwrite to replace it."
        )
    if args.overwrite:
        remove_store(output_path)
        remove_store(tmp_path)

    logger.info("Opening source dataset: %s", args.source)
    data = xr.open_zarr(args.source, chunks={})
    logger.info("Opening means: %s", args.means)
    means = xr.open_zarr(args.means)
    logger.info("Opening stds: %s", args.stds)
    stds = xr.open_zarr(args.stds)

    # NOTE: even though we only *write* boundary, DEFAULT_BOUNDARY_CHANNELS can
    # reference fields from the full source set (e.g. surface Theta/Eta). Keep
    # the full var set available for the boundary build; these are lazy via
    # open_zarr, and we never compute the prognostic array, so nothing extra is
    # read or written.
    required_vars = {
        "U",
        "V",
        "Theta",
        "Salt",
        "Eta",
        "oceTAUX",
        "oceTAUY",
        "oceQnet",
    }
    missing = sorted(required_vars - set(data.data_vars))
    if missing:
        raise KeyError(f"Source dataset is missing required vars: {missing}")
    if "mask_c" not in data.data_vars and "wetmask" not in data.data_vars:
        raise KeyError("Source dataset is missing mask_c/wetmask.")

    selected_vars = sorted(
        (required_vars | {"mask_c", "wetmask"}) & set(data.data_vars)
    )
    data = data[selected_vars]
    logger.info(
        "Slicing source to face=%d i=[%d:%d) j=[%d:%d)",
        args.face,
        args.i_start,
        args.i_end,
        args.j_start,
        args.j_end,
    )
    data = slice_patch(data, args)

    ds_out = build_boundary_only_dataset(data, means, stds, args)
    ds_out.attrs.update(
        {
            "compression_codec": args.compressor,
            "compression_level": args.compression_level,
            "compression_shuffle": args.shuffle,
            "compression_target_vars": "boundary",
            "boundary_channel_count": len(DEFAULT_BOUNDARY_CHANNELS),
        }
    )
    logger.info("Output path: %s", output_path)
    logger.info(
        "Compression: codec=%s level=%s shuffle=%s",
        args.compressor,
        args.compression_level,
        args.shuffle,
    )

    if args.dry_run:
        logger.info("Dry run requested; not writing any data.")
        return

    if args.time_batch <= 0:
        raise ValueError("time-batch must be positive")

    encoding = build_encoding(ds_out, args)
    logger.info(
        "Writing temporary store in time batches of %d: %s",
        args.time_batch,
        tmp_path,
    )
    write_training_ready_in_batches(ds_out, tmp_path, encoding, args.time_batch)

    logger.info("Moving completed store to: %s", output_path)
    if output_path.exists():
        remove_store(output_path)
    shutil.move(str(tmp_path), str(output_path))
    logger.info("Done.")


if __name__ == "__main__":
    main()