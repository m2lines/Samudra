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
"""Build a compressed train+val-only LLC cache in the training-ready layout.

This is the compressed sibling of build_llc_patch_cache_uncompressed_train_val.py.
It keeps the same packed arrays:

- `prognostic[time, prognostic_channel, y, x]`
- `boundary[time, boundary_channel, y, x]`

but compresses the large time-varying chunks with Blosc. The default is LZ4
with byte shuffle, which is usually the right first choice when training is
filesystem-bandwidth bound and CPU decompression is cheaper than reading the
extra bytes.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
import zarr

import numpy as np
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
    build_training_ready_dataset,
    remove_store,
    slice_patch,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


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
        "--append",
        action="store_true",
        help=(
            "Extend an existing store with a later time window instead of "
            "rebuilding it. Requires --append-start/--append-end, and the new "
            "window must start after the last time already stored: zarr appends "
            "by concatenation, so out-of-order times would corrupt the axis."
        ),
    )
    parser.add_argument(
        "--append-start",
        default=None,
        help="First timestamp of the window to append (inclusive).",
    )
    parser.add_argument(
        "--append-end",
        default=None,
        help="Last timestamp of the window to append (inclusive).",
    )
    parser.add_argument(
        "--append-label",
        default="test",
        help="Name recorded in the store attrs for the appended window, e.g. 'test'.",
    )

    parser.add_argument(
        "--float-type",
        default=DEFAULT_FLOAT_TYPE,
        choices=SUPPORTED_FLOAT_TYPES,
        help="Floating point precision for prognostic/boundary data and stats.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=1,
        help="Chunk size along time for packed prognostic/boundary arrays.",
    )
    parser.add_argument(
        "--compressor",
        choices=("lz4", "zstd", "none"),
        default="lz4",
        help="Blosc codec for prognostic/boundary chunks. Use none for no compression.",
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
        f"trainval_ready_{start}_{end}_"
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
    prog_channels = int(ds_out.sizes["prognostic_channel"])
    bound_channels = int(ds_out.sizes["boundary_channel"])
    time_size = int(ds_out.sizes["time"])
    compressor = build_compressor(args)

    return {
        "prognostic": {
            "compressor": compressor,
            "chunks": (args.time_chunk, prog_channels, y_size, x_size),
        },
        "boundary": {
            "compressor": compressor,
            "chunks": (args.time_chunk, bound_channels, y_size, x_size),
        },
        "prognostic_mean": {"compressor": None, "chunks": (prog_channels,)},
        "prognostic_std": {"compressor": None, "chunks": (prog_channels,)},
        "boundary_mean": {"compressor": None, "chunks": (bound_channels,)},
        "boundary_std": {"compressor": None, "chunks": (bound_channels,)},
        "prognostic_mask": {
            "compressor": None,
            "chunks": (prog_channels, y_size, x_size),
        },
        "boundary_mask": {
            "compressor": None,
            "chunks": (bound_channels, y_size, x_size),
        },
        "XC": {"compressor": None, "chunks": (y_size, x_size)},
        "YC": {"compressor": None, "chunks": (y_size, x_size)},
        "rA": {"compressor": None, "chunks": (y_size, x_size)},
        "time": {"compressor": None, "chunks": (min(time_size, 1024),)},
        "prognostic_channel": {"compressor": None, "chunks": (prog_channels,)},
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
    # that phase. time_ds.variables still contains the coordinate variables
    # (boundary_channel, prognostic_channel, ...) which already exist in the store
    # after phase 1; passing encoding for them triggers a "variable already exists,
    # but encoding was provided" error on the append write.
    static_encoding = {k: v for k, v in encoding.items() if k in static_ds.variables}
    time_encoding = {k: v for k, v in encoding.items() if k in time_skeleton.variables}

    # Phase 1: write coords + static vars (masks/means/stds) with real values.
    logger.info("Writing coordinates and static variables")
    static_ds.to_zarr(
        tmp_path, mode="w", encoding=static_encoding, consolidated=False
    )

    # Phase 2: create the big time-varying arrays as metadata-only skeletons.
    # compute=False creates the zarr arrays (correct shape/chunks/compressor)
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


def validate_append_compatible(existing: xr.Dataset, ds_new: xr.Dataset) -> None:
    """Refuse to append anything the existing store cannot absorb cleanly.

    Appending along time leaves every other array untouched, so the new window
    silently inherits the store's geometry, masks, and normalization. If any of
    those actually differ the result is a cache that looks fine and trains wrong,
    which is far worse than a failed build.
    """
    for name in ("x", "y"):
        if name not in existing.coords or name not in ds_new.coords:
            raise ValueError(f"Both stores must carry the {name} index coordinate")
        if not np.array_equal(
            existing.coords[name].to_numpy(), ds_new.coords[name].to_numpy()
        ):
            raise ValueError(
                f"Append refused: {name} index coordinate differs, so the new "
                "window covers a different patch than the store."
            )

    for attr in ("prognostic_channel_names_json", "boundary_channel_names_json"):
        if existing.attrs.get(attr) != ds_new.attrs.get(attr):
            raise ValueError(f"Append refused: {attr} differs from the store.")

    for name in ("prognostic_mean", "prognostic_std", "boundary_mean", "boundary_std"):
        if name not in existing or name not in ds_new:
            continue
        if not np.allclose(
            np.asarray(existing[name].to_numpy(), dtype=np.float64),
            np.asarray(ds_new[name].to_numpy(), dtype=np.float64),
            equal_nan=True,
        ):
            raise ValueError(
                f"Append refused: {name} differs, so the appended window would "
                "be normalized differently from the rest of the store."
            )

    for name in ("prognostic_mask", "boundary_mask"):
        if name not in existing or name not in ds_new:
            continue
        if not np.array_equal(
            np.asarray(existing[name].to_numpy()), np.asarray(ds_new[name].to_numpy())
        ):
            raise ValueError(f"Append refused: {name} differs from the store.")

    for name in ("prognostic", "boundary"):
        if existing[name].dtype != ds_new[name].dtype:
            raise ValueError(
                f"Append refused: {name} dtype {ds_new[name].dtype} does not match "
                f"the store's {existing[name].dtype}; pass a matching --float-type."
            )


def append_time_window(
    output_path: Path,
    ds_new: xr.Dataset,
    *,
    time_batch: int,
    label: str,
) -> None:
    """Concatenate a later time window onto an existing training-ready store."""
    existing = xr.open_zarr(output_path, consolidated=True)
    validate_append_compatible(existing, ds_new)

    existing_times = existing["time"].to_numpy()
    new_times = ds_new["time"].to_numpy()

    already = np.isin(new_times, existing_times)
    if already.all():
        logger.info("Every requested time is already in the store; nothing to do.")
        return
    if already.any():
        logger.info(
            "Dropping %d requested time(s) already present in the store",
            int(already.sum()),
        )
        ds_new = ds_new.isel(time=np.flatnonzero(~already))
        new_times = ds_new["time"].to_numpy()

    if new_times.min() <= existing_times.max():
        raise ValueError(
            f"Append refused: the new window starts at {new_times.min()} but the "
            f"store already runs to {existing_times.max()}. Zarr appends by "
            "concatenation rather than merging, so an overlapping or earlier "
            "window would leave the time axis unsorted. Rebuild instead."
        )

    time_vars = [name for name, var in ds_new.data_vars.items() if "time" in var.dims]
    time_ds = ds_new[time_vars]
    # Everything except `time` already exists in the store and must not be
    # re-sent, or xarray tries to append along a dimension they do not have.
    time_ds = time_ds.drop_vars([c for c in time_ds.coords if c != "time"])
    # xarray rewrites the root group's attrs on every append and offers no way
    # to opt out, so hand it the store's existing attrs to write back unchanged.
    # An empty dict here would erase the store's identity -- channel names,
    # train/val counts, geometry -- and the next append would then reject it.
    time_ds.attrs = dict(existing.attrs)

    total = int(time_ds.sizes["time"])
    n_batches = (total + time_batch - 1) // time_batch
    logger.info(
        "Appending %d timestamps (%s -> %s) in %d batch(es)",
        total,
        new_times.min(),
        new_times.max(),
        n_batches,
    )
    for index, start in enumerate(range(0, total, time_batch)):
        stop = min(start + time_batch, total)
        logger.info("Appending batch %d/%d: time[%d:%d)", index + 1, n_batches, start, stop)
        time_ds.isel(time=slice(start, stop)).to_zarr(
            output_path,
            mode="a",
            append_dim="time",
            consolidated=False,
        )

    store = zarr.open(str(output_path), mode="a")
    store.attrs.update(
        {
            f"{label}_start": str(np.datetime_as_string(new_times.min(), unit="D")),
            f"{label}_end": str(np.datetime_as_string(new_times.max(), unit="D")),
            f"{label}_time_count": total,
        }
    )
    logger.info("Consolidating metadata")
    zarr.consolidate_metadata(str(output_path))

    check = xr.open_zarr(output_path, consolidated=True)
    times = check["time"].to_numpy()
    if not np.all(np.diff(times.astype("datetime64[ns]")) > np.timedelta64(0)):
        raise RuntimeError(
            "Append produced a non-monotonic time axis; the store is corrupt."
        )
    logger.info(
        "Store now covers %s -> %s (%d timestamps)",
        np.datetime_as_string(times.min(), unit="s"),
        np.datetime_as_string(times.max(), unit="s"),
        times.size,
    )


def main() -> None:
    args = parse_args()
    if args.time_chunk <= 0:
        raise ValueError("time-chunk must be positive")
    if args.i_end <= args.i_start:
        raise ValueError("i-end must be greater than i-start")
    if args.j_end <= args.j_start:
        raise ValueError("j-end must be greater than j-start")

    # The store name encodes the ORIGINAL train/val window, so resolve it before
    # the append window overrides those args below.
    output_path = build_output_path(args)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")

    if args.append:
        if not args.append_start or not args.append_end:
            raise ValueError("--append requires --append-start and --append-end")
        if not output_path.exists():
            raise FileNotFoundError(
                f"--append needs an existing store, but {output_path} is missing. "
                "Build it first, or drop --append to create it."
            )
        logger.info(
            "Append mode: extending %s with %s -> %s",
            output_path,
            args.append_start,
            args.append_end,
        )
        # select_train_val_times concatenates and de-duplicates its two windows,
        # so pointing both at the append window yields exactly that window once.
        args.train_start = args.val_start = args.append_start
        args.train_end = args.val_end = args.append_end
    elif output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it, or "
            "--append with --append-start/--append-end to extend it in place."
        )
    if not args.append:
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

    required_vars = {
        "U",
        "V",
        "Theta",
        "Salt",
        "Eta",
        "oceTAUX",
        "oceTAUY",
        "oceQnet",
        "XC",
        "YC",
        "rA",
    }
    missing = sorted(required_vars - set(data.data_vars))
    if missing:
        raise KeyError(f"Source dataset is missing required vars: {missing}")
    if "mask_c" not in data.data_vars and "wetmask" not in data.data_vars:
        raise KeyError("Source dataset is missing mask_c/wetmask.")

    selected_vars = sorted(
        (required_vars | {"mask_c", "wetmask", "XC", "YC", "rA"})
        & set(data.data_vars)
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

    ds_out = build_training_ready_dataset(data, means, stds, args)
    ds_out.attrs.update(
        {
            "compression_codec": args.compressor,
            "compression_level": args.compression_level,
            "compression_shuffle": args.shuffle,
            "compression_target_vars": "prognostic,boundary",
            "prognostic_channel_count": len(DEFAULT_PROGNOSTIC_CHANNELS),
            "boundary_channel_count": len(DEFAULT_BOUNDARY_CHANNELS),
            # A global LLC coordinate is (face, i, j), but only i and j survive
            # in the x/y index arrays. Record the face so a tile catalog never
            # has to parse it back out of the filename.
            "llc_face": args.face,
            "llc_i_start": args.i_start,
            "llc_i_end": args.i_end,
            "llc_j_start": args.j_start,
            "llc_j_end": args.j_end,
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

    if args.append:
        # Only the time-varying arrays are appended; masks, stats, XC/YC/rA and
        # the encoding all stay as the store already has them, which is exactly
        # why validate_append_compatible checks they would have matched. The
        # attrs must reach that check intact -- they are what it compares.
        append_time_window(
            output_path,
            ds_out,
            time_batch=args.time_batch,
            label=args.append_label,
        )
        return

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
