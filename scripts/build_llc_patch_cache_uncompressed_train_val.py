#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0",
#   "xarray[io]>=2025.1.2",
#   "dask[array]>=2025.2",
#   "zarr<3",
# ]
# ///
"""Build a train+val-only LLC cache in a training-ready layout.

This script slices an Agulhas LLC patch from the full LLC source (or from an
already-sliced patch cache) and repacks just the train+val time range into a
layout that is easier to train from directly:

- `prognostic[time, prognostic_channel, y, x]`
- `boundary[time, boundary_channel, y, x]`
- flattened means/stds for each channel
- flattened prognostic/boundary masks

The output is written uncompressed and with full-channel/full-patch chunks so it
can later be copied to node-local scratch or another fast filesystem without any
extra format conversion.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS

DEFAULT_SOURCE = Path("/orcd/data/abodner/003/LLC4320/LLC4320")
DEFAULT_MEANS = Path("/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_means.zarr")
DEFAULT_STDS = Path("/orcd/data/abodner/002/cody/LLC_means_stds/var_96_LLC_stds.zarr")
DEFAULT_OUTPUT_ROOT = Path("/orcd/data/abodner/002/cody/LLC_patch")
DEFAULT_PROGNOSTIC_CHANNELS = PROGNOSTIC_VARS["all"]
DEFAULT_BOUNDARY_CHANNELS = BOUNDARY_VARS["all"]

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
    parser.add_argument("--val-end", default="2012-10-01")

    parser.add_argument(
        "--time-chunk",
        type=int,
        default=1,
        help="Chunk size along time for packed prognostic/boundary arrays.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_name(args: argparse.Namespace) -> str:
    start = args.train_start.replace("-", "")
    end = args.val_end.replace("-", "")
    return (
        f"LLC4320_face{args.face}_"
        f"i{args.i_start}-{args.i_end}_"
        f"j{args.j_start}-{args.j_end}_"
        f"trainval_ready_{start}_{end}_t{args.time_chunk}.zarr"
    )


def build_output_path(args: argparse.Namespace) -> Path:
    output_name = args.output_name or default_output_name(args)
    if not output_name.endswith(".zarr"):
        output_name = f"{output_name}.zarr"
    return args.output_root / output_name


def remove_store(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def slice_llc_dim(ds: xr.Dataset, *, dim: str, start: int, end: int) -> xr.Dataset:
    if dim not in ds.dims:
        return ds
    if end <= start:
        raise ValueError(f"Invalid LLC slice for {dim}: [{start}:{end})")

    coord = ds.coords.get(dim)
    if coord is not None and coord.ndim == 1:
        coord_values = coord.to_numpy()
        min_coord = int(np.nanmin(coord_values))
        max_coord = int(np.nanmax(coord_values))
        if min_coord <= start and max_coord >= end - 1:
            return ds.sel({dim: slice(start, end - 1)})

    size = int(ds.sizes[dim])
    if 0 <= start < end <= size:
        return ds.isel({dim: slice(start, end)})

    raise ValueError(
        f"Requested LLC slice {dim}=[{start}:{end}) is incompatible with "
        f"dimension size/coords."
    )


def slice_patch(ds: xr.Dataset, args: argparse.Namespace) -> xr.Dataset:
    if "face" in ds.dims:
        ds = ds.sel(face=args.face, drop=True)
    else:
        logger.info(
            "No face dimension in source; assuming it is already a face-sliced patch."
        )

    ds = slice_llc_dim(ds, dim="i", start=args.i_start, end=args.i_end)
    ds = slice_llc_dim(ds, dim="j", start=args.j_start, end=args.j_end)
    ds = slice_llc_dim(ds, dim="i_g", start=args.i_start, end=args.i_end)
    ds = slice_llc_dim(ds, dim="j_g", start=args.j_start, end=args.j_end)
    return ds


def stats_var_name(channel_name: str, stats: xr.Dataset) -> str:
    if channel_name in stats.data_vars:
        return channel_name
    if "_" not in channel_name:
        raise KeyError(f"Could not find stats variable for {channel_name}")
    base, level = channel_name.rsplit("_", 1)
    candidate = f"{base}_lev_{level}"
    if candidate not in stats.data_vars:
        raise KeyError(f"Could not find stats variable for {channel_name} ({candidate})")
    return candidate


def select_train_val_times(ds: xr.Dataset, args: argparse.Namespace) -> tuple[xr.Dataset, int, int]:
    train = ds.sel(time=slice(args.train_start, args.train_end))
    val = ds.sel(time=slice(args.val_start, args.val_end))

    if train.sizes.get("time", 0) == 0:
        raise ValueError("Train time selection is empty.")
    if val.sizes.get("time", 0) == 0:
        raise ValueError("Val time selection is empty.")

    combined = xr.concat([train, val], dim="time")
    _, keep = np.unique(combined.time.values, return_index=True)
    combined = combined.isel(time=np.sort(keep)).sortby("time")
    return combined, int(train.sizes["time"]), int(val.sizes["time"])


def standardize_spatial_dims(da_in: xr.DataArray) -> xr.DataArray:
    rename_map = {}
    if "k" in da_in.dims:
        rename_map["k"] = "lev"
    if "j" in da_in.dims:
        rename_map["j"] = "y"
    if "j_g" in da_in.dims:
        rename_map["j_g"] = "y"
    if "i" in da_in.dims:
        rename_map["i"] = "x"
    if "i_g" in da_in.dims:
        rename_map["i_g"] = "x"
    if rename_map:
        da_in = da_in.rename(rename_map)
    return da_in


def extract_channel(ds: xr.Dataset, channel_name: str) -> xr.DataArray:
    if channel_name in ds.data_vars:
        da_out = ds[channel_name]
    else:
        base_name, level = channel_name.rsplit("_", 1)
        if base_name not in ds.data_vars:
            raise KeyError(f"Missing source variable for channel {channel_name}")
        source_da = ds[base_name]
        lev_dim = "k" if "k" in source_da.dims else "lev"
        da_out = source_da.isel({lev_dim: int(level)})

    da_out = standardize_spatial_dims(da_out)
    if "time" not in da_out.dims:
        raise ValueError(f"Channel {channel_name} is missing a time dimension")
    return da_out.transpose("time", "y", "x")


def extract_xy_coords(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    for var_name in ["Eta", "Theta", "Salt", "oceQnet", "U", "V"]:
        if var_name not in ds.data_vars:
            continue
        da_var = standardize_spatial_dims(ds[var_name])
        if "y" in da_var.coords and "x" in da_var.coords:
            return da_var["y"].to_numpy(), da_var["x"].to_numpy()

    sample = standardize_spatial_dims(next(iter(ds.data_vars.values())))
    y_size = int(sample.sizes["y"])
    x_size = int(sample.sizes["x"])
    return np.arange(y_size, dtype=np.int32), np.arange(x_size, dtype=np.int32)


def build_packed_data_array(
    ds: xr.Dataset,
    channel_names: list[str],
    *,
    dim_name: str,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    time_chunk: int,
) -> xr.DataArray:
    channel_arrays: list[da.Array] = []
    for channel_name in channel_names:
        da_channel = extract_channel(ds, channel_name)
        channel_arrays.append(da_channel.data[:, None, :, :].astype(np.float32))

    packed = da.concatenate(channel_arrays, axis=1)
    packed_da = xr.DataArray(
        packed,
        dims=("time", dim_name, "y", "x"),
        coords={
            "time": ds.time,
            dim_name: np.arange(len(channel_names), dtype=np.int32),
            "y": y_coords,
            "x": x_coords,
        },
        name="packed",
    )
    return packed_da.chunk(
        {
            "time": time_chunk,
            dim_name: len(channel_names),
            "y": len(y_coords),
            "x": len(x_coords),
        }
    )


def build_flat_stats(
    stats: xr.Dataset,
    channel_names: list[str],
    *,
    dim_name: str,
) -> xr.DataArray:
    values = np.asarray(
        [stats[stats_var_name(name, stats)].item() for name in channel_names],
        dtype=np.float32,
    )
    return xr.DataArray(
        values,
        dims=(dim_name,),
        coords={dim_name: np.arange(len(channel_names), dtype=np.int32)},
    )


def extract_mask_cube(ds: xr.Dataset) -> np.ndarray:
    if "mask_c" in ds.data_vars:
        mask = ds["mask_c"]
    elif "wetmask" in ds.data_vars:
        mask = ds["wetmask"]
    else:
        raise KeyError("Source dataset must contain either mask_c or wetmask.")

    mask = standardize_spatial_dims(mask)
    lev_dim = "lev" if "lev" in mask.dims else "k"
    return mask.transpose(lev_dim, "y", "x").to_numpy().astype(bool, copy=False)


def build_flat_masks(
    ds: xr.Dataset,
    prognostic_channel_names: list[str],
    boundary_channel_names: list[str],
    *,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray]:
    mask_cube = extract_mask_cube(ds)
    surface_mask = mask_cube[0]

    prognostic_masks = []
    for channel_name in prognostic_channel_names:
        if "_" in channel_name:
            _, level = channel_name.rsplit("_", 1)
            prognostic_masks.append(mask_cube[int(level)])
        else:
            prognostic_masks.append(surface_mask)

    boundary_masks = [surface_mask for _ in boundary_channel_names]

    prognostic_mask_da = xr.DataArray(
        np.stack(prognostic_masks, axis=0),
        dims=("prognostic_channel", "y", "x"),
        coords={
            "prognostic_channel": np.arange(
                len(prognostic_channel_names), dtype=np.int32
            ),
            "y": y_coords,
            "x": x_coords,
        },
    )
    boundary_mask_da = xr.DataArray(
        np.stack(boundary_masks, axis=0),
        dims=("boundary_channel", "y", "x"),
        coords={
            "boundary_channel": np.arange(len(boundary_channel_names), dtype=np.int32),
            "y": y_coords,
            "x": x_coords,
        },
    )
    return prognostic_mask_da, boundary_mask_da


def estimate_output_size_bytes(
    n_time: int,
    y_size: int,
    x_size: int,
    prognostic_channels: int,
    boundary_channels: int,
) -> int:
    float32_size = 4
    bool_size = 1
    prognostic_bytes = n_time * prognostic_channels * y_size * x_size * float32_size
    boundary_bytes = n_time * boundary_channels * y_size * x_size * float32_size
    stats_bytes = 2 * (
        (prognostic_channels + boundary_channels) * float32_size
    )  # mean + std
    mask_bytes = (
        prognostic_channels + boundary_channels
    ) * y_size * x_size * bool_size
    return prognostic_bytes + boundary_bytes + stats_bytes + mask_bytes


def build_training_ready_dataset(
    data: xr.Dataset,
    means: xr.Dataset,
    stds: xr.Dataset,
    args: argparse.Namespace,
) -> xr.Dataset:
    data, train_count, val_count = select_train_val_times(data, args)
    y_coords, x_coords = extract_xy_coords(data)

    prognostic = build_packed_data_array(
        data,
        DEFAULT_PROGNOSTIC_CHANNELS,
        dim_name="prognostic_channel",
        y_coords=y_coords,
        x_coords=x_coords,
        time_chunk=args.time_chunk,
    )
    boundary = build_packed_data_array(
        data,
        DEFAULT_BOUNDARY_CHANNELS,
        dim_name="boundary_channel",
        y_coords=y_coords,
        x_coords=x_coords,
        time_chunk=args.time_chunk,
    )
    prognostic_mean = build_flat_stats(
        means, DEFAULT_PROGNOSTIC_CHANNELS, dim_name="prognostic_channel"
    )
    prognostic_std = build_flat_stats(
        stds, DEFAULT_PROGNOSTIC_CHANNELS, dim_name="prognostic_channel"
    )
    boundary_mean = build_flat_stats(
        means, DEFAULT_BOUNDARY_CHANNELS, dim_name="boundary_channel"
    )
    boundary_std = build_flat_stats(
        stds, DEFAULT_BOUNDARY_CHANNELS, dim_name="boundary_channel"
    )
    prognostic_mask, boundary_mask = build_flat_masks(
        data,
        DEFAULT_PROGNOSTIC_CHANNELS,
        DEFAULT_BOUNDARY_CHANNELS,
        y_coords=y_coords,
        x_coords=x_coords,
    )

    ds_out = xr.Dataset(
        data_vars={
            "prognostic": prognostic,
            "boundary": boundary,
            "prognostic_mean": prognostic_mean,
            "prognostic_std": prognostic_std,
            "boundary_mean": boundary_mean,
            "boundary_std": boundary_std,
            "prognostic_mask": prognostic_mask,
            "boundary_mask": boundary_mask,
        },
        coords={
            "time": prognostic.time,
            "prognostic_channel": prognostic.prognostic_channel,
            "boundary_channel": boundary.boundary_channel,
            "y": prognostic.y,
            "x": prognostic.x,
        },
        attrs={
            "cache_format": "llc-train-ready-v1",
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
            "prognostic_channel_names_json": json.dumps(
                DEFAULT_PROGNOSTIC_CHANNELS
            ),
            "boundary_channel_names_json": json.dumps(DEFAULT_BOUNDARY_CHANNELS),
        },
    )

    total_bytes = estimate_output_size_bytes(
        n_time=int(ds_out.sizes["time"]),
        y_size=int(ds_out.sizes["y"]),
        x_size=int(ds_out.sizes["x"]),
        prognostic_channels=len(DEFAULT_PROGNOSTIC_CHANNELS),
        boundary_channels=len(DEFAULT_BOUNDARY_CHANNELS),
    )
    logger.info(
        "Selected %d train times + %d val times = %d total times",
        train_count,
        val_count,
        int(ds_out.sizes["time"]),
    )
    logger.info(
        "Packed prognostic shape=%s boundary shape=%s estimated raw size=%.2f GiB",
        tuple(int(ds_out.sizes[d]) for d in ("time", "prognostic_channel", "y", "x")),
        tuple(int(ds_out.sizes[d]) for d in ("time", "boundary_channel", "y", "x")),
        total_bytes / (1024**3),
    )
    return ds_out


def build_encoding(ds_out: xr.Dataset, args: argparse.Namespace) -> dict[str, dict[str, object]]:
    y_size = int(ds_out.sizes["y"])
    x_size = int(ds_out.sizes["x"])
    prog_channels = int(ds_out.sizes["prognostic_channel"])
    bound_channels = int(ds_out.sizes["boundary_channel"])
    time_size = int(ds_out.sizes["time"])

    encoding: dict[str, dict[str, object]] = {
        "prognostic": {
            "compressor": None,
            "chunks": (args.time_chunk, prog_channels, y_size, x_size),
        },
        "boundary": {
            "compressor": None,
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
        "time": {"compressor": None, "chunks": (min(time_size, 1024),)},
        "prognostic_channel": {"compressor": None, "chunks": (prog_channels,)},
        "boundary_channel": {"compressor": None, "chunks": (bound_channels,)},
        "y": {"compressor": None, "chunks": (y_size,)},
        "x": {"compressor": None, "chunks": (x_size,)},
    }
    return encoding


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

    selected_vars = sorted(required_vars | {"mask_c", "wetmask"} & set(data.data_vars))
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
    logger.info("Output path: %s", output_path)

    if args.dry_run:
        logger.info("Dry run requested; not writing any data.")
        return

    encoding = build_encoding(ds_out, args)
    logger.info("Writing temporary store: %s", tmp_path)
    ds_out.to_zarr(tmp_path, mode="w", consolidated=True, encoding=encoding)

    logger.info("Moving completed store to: %s", output_path)
    tmp_path.replace(output_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
