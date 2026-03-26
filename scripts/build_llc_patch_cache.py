#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "xarray[io]>=2025.1.2",
#   "dask>=2025.2",
#   "zarr<3",
# ]
# ///
"""Build a local LLC patch cache store with explicit rechunking.

This script supports:
- writing only a selected subset of variables (faster cache generation), and
- resumable writes via a temporary zarr store.
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
from pathlib import Path

import xarray as xr
import zarr

DEFAULT_SOURCE = Path("/orcd/data/abodner/003/LLC4320/LLC4320")
DEFAULT_OUTPUT_ROOT = Path("/orcd/data/abodner/002/cody/LLC_patch")
DEFAULT_REQUIRED_VARS = [
    "U",
    "V",
    "Theta",
    "Salt",
    "Eta",
    "oceTAUX",
    "oceTAUY",
    "oceQnet",
    "mask_c",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def parse_var_list(values: list[str] | None) -> list[str]:
    if values is None:
        return []

    parsed: list[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                parsed.append(token)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-name", default=None)

    parser.add_argument("--face", type=int, default=1)
    parser.add_argument("--i-start", type=int, default=2880)
    parser.add_argument("--i-end", type=int, default=3600)
    parser.add_argument("--j-start", type=int, default=720)
    parser.add_argument("--j-end", type=int, default=1440)

    parser.add_argument("--time-chunk", type=int, default=1)
    parser.add_argument("--i-chunk", type=int, default=720)
    parser.add_argument("--j-chunk", type=int, default=720)
    parser.add_argument("--k-chunk", type=int, default=51)
    parser.add_argument("--kp1-chunk", type=int, default=52)

    parser.add_argument(
        "--include-vars",
        nargs="+",
        default=DEFAULT_REQUIRED_VARS,
        help=(
            "Variables to cache (space or comma separated). "
            "Ignored when --include-all-vars is set."
        ),
    )
    parser.add_argument(
        "--include-all-vars",
        action="store_true",
        help="Cache all data variables from the sliced patch.",
    )

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing .tmp store when present (default).",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and fail if .tmp exists (unless --overwrite).",
    )
    args = parser.parse_args()
    args.include_vars = parse_var_list(args.include_vars)
    return args


def default_output_name(args: argparse.Namespace) -> str:
    return (
        f"LLC4320_face{args.face}_"
        f"i{args.i_start}-{args.i_end}_"
        f"j{args.j_start}-{args.j_end}.zarr"
    )


def build_output_path(args: argparse.Namespace) -> Path:
    output_name = args.output_name or default_output_name(args)
    if not output_name.endswith(".zarr"):
        output_name = f"{output_name}.zarr"
    return args.output_root / output_name


def validate_args(args: argparse.Namespace) -> None:
    if args.i_end <= args.i_start:
        raise ValueError("i-end must be greater than i-start")
    if args.j_end <= args.j_start:
        raise ValueError("j-end must be greater than j-start")
    if args.time_chunk <= 0:
        raise ValueError("time-chunk must be positive")
    if not args.include_all_vars and not args.include_vars:
        raise ValueError("include-vars is empty; pass --include-all-vars or non-empty --include-vars")


def slice_patch(ds: xr.Dataset, args: argparse.Namespace) -> xr.Dataset:
    if "face" in ds.dims:
        ds = ds.sel(face=args.face, drop=True)

    indexers: dict[str, slice] = {}
    if "i" in ds.dims:
        indexers["i"] = slice(args.i_start, args.i_end)
    if "j" in ds.dims:
        indexers["j"] = slice(args.j_start, args.j_end)
    if "i_g" in ds.dims:
        indexers["i_g"] = slice(args.i_start, args.i_end)
    if "j_g" in ds.dims:
        indexers["j_g"] = slice(args.j_start, args.j_end)

    if indexers:
        ds = ds.isel(**indexers)
    return ds


def get_chunk_spec(ds: xr.Dataset, args: argparse.Namespace) -> dict[str, int]:
    chunk_spec = {
        "time": args.time_chunk,
        "i": args.i_chunk,
        "j": args.j_chunk,
        "i_g": args.i_chunk,
        "j_g": args.j_chunk,
        "k": args.k_chunk,
        "k_l": args.k_chunk,
        "k_u": args.k_chunk,
        "k_p1": args.kp1_chunk,
    }
    return {k: v for k, v in chunk_spec.items() if k in ds.dims}


def rechunk(ds: xr.Dataset, chunk_spec: dict[str, int]) -> xr.Dataset:
    if chunk_spec:
        ds = ds.chunk(chunk_spec)
    return ds


def remove_store(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def is_zarr_array_dir(path: Path) -> bool:
    return path.is_dir() and (path / ".zarray").exists()


def count_data_chunk_files(array_dir: Path) -> int:
    if not array_dir.exists() or not array_dir.is_dir():
        return 0
    return sum(1 for f in array_dir.iterdir() if f.is_file() and not f.name.startswith("."))


def expected_chunks(da: xr.DataArray, chunk_spec: dict[str, int]) -> int:
    n_chunks = 1
    for dim, size in da.sizes.items():
        chunk = chunk_spec.get(dim, size)
        n_chunks *= math.ceil(size / chunk)
    return n_chunks


def select_vars(ds: xr.Dataset, args: argparse.Namespace) -> xr.Dataset:
    if args.include_all_vars:
        selected = list(ds.data_vars)
    else:
        missing = [v for v in args.include_vars if v not in ds.data_vars]
        if missing:
            raise ValueError(f"Requested variables missing from source: {missing}")
        selected = args.include_vars

    logger.info("Selected %d/%d data vars for cache", len(selected), len(ds.data_vars))
    logger.info("Variables: %s", ", ".join(selected))
    return ds[selected]


def prune_unselected_arrays(tmp_output: Path, keep_names: set[str]) -> None:
    if not tmp_output.exists():
        return
    for child in tmp_output.iterdir():
        if is_zarr_array_dir(child) and child.name not in keep_names:
            logger.info("Removing unselected array from temp store: %s", child.name)
            remove_store(child)



def main() -> None:
    logger.info("Initializing")
    args = parse_args()
    validate_args(args)

    source = args.source
    output = build_output_path(args)
    tmp_output = output.with_name(f"{output.name}.tmp")

    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        if output.exists():
            logger.info("Removing existing output store: %s", output)
            remove_store(output)
        if tmp_output.exists():
            logger.info("Removing existing temporary store: %s", tmp_output)
            remove_store(tmp_output)
    else:
        if output.exists():
            logger.info("Output already exists and overwrite=false: %s", output)
            logger.info("Nothing to do.")
            return
        if tmp_output.exists() and not args.resume:
            raise FileExistsError(
                f"Temporary output already exists: {tmp_output}. "
                "Pass --resume (default) or --overwrite."
            )

    logger.info("Opening source: %s", source)
    ds = xr.open_zarr(str(source), consolidated=False, chunks={})

    ds = slice_patch(ds, args)
    ds = select_vars(ds, args)

    chunk_spec = get_chunk_spec(ds, args)
    ds = rechunk(ds, chunk_spec)

    logger.info("Final dataset summary:")
    logger.info(ds)
    logger.info("Writing output to: %s", output)

    if args.dry_run:
        logger.info("Dry run enabled; not writing output.")
        return

    # If resuming, prune arrays we do not intend to keep.
    keep_arrays = set(ds.data_vars).union(set(ds.coords))
    if tmp_output.exists() and args.resume:
        zmetadata = tmp_output / ".zmetadata"
        if zmetadata.exists():
            zmetadata.unlink()
        prune_unselected_arrays(tmp_output, keep_arrays)

    store_initialized = (tmp_output / ".zgroup").exists()
    data_vars = list(ds.data_vars)

    for idx, var_name in enumerate(data_vars, start=1):
        var_da = ds[var_name]
        var_dir = tmp_output / var_name
        expected = expected_chunks(var_da, chunk_spec)
        current = count_data_chunk_files(var_dir)

        if args.resume and expected > 0 and current >= expected:
            logger.info(
                "[%d/%d] Skipping complete variable %s (%d/%d chunks)",
                idx,
                len(data_vars),
                var_name,
                current,
                expected,
            )
            continue

        if args.resume and 0 < current < expected:
            logger.info(
                "[%d/%d] Rewriting incomplete variable %s (%d/%d chunks)",
                idx,
                len(data_vars),
                var_name,
                current,
                expected,
            )
            remove_store(var_dir)

        mode = "a" if store_initialized else "w"
        logger.info(
            "[%d/%d] Writing variable %s (expected_chunks=%d, mode=%s)",
            idx,
            len(data_vars),
            var_name,
            expected,
            mode,
        )
        ds[[var_name]].to_zarr(str(tmp_output), mode=mode, consolidated=False)
        store_initialized = True

    incomplete: list[tuple[str, int, int]] = []
    for var_name in data_vars:
        expected = expected_chunks(ds[var_name], chunk_spec)
        current = count_data_chunk_files(tmp_output / var_name)
        if current < expected:
            incomplete.append((var_name, current, expected))

    if incomplete:
        msg = ", ".join(
            f"{name}:{current}/{expected}" for name, current, expected in incomplete[:10]
        )
        raise RuntimeError(
            "Temporary store is incomplete after write. "
            f"Incomplete vars (sample): {msg}"
        )

    logger.info("Consolidating metadata for temporary store")
    zarr.consolidate_metadata(str(tmp_output))

    if output.exists():
        if args.overwrite:
            remove_store(output)
        else:
            raise FileExistsError(
                f"Output already exists: {output}. Pass --overwrite to replace it."
            )

    tmp_output.replace(output)
    logger.info("Done. Cache store created at: %s", output)


if __name__ == "__main__":
    main()
