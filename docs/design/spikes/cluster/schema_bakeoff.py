# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Write and read one real-data sharded Zarr candidate."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]
from zarr.codecs import BloscCodec, ZstdCodec  # type: ignore[import-untyped]

FIXTURE_OFFSET = 720
GLOBAL_SIZE = 4320
TILES = (
    (1424, 2176, 1424, 2176),
    (1424, 2176, 2144, 2896),
    (2144, 2896, 1424, 2176),
    (2144, 2896, 2144, 2896),
)
UNION = (1424, 2896, 1424, 2896)

CODEC_CANDIDATES = (
    ("blosc-zstd-bitshuffle-1", "blosc", "zstd", 1, "bitshuffle"),
    ("blosc-zstd-bitshuffle-3", "blosc", "zstd", 3, "bitshuffle"),
    ("blosc-zstd-bitshuffle-5", "blosc", "zstd", 5, "bitshuffle"),
    ("blosc-lz4-bitshuffle-5", "blosc", "lz4", 5, "bitshuffle"),
    ("blosc-lz4-shuffle-5", "blosc", "lz4", 5, "shuffle"),
    ("zstd-1", "zstd", "zstd", 1, "none"),
    ("zstd-3", "zstd", "zstd", 3, "none"),
)


def codec(candidate: tuple[str, str, str, int, str]):  # noqa: ANN202
    _, family, compressor, level, shuffle = candidate
    if family == "blosc":
        return BloscCodec(cname=compressor, clevel=level, shuffle=shuffle)
    return ZstdCodec(level=level)


def tree_size(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def local_bounds(bounds: tuple[int, int, int, int]) -> tuple[slice, slice]:
    j0, j1, i0, i1 = bounds
    return (
        slice(j0 - FIXTURE_OFFSET, j1 - FIXTURE_OFFSET),
        slice(i0 - FIXTURE_OFFSET, i1 - FIXTURE_OFFSET),
    )


def write_array(
    root,
    *,
    name: str,
    fixture: np.ndarray,
    candidate: tuple[str, str, str, int, str],
    depth_inner: int,
    inner: int,
    outer: int,
    time_shard: int,
) -> tuple[Any, float]:
    array = root.create_array(
        name,
        shape=(fixture.shape[0], fixture.shape[1], GLOBAL_SIZE, GLOBAL_SIZE),
        chunks=(1, depth_inner, inner, inner),
        shards=(time_shard, fixture.shape[1], outer, outer),
        dtype=fixture.dtype,
        fill_value=np.nan,
        compressors=[codec(candidate)],
        dimension_names=("time", "k", "j", "i"),
    )
    started = time.perf_counter()
    region_start = FIXTURE_OFFSET
    region_stop = FIXTURE_OFFSET + fixture.shape[-1]
    for time_start in range(0, fixture.shape[0], time_shard):
        time_stop = min(time_start + time_shard, fixture.shape[0])
        for shard_j0 in range(0, GLOBAL_SIZE, outer):
            j0 = max(shard_j0, region_start)
            j1 = min(shard_j0 + outer, region_stop)
            if j0 >= j1:
                continue
            for shard_i0 in range(0, GLOBAL_SIZE, outer):
                i0 = max(shard_i0, region_start)
                i1 = min(shard_i0 + outer, region_stop)
                if i0 >= i1:
                    continue
                array[time_start:time_stop, :, j0:j1, i0:i1] = fixture[
                    time_start:time_stop,
                    :,
                    j0 - region_start : j1 - region_start,
                    i0 - region_start : i1 - region_start,
                ]
    return array, time.perf_counter() - started


def read_and_validate(
    arrays: dict[str, Any], fixtures: dict[str, np.ndarray], bounds, time_index: int
) -> float:  # noqa: ANN001
    started = time.perf_counter()
    for name, array in arrays.items():
        actual = np.asarray(
            array[time_index, :, bounds[0] : bounds[1], bounds[2] : bounds[3]]
        )
        j_slice, i_slice = local_bounds(bounds)
        expected = np.asarray(fixtures[name][time_index, :, j_slice, i_slice])
        np.testing.assert_equal(actual, expected)
    return time.perf_counter() - started


def read_general_selection(
    arrays: dict[str, Any],
    fixtures: dict[str, np.ndarray],
    *,
    depth: slice,
    bounds: tuple[int, int, int, int],
    time_index: int,
) -> float:
    """Measure a non-Samudra-specific, contiguous Xarray-style selection."""
    started = time.perf_counter()
    j_slice, i_slice = local_bounds(bounds)
    for name, array in arrays.items():
        actual = np.asarray(
            array[
                time_index,
                depth,
                bounds[0] : bounds[1],
                bounds[2] : bounds[3],
            ]
        )
        expected = np.asarray(fixtures[name][time_index, depth, j_slice, i_slice])
        np.testing.assert_equal(actual, expected)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--depth-inner", type=int, default=51)
    parser.add_argument("--inner", type=int, default=72)
    parser.add_argument("--outer", type=int, default=1440)
    parser.add_argument("--time-shard", type=int, default=1)
    parser.add_argument("--arrays", nargs="+", default=["U", "Theta"])
    parser.add_argument("--time-count", type=int, default=2)
    parser.add_argument("--read-repeats", type=int, default=3)
    args = parser.parse_args()
    candidate = CODEC_CANDIDATES[args.candidate]
    for inner_or_outer in (args.inner, args.outer):
        if GLOBAL_SIZE % inner_or_outer:
            parser.error(f"{inner_or_outer} must divide {GLOBAL_SIZE}")
    if args.outer % args.inner:
        parser.error("inner chunk width must divide outer shard width")
    if 51 % args.depth_inner:
        parser.error("depth inner chunk must divide 51")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    zarr.config.set({"async.concurrency": 4})
    fixtures = {
        name: np.load(args.fixture / f"{name}.npy", mmap_mode="r")[: args.time_count]
        for name in args.arrays
    }
    args.output.mkdir(parents=True)
    store_path = args.output / "candidate.zarr"
    root = zarr.create_group(store=str(store_path), zarr_format=3)
    arrays: dict[str, Any] = {}
    encode_seconds = {}
    for name, fixture in fixtures.items():
        arrays[name], encode_seconds[name] = write_array(
            root,
            name=name,
            fixture=fixture,
            candidate=candidate,
            depth_inner=args.depth_inner,
            inner=args.inner,
            outer=args.outer,
            time_shard=args.time_shard,
        )

    independent_runs = []
    union_runs = []
    for repeat in range(args.read_repeats):
        time_index = repeat % args.time_count
        independent_runs.append(
            sum(
                read_and_validate(arrays, fixtures, tile, time_index=time_index)
                for tile in TILES
            )
        )
        union_runs.append(
            read_and_validate(arrays, fixtures, UNION, time_index=time_index)
        )
    physical_objects, physical_bytes = tree_size(store_path)
    general_reads = {
        "single_depth_512x384_seconds": read_general_selection(
            arrays,
            fixtures,
            depth=slice(20, 21),
            bounds=(1500, 2012, 1700, 2084),
            time_index=0,
        ),
        "seventeen_depth_300x500_seconds": read_general_selection(
            arrays,
            fixtures,
            depth=slice(17, 34),
            bounds=(1800, 2100, 1900, 2400),
            time_index=1 % args.time_count,
        ),
        "all_depth_300x500_seconds": read_general_selection(
            arrays,
            fixtures,
            depth=slice(None),
            bounds=(1800, 2100, 1900, 2400),
            time_index=1 % args.time_count,
        ),
    }
    logical_bytes = sum(fixture.nbytes for fixture in fixtures.values())
    result = {
        "candidate": candidate[0],
        "candidate_index": args.candidate,
        "arrays": args.arrays,
        "time_count": args.time_count,
        "depth_inner": args.depth_inner,
        "inner": args.inner,
        "outer": args.outer,
        "time_shard": args.time_shard,
        "encode_seconds": encode_seconds,
        "independent_total_seconds": independent_runs,
        "independent_median_seconds": statistics.median(independent_runs),
        "union_read_seconds": union_runs,
        "union_median_seconds": statistics.median(union_runs),
        "general_read_seconds": general_reads,
        "physical_objects": physical_objects,
        "physical_bytes": physical_bytes,
        "fixture_logical_bytes": logical_bytes,
        "physical_to_fixture_ratio": physical_bytes / logical_bytes,
        "final_rss_bytes": psutil.Process().memory_info().rss,
        "exact_validation": True,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
