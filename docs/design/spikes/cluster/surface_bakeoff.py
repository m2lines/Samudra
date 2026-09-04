# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Write and read one real-data surface-array shard candidate."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import psutil  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]
from schema_bakeoff import CODEC_CANDIDATES, TILES, UNION, codec, tree_size


def read_and_validate(array, fixture, bounds, time_index: int) -> float:  # noqa: ANN001, ANN202
    started = time.perf_counter()
    actual = np.asarray(array[time_index, bounds[0] : bounds[1], bounds[2] : bounds[3]])
    expected = np.asarray(
        fixture[time_index, bounds[0] : bounds[1], bounds[2] : bounds[3]]
    )
    np.testing.assert_equal(actual, expected)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--inner", type=int, required=True)
    parser.add_argument("--outer", type=int, required=True)
    parser.add_argument("--time-shard", type=int, required=True)
    parser.add_argument("--read-repeats", type=int, default=3)
    args = parser.parse_args()
    if 4320 % args.inner or 4320 % args.outer or args.outer % args.inner:
        parser.error("inner and outer widths must form a divisor chain of 4320")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    candidate = CODEC_CANDIDATES[args.candidate]
    fixture = np.load(args.fixture / "Eta.npy", mmap_mode="r")
    args.output.mkdir(parents=True)
    store_path = args.output / "candidate.zarr"
    root = zarr.create_group(store=str(store_path), zarr_format=3)
    array = root.create_array(
        "Eta",
        shape=fixture.shape,
        chunks=(1, args.inner, args.inner),
        shards=(args.time_shard, args.outer, args.outer),
        dtype=fixture.dtype,
        fill_value=np.nan,
        compressors=[codec(candidate)],
        dimension_names=("time", "j", "i"),
    )
    started = time.perf_counter()
    for t0 in range(0, fixture.shape[0], args.time_shard):
        t1 = min(t0 + args.time_shard, fixture.shape[0])
        for j0 in range(0, 4320, args.outer):
            for i0 in range(0, 4320, args.outer):
                array[t0:t1, j0 : j0 + args.outer, i0 : i0 + args.outer] = fixture[
                    t0:t1, j0 : j0 + args.outer, i0 : i0 + args.outer
                ]
    encode_seconds = time.perf_counter() - started

    independent_runs = []
    union_runs = []
    for repeat in range(args.read_repeats):
        time_index = repeat % fixture.shape[0]
        independent_runs.append(
            sum(read_and_validate(array, fixture, tile, time_index) for tile in TILES)
        )
        union_runs.append(read_and_validate(array, fixture, UNION, time_index))
    general_reads = {
        "arbitrary_512x384_seconds": read_and_validate(
            array, fixture, (1500, 2012, 1700, 2084), 0
        ),
        "full_face_seconds": read_and_validate(array, fixture, (0, 4320, 0, 4320), 0),
    }
    physical_objects, physical_bytes = tree_size(store_path)
    result = {
        "candidate": (
            f"surface-{candidate[0]}-inner-{args.inner}-outer-{args.outer}"
            f"-time-{args.time_shard}"
        ),
        "codec": candidate[0],
        "candidate_index": args.candidate,
        "inner": args.inner,
        "outer": args.outer,
        "time_shard": args.time_shard,
        "encode_seconds": {"Eta": encode_seconds},
        "independent_total_seconds": independent_runs,
        "independent_median_seconds": statistics.median(independent_runs),
        "union_read_seconds": union_runs,
        "union_median_seconds": statistics.median(union_runs),
        "general_read_seconds": general_reads,
        "physical_objects": physical_objects,
        "physical_bytes": physical_bytes,
        "fixture_logical_bytes": fixture.nbytes,
        "physical_to_fixture_ratio": physical_bytes / fixture.nbytes,
        "final_rss_bytes": psutil.Process().memory_info().rss,
        "exact_validation": True,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
