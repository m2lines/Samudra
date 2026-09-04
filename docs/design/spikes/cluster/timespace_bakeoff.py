# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare constant-volume physical time/space shard envelopes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import psutil  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]
from schema_bakeoff import CODEC_CANDIDATES, codec, tree_size

TILES = (
    (704, 1456, 704, 1456),
    (704, 1456, 1424, 2176),
    (1424, 2176, 704, 1456),
    (1424, 2176, 1424, 2176),
)
UNION = (704, 2176, 704, 2176)


def timed_read(array, fixture, time_selection, depth, bounds) -> float:  # noqa: ANN001, ANN202
    j0, j1, i0, i1 = bounds
    started = time.perf_counter()
    actual = np.asarray(array[time_selection, depth, j0:j1, i0:i1])
    expected = np.asarray(fixture[time_selection, depth, j0:j1, i0:i1])
    np.testing.assert_equal(actual, expected)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--time-shard", type=int, required=True)
    parser.add_argument("--outer", type=int, required=True)
    args = parser.parse_args()
    if args.outer % 120 or 8 % args.time_shard:
        parser.error("outer must divide by 120 and time shard must divide 8")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    fixture = np.load(args.fixture / "Theta.npy", mmap_mode="r")
    candidate = CODEC_CANDIDATES[args.candidate]
    store_path = args.output / "candidate.zarr"
    root = zarr.create_group(store=str(store_path), zarr_format=3)
    array = root.create_array(
        "Theta",
        shape=fixture.shape,
        chunks=(1, 17, 120, 120),
        shards=(args.time_shard, 51, args.outer, args.outer),
        dtype=fixture.dtype,
        compressors=[codec(candidate)],
        dimension_names=("time", "k", "j", "i"),
    )
    started = time.perf_counter()
    array[:] = fixture
    encode_seconds = time.perf_counter() - started

    independent = []
    union = []
    point_series = []
    crop_series = []
    for repeat in range(3):
        time_index = repeat
        independent.append(
            sum(
                timed_read(array, fixture, time_index, slice(None), tile)
                for tile in TILES
            )
        )
        union.append(timed_read(array, fixture, time_index, slice(None), UNION))
        point_series.append(
            timed_read(
                array,
                fixture,
                slice(None),
                slice(20, 21),
                (1000, 1001, 1000, 1001),
            )
        )
        crop_series.append(
            timed_read(
                array,
                fixture,
                slice(None),
                slice(17, 34),
                (900, 1200, 1000, 1500),
            )
        )

    physical_objects, physical_bytes = tree_size(store_path)
    result = {
        "candidate": f"timespace-time-{args.time_shard}-outer-{args.outer}",
        "codec": candidate[0],
        "time_shard": args.time_shard,
        "outer": args.outer,
        "inner": 120,
        "depth_inner": 17,
        "encode_seconds": encode_seconds,
        "independent_median_seconds": statistics.median(independent),
        "union_median_seconds": statistics.median(union),
        "point_series_median_seconds": statistics.median(point_series),
        "crop_series_median_seconds": statistics.median(crop_series),
        "physical_objects": physical_objects,
        "physical_bytes": physical_bytes,
        "fixture_logical_bytes": fixture.nbytes,
        "final_rss_bytes": psutil.Process().memory_info().rss,
        "exact_validation": True,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
