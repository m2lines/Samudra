# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare constant-volume surface time/space shard envelopes."""

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


def timed_read(array, fixture, selection) -> float:  # noqa: ANN001, ANN202
    started = time.perf_counter()
    actual = np.asarray(array[selection])
    np.testing.assert_equal(actual, fixture[selection])
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--time-shard", type=int, required=True)
    parser.add_argument("--outer", type=int, required=True)
    args = parser.parse_args()
    if 24 % args.time_shard or 4320 % args.outer or args.outer % 360:
        parser.error("candidate must divide the 24 x 4320 x 4320 fixture")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    fixture = np.load(args.fixture / "Eta.npy", mmap_mode="r")
    candidate = CODEC_CANDIDATES[args.candidate]
    store_path = args.output / "candidate.zarr"
    root = zarr.create_group(store=str(store_path), zarr_format=3)
    array = root.create_array(
        "Eta",
        shape=fixture.shape,
        chunks=(1, 360, 360),
        shards=(args.time_shard, args.outer, args.outer),
        dtype=fixture.dtype,
        compressors=[codec(candidate)],
        dimension_names=("time", "j", "i"),
    )
    started = time.perf_counter()
    array[:] = fixture
    encode_seconds = time.perf_counter() - started
    selections = {
        "tile_one_time": (0, slice(1424, 2176), slice(1424, 2176)),
        "full_face_one_time": (0, slice(None), slice(None)),
        "point_all_times": (slice(None), slice(2000, 2001), slice(2000, 2001)),
        "crop_all_times": (slice(None), slice(1800, 2100), slice(1900, 2400)),
    }
    runs = {
        name: [timed_read(array, fixture, selection) for _ in range(3)]
        for name, selection in selections.items()
    }
    physical_objects, physical_bytes = tree_size(store_path)
    result = {
        "candidate": f"surface-timespace-time-{args.time_shard}-outer-{args.outer}",
        "codec": candidate[0],
        "time_shard": args.time_shard,
        "outer": args.outer,
        "inner": 360,
        "encode_seconds": encode_seconds,
        "read_medians": {
            name: statistics.median(values) for name, values in runs.items()
        },
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
