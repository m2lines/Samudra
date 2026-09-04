# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare physical time packing for analysis and training-like reads."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import zarr  # type: ignore[import-untyped]
from schema_bakeoff import CODEC_CANDIDATES, codec, tree_size


def timed_read(array, selection, expected) -> float:  # noqa: ANN001, ANN202
    started = time.perf_counter()
    actual = np.asarray(array[selection])
    np.testing.assert_equal(actual, expected)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--time-shard", type=int, required=True)
    args = parser.parse_args()
    if 24 % args.time_shard:
        parser.error("time shard must divide the 24-time fixture")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    candidate = CODEC_CANDIDATES[args.candidate]
    theta = np.load(args.fixture / "Theta.npy", mmap_mode="r")
    eta = np.load(args.fixture / "Eta.npy", mmap_mode="r")
    root = zarr.create_group(store=str(args.output / "candidate.zarr"), zarr_format=3)
    theta_out = root.create_array(
        "Theta",
        shape=theta.shape,
        chunks=(1, 17, 90, 90),
        shards=(args.time_shard, 51, 720, 720),
        dtype=theta.dtype,
        compressors=[codec(candidate)],
        dimension_names=("time", "k", "j", "i"),
    )
    eta_out = root.create_array(
        "Eta",
        shape=eta.shape,
        chunks=(1, 360, 360),
        shards=(args.time_shard, 720, 720),
        dtype=eta.dtype,
        compressors=[codec(candidate)],
        dimension_names=("time", "j", "i"),
    )
    started = time.perf_counter()
    theta_out[:] = theta
    eta_out[:] = eta
    encode_seconds = time.perf_counter() - started

    runs: dict[str, list[float]] = {
        "point_all_times": [],
        "crop_all_times": [],
        "full_depth_one_time": [],
        "two_training_times": [],
        "surface_crop_all_times": [],
    }
    for repeat in range(3):
        time_index = repeat
        runs["point_all_times"].append(
            timed_read(
                theta_out,
                (slice(None), slice(0, 1), slice(300, 301), slice(300, 301)),
                theta[:, 0:1, 300:301, 300:301],
            )
        )
        runs["crop_all_times"].append(
            timed_read(
                theta_out,
                (slice(None), slice(0, 17), slice(240, 480), slice(240, 480)),
                theta[:, 0:17, 240:480, 240:480],
            )
        )
        runs["full_depth_one_time"].append(
            timed_read(
                theta_out,
                (time_index, slice(None), slice(None), slice(None)),
                theta[time_index],
            )
        )
        runs["two_training_times"].append(
            timed_read(
                theta_out,
                (
                    slice(time_index, time_index + 2),
                    slice(None),
                    slice(None),
                    slice(None),
                ),
                theta[time_index : time_index + 2],
            )
        )
        runs["surface_crop_all_times"].append(
            timed_read(
                eta_out,
                (slice(None), slice(240, 480), slice(240, 480)),
                eta[:, 240:480, 240:480],
            )
        )

    physical_objects, physical_bytes = tree_size(args.output / "candidate.zarr")
    result = {
        "candidate": f"time-{args.time_shard}-{candidate[0]}",
        "codec": candidate[0],
        "time_shard": args.time_shard,
        "encode_seconds": encode_seconds,
        "read_seconds": runs,
        "read_medians": {
            name: statistics.median(values) for name, values in runs.items()
        },
        "physical_objects": physical_objects,
        "physical_bytes": physical_bytes,
        "fixture_logical_bytes": theta.nbytes + eta.nbytes,
        "exact_validation": True,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
