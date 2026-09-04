# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Select a logical depth chunk for the 52-interface-level W field."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr  # type: ignore[import-untyped]
from zarr.codecs import BloscCodec  # type: ignore[import-untyped]


def elapsed(read: Any, repeats: int = 3) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        np.asarray(read())
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def tree_size(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def candidate(
    root: Path, values: np.ndarray, depth: int, shard_depth: int
) -> dict[str, Any]:
    path = root / f"depth-{depth}"
    if path.exists():
        shutil.rmtree(path)
    array = zarr.create_array(
        store=str(path),
        shape=values.shape,
        chunks=(1, depth, 1, 120, 120),
        shards=(2, shard_depth, 1, 1440, 1440),
        dtype="float32",
        compressors=[BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")],
        dimension_names=("time", "k_p1", "face", "j", "i"),
    )
    started = time.perf_counter()
    array[:] = values
    encode_seconds = time.perf_counter() - started
    np.testing.assert_equal(array[:], values)
    objects, physical_bytes = tree_size(path)
    return {
        "logical_depth": depth,
        "physical_depth": shard_depth,
        "objects": objects,
        "physical_bytes": physical_bytes,
        "encode_seconds": encode_seconds,
        "full_depth_tile_seconds": elapsed(lambda: array[0, :, 0, 344:1096, 344:1096]),
        "single_depth_crop_seconds": elapsed(lambda: array[0, 25, 0, 480:960, 480:960]),
        "seventeen_depth_crop_seconds": elapsed(
            lambda: array[0, 17:34, 0, 480:960, 480:960]
        ),
        "exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = zarr.open_array(store=args.source, path="W", mode="r")
    values = np.asarray(source[0:2, :, 1:2, 1440:2880, 1440:2880])
    results = [
        candidate(args.output, values, depth, shard_depth)
        for depth, shard_depth in ((13, 52), (17, 51), (17, 68), (26, 52))
    ]
    smallest = min(item["physical_bytes"] for item in results)
    eligible = [item for item in results if item["physical_bytes"] <= 1.05 * smallest]
    winner = min(
        eligible,
        key=lambda item: (
            item["full_depth_tile_seconds"]
            * item["single_depth_crop_seconds"]
            * item["seventeen_depth_crop_seconds"]
        )
        ** (1 / 3),
    )
    print(
        json.dumps(
            {
                "fixture_shape": list(values.shape),
                "decoded_bytes": values.nbytes,
                "candidates": results,
                "winner": winner,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
