# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Measure a training-specific destination-ready halo sidecar control."""

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
from schema_bakeoff import CODEC_CANDIDATES, codec, tree_size

FIXTURE_OFFSET = 720
CORES = (
    (1440, 2160, 1440, 2160),
    (1440, 2160, 2160, 2880),
    (2160, 2880, 1440, 2160),
    (2160, 2880, 2160, 2880),
)


def expected_tile(fixture: np.ndarray, time_index: int, core, halo: int) -> np.ndarray:  # noqa: ANN001
    j0, j1, i0, i1 = core
    return np.asarray(
        fixture[
            time_index,
            :,
            j0 - halo - FIXTURE_OFFSET : j1 + halo - FIXTURE_OFFSET,
            i0 - halo - FIXTURE_OFFSET : i1 + halo - FIXTURE_OFFSET,
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--halo", type=int, required=True)
    parser.add_argument("--read-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    candidate = CODEC_CANDIDATES[args.candidate]
    fixtures = {
        name: np.load(args.fixture / f"{name}.npy", mmap_mode="r")
        for name in ("U", "Theta")
    }
    width = 720 + 2 * args.halo
    args.output.mkdir(parents=True)
    store_path = args.output / "halo.zarr"
    root = zarr.create_group(store=str(store_path), zarr_format=3)
    arrays: dict[str, Any] = {}
    encode_seconds: dict[str, float] = {}
    for name, fixture in fixtures.items():
        array = root.create_array(
            name,
            shape=(fixture.shape[0], len(CORES), fixture.shape[1], width, width),
            chunks=(1, 1, fixture.shape[1], width, width),
            dtype=fixture.dtype,
            compressors=[codec(candidate)],
            dimension_names=("time", "tile", "k", "j_with_halo", "i_with_halo"),
        )
        started = time.perf_counter()
        for time_index in range(fixture.shape[0]):
            for tile_index, core in enumerate(CORES):
                array[time_index, tile_index] = expected_tile(
                    fixture, time_index, core, args.halo
                )
        encode_seconds[name] = time.perf_counter() - started
        arrays[name] = array

    read_runs = []
    for repeat in range(args.read_repeats):
        time_index = repeat % 2
        started = time.perf_counter()
        for name, array in arrays.items():
            for tile_index, core in enumerate(CORES):
                actual = np.asarray(array[time_index, tile_index])
                np.testing.assert_equal(
                    actual, expected_tile(fixtures[name], time_index, core, args.halo)
                )
        read_runs.append(time.perf_counter() - started)

    physical_objects, physical_bytes = tree_size(store_path)
    unique_core_bytes = 2 * 2 * len(CORES) * 51 * 4 * 720 * 720
    result = {
        "candidate": f"halo-{args.halo}-{candidate[0]}",
        "codec": candidate[0],
        "halo": args.halo,
        "tile_width": width,
        "encode_seconds": encode_seconds,
        "four_tile_read_seconds": read_runs,
        "four_tile_median_seconds": statistics.median(read_runs),
        "physical_objects": physical_objects,
        "physical_bytes": physical_bytes,
        "unique_core_logical_bytes": unique_core_bytes,
        "decoded_duplication_factor": (width / 720) ** 2,
        "final_rss_bytes": psutil.Process().memory_info().rss,
        "exact_validation": True,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
