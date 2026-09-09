# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Read identical LLC layout candidates with native Zarr implementations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import psutil  # type: ignore[import-untyped]

Selection = tuple[Any, Any, int, slice, slice]
Reader = Callable[[Selection], np.ndarray[Any, Any]]

TILES = (
    (704, 1456, 704, 1456),
    (704, 1456, 1424, 2176),
    (1424, 2176, 704, 1456),
    (1424, 2176, 1424, 2176),
)


def open_zarrs(path: Path) -> tuple[Reader, dict[str, str]]:
    import zarr  # type: ignore[import-untyped]
    import zarrs  # type: ignore[import-not-found,unused-ignore]  # noqa: F401

    zarr.config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
            "threading.max_workers": 8,
        }
    )
    array = zarr.open_group(store=str(path), mode="r")["Theta"]

    def read(selection: Selection) -> np.ndarray[Any, Any]:
        return np.asarray(array[selection])

    return read, {
        "zarr": importlib.metadata.version("zarr"),
        "zarrs": importlib.metadata.version("zarrs"),
    }


def open_tensorstore(path: Path) -> tuple[Reader, dict[str, str]]:
    import tensorstore as ts  # type: ignore[import-not-found,unused-ignore]

    array_path = path / "Theta"
    array = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": f"{array_path}/"},
        },
        open=True,
    ).result()

    def read(selection: Selection) -> np.ndarray[Any, Any]:
        return np.asarray(array[selection].read().result())

    return read, {"tensorstore": importlib.metadata.version("tensorstore")}


def check_read(
    read: Reader,
    fixture: np.ndarray[Any, Any],
    time_selection: Any,
    depth_selection: Any,
    bounds: tuple[int, int, int, int],
) -> None:
    j0, j1, i0, i1 = bounds
    selection = (
        time_selection,
        depth_selection,
        0,
        slice(j0, j1),
        slice(i0, i1),
    )
    actual = read(selection)
    expected = np.asarray(fixture[time_selection, depth_selection, j0:j1, i0:i1])
    np.testing.assert_equal(actual, expected)


def timed(
    read: Reader,
    fixture: np.ndarray[Any, Any],
    time_selection: Any,
    depth_selection: Any,
    bounds: tuple[int, int, int, int],
) -> dict[str, float]:
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    check_read(read, fixture, time_selection, depth_selection, bounds)
    return {
        "wall_seconds": time.perf_counter() - wall_started,
        "process_cpu_seconds": time.process_time() - cpu_started,
    }


def workloads(
    read: Reader,
    fixture: np.ndarray[Any, Any],
) -> dict[str, Any]:
    specifications = {
        "aligned_720": (0, slice(None), (720, 1440, 720, 1440)),
        "halo_16_752": (0, slice(None), (704, 1456, 704, 1456)),
        "halo_32_784": (0, slice(None), (688, 1472, 688, 1472)),
        "union_1472": (0, slice(None), (704, 2176, 704, 2176)),
        "misaligned_720": (0, slice(None), (777, 1497, 833, 1553)),
        "shallow_17_halo": (0, slice(17, 34), (704, 1456, 704, 1456)),
        "point_series": (slice(None), slice(20, 21), (1000, 1001, 1000, 1001)),
        "pair_aligned": (slice(0, 2), slice(None), (720, 1440, 720, 1440)),
        "pair_offset": (slice(1, 3), slice(None), (720, 1440, 720, 1440)),
    }
    results: dict[str, Any] = {}
    for name, (time_selection, depth_selection, bounds) in specifications.items():
        trials = [
            timed(read, fixture, time_selection, depth_selection, bounds)
            for _ in range(3)
        ]
        results[name] = {
            "median_wall_seconds": statistics.median(
                item["wall_seconds"] for item in trials
            ),
            "median_process_cpu_seconds": statistics.median(
                item["process_cpu_seconds"] for item in trials
            ),
            "trials": trials,
        }

    trials = []
    for _ in range(3):
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        for bounds in TILES:
            check_read(read, fixture, 0, slice(None), bounds)
        trials.append(
            {
                "wall_seconds": time.perf_counter() - wall_started,
                "process_cpu_seconds": time.process_time() - cpu_started,
            }
        )
    results["independent_four_752"] = {
        "median_wall_seconds": statistics.median(
            item["wall_seconds"] for item in trials
        ),
        "median_process_cpu_seconds": statistics.median(
            item["process_cpu_seconds"] for item in trials
        ),
        "trials": trials,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("candidate_store", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reader", choices=("zarrs", "tensorstore"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    fixture = np.load(args.fixture, mmap_mode="r")
    read, versions = (
        open_zarrs(args.candidate_store)
        if args.reader == "zarrs"
        else open_tensorstore(args.candidate_store)
    )

    validation_started = time.perf_counter()
    for time_index in range(fixture.shape[0]):
        check_read(
            read,
            fixture,
            time_index,
            slice(None),
            (0, fixture.shape[2], 0, fixture.shape[3]),
        )
    result = {
        "reader": args.reader,
        "versions": versions,
        "candidate_store": str(args.candidate_store.resolve()),
        "exact_full_validation": True,
        "validation_seconds": time.perf_counter() - validation_started,
        "workloads": workloads(read, fixture),
        "final_rss_bytes": psutil.Process().memory_info().rss,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
