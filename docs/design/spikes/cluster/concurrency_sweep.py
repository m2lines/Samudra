# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Measure bounded source-mount and Icechunk scratch read concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import icechunk as ic  # type: ignore[import-not-found]
import numpy as np
import zarr  # type: ignore[import-untyped]


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values), quantile))


def sweep(read: Callable[[int], int]) -> list[dict[str, Any]]:
    results = []
    for workers in (1, 2, 4, 8, 16):
        durations: list[float] = []

        def timed(index: int) -> int:
            started = time.perf_counter()
            decoded_bytes = read(index)
            durations.append(time.perf_counter() - started)
            return decoded_bytes

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            decoded = list(executor.map(timed, range(workers)))
        wall = time.perf_counter() - started
        results.append(
            {
                "readers": workers,
                "wall_seconds": wall,
                "p50_seconds": statistics.median(durations),
                "p95_seconds": percentile(durations, 0.95),
                "decoded_bytes": sum(decoded),
                "decoded_mib_per_second": sum(decoded) / (1024**2) / wall,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_repo", type=Path)
    parser.add_argument("pilot_result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    zarr.config.set({"async.concurrency": 1})

    source_root = Path("/orcd/data/abodner/003/LLC4320/LLC4320")
    source_arrays = {
        "source_003_U": zarr.open_array(str(source_root / "U"), mode="r"),
        "source_002_Theta": zarr.open_array(str(source_root / "Theta"), mode="r"),
    }

    def source_reader(array):  # noqa: ANN001, ANN202
        def read(index: int) -> int:
            tile = index % 36
            j0 = (tile // 6) * 720
            i0 = (tile % 6) * 720
            values = np.asarray(array[100 + index, :, 1, j0 : j0 + 720, i0 : i0 + 720])
            return values.nbytes

        return read

    result: dict[str, Any] = {
        name: {
            "first_pass": sweep(source_reader(array)),
            "repeat_pass": sweep(source_reader(array)),
        }
        for name, array in source_arrays.items()
    }

    snapshot = json.loads(args.pilot_result.read_text())["snapshot"]
    repo = ic.Repository.open(ic.local_filesystem_storage(str(args.pilot_repo)))
    pilot = zarr.open_array(
        repo.readonly_session(snapshot_id=snapshot).store,
        path="Theta",
        mode="r",
    )

    def pilot_read(index: int) -> int:
        time_index = index % 8
        tile = index % 4
        j0 = 704 if tile < 2 else 1424
        i0 = 704 if tile % 2 == 0 else 1424
        values = np.asarray(pilot[time_index, :, j0 : j0 + 752, i0 : i0 + 752])
        return values.nbytes

    result["scratch_icechunk_Theta"] = {
        "first_pass": sweep(pilot_read),
        "repeat_pass": sweep(pilot_read),
    }
    result["limitations"] = (
        "OS cache state is uncontrolled and scratch is not the approved production "
        "destination; use this sweep to find local saturation behavior, not to set "
        "the production worker count."
    )
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
