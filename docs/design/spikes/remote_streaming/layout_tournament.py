# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Build and benchmark focused LLC Zarr v3 sharding candidates A--E."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import psutil  # type: ignore[import-untyped]
import zarr  # type: ignore[import-untyped]
from zarr.abc.store import ByteRequest  # type: ignore[import-untyped]
from zarr.codecs import BloscCodec  # type: ignore[import-untyped]
from zarr.core.buffer import Buffer, BufferPrototype  # type: ignore[import-untyped]
from zarr.storage import LocalStore, WrapperStore  # type: ignore[import-untyped]

CANDIDATES = {
    "A": {"chunks": (1, 17, 1, 120, 120), "shards": (1, 51, 1, 720, 720)},
    "B": {"chunks": (1, 17, 1, 60, 60), "shards": (1, 51, 1, 720, 720)},
    "C": {"chunks": (1, 51, 1, 120, 120), "shards": (1, 51, 1, 720, 720)},
    "D": {"chunks": (1, 17, 1, 120, 120), "shards": (1, 51, 1, 1440, 1440)},
    "E": {"chunks": (1, 17, 1, 120, 120), "shards": (2, 51, 1, 720, 720)},
}

TILES = (
    (704, 1456, 704, 1456),
    (704, 1456, 1424, 2176),
    (1424, 2176, 704, 1456),
    (1424, 2176, 1424, 2176),
)


def buffer_bytes(buffer: Buffer | None) -> int:
    return 0 if buffer is None else len(buffer)


class TrackingStore(WrapperStore[LocalStore]):
    """Record physical reads made by Zarr, including byte-range reads."""

    def __init__(self, store: LocalStore) -> None:
        super().__init__(store)
        self.records: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def _with_store(self, store: LocalStore) -> TrackingStore:
        clone = type(self)(store)
        clone.records = self.records
        clone.lock = self.lock
        return clone

    def reset(self) -> None:
        with self.lock:
            self.records.clear()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            records = list(self.records)
        data_records = [item for item in records if item["key"] != "zarr.json"]
        return {
            "calls": len(records),
            "data_calls": len(data_records),
            "returned_bytes": sum(item["returned_bytes"] for item in records),
            "data_returned_bytes": sum(item["returned_bytes"] for item in data_records),
            "unique_data_objects": len({item["key"] for item in data_records}),
            "range_descriptions": sorted({item["byte_range"] for item in data_records}),
        }

    def _record(self, key: str, byte_range: ByteRequest | None, size: int) -> None:
        with self.lock:
            self.records.append(
                {
                    "key": key,
                    "byte_range": repr(byte_range),
                    "returned_bytes": size,
                }
            )

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        result = await self._store.get(key, prototype, byte_range)
        self._record(key, byte_range, buffer_bytes(result))
        return result

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        result = self._store.get_sync(
            key,
            prototype=prototype,
            byte_range=byte_range,
        )
        self._record(key, byte_range, buffer_bytes(result))
        return result

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        materialized = list(key_ranges)
        results = await self._store.get_partial_values(prototype, materialized)
        for (key, byte_range), result in zip(materialized, results, strict=True):
            self._record(key, byte_range, buffer_bytes(result))
        return results


def tree_size(path: Path) -> tuple[int, int, int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    data_files = [item for item in files if item.name != "zarr.json"]
    return (
        len(files),
        sum(item.stat().st_size for item in files),
        len(data_files),
        sum(item.stat().st_size for item in data_files),
    )


def read_and_check(
    array: Any,
    fixture: np.ndarray[Any, Any],
    time_selection: Any,
    depth_selection: Any,
    bounds: tuple[int, int, int, int],
) -> None:
    j0, j1, i0, i1 = bounds
    actual = np.asarray(array[time_selection, depth_selection, 0, j0:j1, i0:i1])
    expected = np.asarray(fixture[time_selection, depth_selection, j0:j1, i0:i1])
    np.testing.assert_equal(actual, expected)


def benchmark(
    store: TrackingStore,
    array: Any,
    fixture: np.ndarray[Any, Any],
    time_selection: Any,
    depth_selection: Any,
    bounds: tuple[int, int, int, int],
) -> dict[str, Any]:
    store.reset()
    started = time.perf_counter()
    read_and_check(array, fixture, time_selection, depth_selection, bounds)
    seconds = time.perf_counter() - started
    return {"seconds": seconds, **store.snapshot()}


def run_workloads(
    store: TrackingStore,
    array: Any,
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
            benchmark(
                store,
                array,
                fixture,
                time_selection,
                depth_selection,
                bounds,
            )
            for _ in range(3)
        ]
        results[name] = {
            "median_seconds": statistics.median(item["seconds"] for item in trials),
            "trials": trials,
        }

    independent_trials = []
    for _ in range(3):
        store.reset()
        started = time.perf_counter()
        for bounds in TILES:
            read_and_check(array, fixture, 0, slice(None), bounds)
        independent_trials.append(
            {"seconds": time.perf_counter() - started, **store.snapshot()}
        )
    results["independent_four_752"] = {
        "median_seconds": statistics.median(
            item["seconds"] for item in independent_trials
        ),
        "trials": independent_trials,
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    fixture = np.load(args.fixture, mmap_mode="r")
    if fixture.shape != (8, 51, 2880, 2880):
        raise ValueError(f"unexpected fixture shape: {fixture.shape}")
    selected = CANDIDATES[args.candidate]
    store_path = args.output / "candidate.zarr"
    root = zarr.create_group(store=str(store_path), zarr_format=3)
    array = root.create_array(
        "Theta",
        shape=(fixture.shape[0], fixture.shape[1], 1, *fixture.shape[2:]),
        chunks=selected["chunks"],
        shards=selected["shards"],
        dtype=fixture.dtype,
        compressors=[BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")],
        dimension_names=("time", "k", "face", "j", "i"),
    )
    encode_started = time.perf_counter()
    array[:, :, 0, :, :] = fixture
    encode_seconds = time.perf_counter() - encode_started

    file_count, stored_bytes, data_objects, data_bytes = tree_size(store_path)
    tracking_store = TrackingStore(LocalStore(store_path, read_only=True))
    read_root = zarr.open_group(store=tracking_store, mode="r")
    read_array = read_root["Theta"]

    validation_started = time.perf_counter()
    for time_index in range(fixture.shape[0]):
        read_and_check(
            read_array,
            fixture,
            time_index,
            slice(None),
            (0, fixture.shape[2], 0, fixture.shape[3]),
        )
    validation_seconds = time.perf_counter() - validation_started

    result = {
        "candidate": args.candidate,
        "logical_chunks": selected["chunks"],
        "physical_shards": selected["shards"],
        "codec": "blosc-zstd-level-5-bitshuffle",
        "fixture": str(args.fixture.resolve()),
        "fixture_shape_without_face": fixture.shape,
        "fixture_decoded_bytes": fixture.nbytes,
        "encode_seconds": encode_seconds,
        "validation_seconds": validation_seconds,
        "exact_full_validation": True,
        "stored_files": file_count,
        "stored_bytes": stored_bytes,
        "physical_data_objects": data_objects,
        "physical_data_bytes": data_bytes,
        "compression_ratio": data_bytes / fixture.nbytes,
        "workloads": run_workloads(tracking_store, read_array, fixture),
        "final_rss_bytes": psutil.Process().memory_info().rss,
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
