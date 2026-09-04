# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Estimate LLC4320 physical bytes with a bounded metadata sample.

Time-dependent arrays are sampled by complete time-chunk directory. Static
arrays are scanned exactly. Run this through Slurm, never on a login node.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Size:
    payload: int = 0
    allocated: int = 0
    files: int = 0

    def __add__(self, other: Size) -> Size:
        return Size(
            self.payload + other.payload,
            self.allocated + other.allocated,
            self.files + other.files,
        )


def entry_size(entry: os.DirEntry[str]) -> Size:
    stat = entry.stat(follow_symlinks=True)
    return Size(
        payload=0 if entry.is_dir(follow_symlinks=True) else stat.st_size,
        allocated=stat.st_blocks * 512,
        files=0 if entry.is_dir(follow_symlinks=True) else 1,
    )


def scan_tree(path: Path) -> Size:
    total = Size()
    pending = [path]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                total += entry_size(entry)
                if entry.is_dir(follow_symlinks=True):
                    pending.append(Path(entry.path))
    return total


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if count >= len(paths):
        return paths
    if count == 1:
        return [paths[len(paths) // 2]]
    indexes = {round(i * (len(paths) - 1) / (count - 1)) for i in range(count)}
    return [paths[index] for index in sorted(indexes)]


def estimate_array(path: Path, sample_count: int) -> dict[str, object]:
    metadata = json.loads((path / ".zarray").read_text())
    attrs = json.loads((path / ".zattrs").read_text())
    dimensions = attrs.get("_ARRAY_DIMENSIONS", [])
    time_chunked = bool(dimensions and dimensions[0] == "time") and (
        metadata["chunks"][0] == 1
    )

    if not time_chunked:
        size = scan_tree(path)
        return {
            "array": path.name,
            "method": "exact",
            **asdict(size),
        }

    root_size = Size()
    time_paths: list[Path] = []
    with os.scandir(path) as entries:
        for entry in entries:
            root_size += entry_size(entry)
            if entry.is_dir(follow_symlinks=True) and entry.name.isdigit():
                time_paths.append(Path(entry.path))
    time_paths.sort(key=lambda item: int(item.name))
    if not time_paths:
        return {
            "array": path.name,
            "method": "exact-no-time-subtrees",
            **asdict(root_size),
        }
    sampled_paths = evenly_spaced(time_paths, sample_count)
    samples = [scan_tree(sample) for sample in sampled_paths]

    def project(field: str) -> tuple[int, float]:
        values = [getattr(sample, field) for sample in samples]
        mean = statistics.fmean(values)
        standard_error = (
            statistics.stdev(values) / math.sqrt(len(values))
            if len(values) > 1
            else 0.0
        )
        root_value = getattr(root_size, field)
        return round(root_value + len(time_paths) * mean), len(
            time_paths
        ) * standard_error

    payload, payload_se = project("payload")
    allocated, allocated_se = project("allocated")
    files, files_se = project("files")
    return {
        "array": path.name,
        "method": "sampled-time-subtrees",
        "time_directories": len(time_paths),
        "sample_count": len(samples),
        "sampled_times": [sample.name for sample in sampled_paths],
        "payload": payload,
        "allocated": allocated,
        "files": files,
        "payload_standard_error": round(payload_se),
        "allocated_standard_error": round(allocated_se),
        "files_standard_error": round(files_se),
    }


def numeric(result: dict[str, object], key: str) -> int | float:
    value = result.get(key, 0)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} is not numeric: {value!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    arrays = sorted(
        path
        for path in args.root.iterdir()
        if path.is_dir() and (path / ".zarray").is_file()
    )
    results = [estimate_array(array, args.samples) for array in arrays]
    totals = {
        field: sum(int(numeric(result, field)) for result in results)
        for field in ("payload", "allocated", "files")
    }
    totals.update(
        {
            f"{field}_standard_error": round(
                math.sqrt(
                    sum(
                        float(numeric(result, f"{field}_standard_error")) ** 2
                        for result in results
                    )
                )
            )
            for field in ("payload", "allocated", "files")
        }
    )
    print(
        json.dumps(
            {"root": str(args.root), "arrays": results, "totals": totals}, indent=2
        )
    )


if __name__ == "__main__":
    main()
