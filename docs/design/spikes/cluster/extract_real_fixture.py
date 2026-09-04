# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Extract one immutable, bounded LLC fixture for layout experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr  # type: ignore[import-untyped]

REGION = (720, 3600, 720, 3600)  # j0, j1, i0, i1; raw-chunk aligned
VOLUME_NAMES = ("U", "Theta")
SURFACE_NAMES = ("Eta",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_objects(
    array_path: Path,
    *,
    times: list[int],
    face: int,
    volume: bool,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for time_index in times:
        if volume:
            j0, j1, i0, i1 = REGION
            coordinates: list[tuple[int, ...]] = [
                (time_index, 0, face, j, i)
                for j in range(j0 // 720, j1 // 720)
                for i in range(i0 // 720, i1 // 720)
            ]
        else:
            coordinates = [(time_index, 0, 0, 0)]
        for coordinate in coordinates:
            path = array_path.joinpath(*(str(value) for value in coordinate))
            stat = path.stat()
            objects.append(
                {
                    "coordinate": coordinate,
                    "path": str(path.resolve()),
                    "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return objects


def extract_volume(
    source_root: Path,
    output: Path,
    name: str,
    times: list[int],
    face: int,
) -> dict[str, Any]:
    source_path = source_root / name
    array = zarr.open_array(str(source_path), mode="r")
    j0, j1, i0, i1 = REGION
    shape = (len(times), array.shape[1], j1 - j0, i1 - i0)
    destination = output / f"{name}.npy"
    fixture = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=array.dtype,
        shape=shape,
    )
    durations = []
    for output_index, time_index in enumerate(times):
        started = time.perf_counter()
        fixture[output_index] = array[time_index, :, face, j0:j1, i0:i1]
        fixture.flush()
        durations.append(time.perf_counter() - started)
    del fixture
    objects = source_objects(
        source_path,
        times=times,
        face=face,
        volume=True,
    )
    return {
        "name": name,
        "source_path": str(source_path.resolve()),
        "shape": shape,
        "dtype": str(array.dtype),
        "source_chunks": array.chunks,
        "read_seconds": durations,
        "source_objects": objects,
        "source_compressed_bytes": sum(item["bytes"] for item in objects),
        "fixture_path": str(destination),
        "fixture_bytes": destination.stat().st_size,
        "fixture_sha256": sha256(destination),
    }


def extract_surface(
    source_root: Path,
    output: Path,
    name: str,
    times: list[int],
    face: int,
) -> dict[str, Any]:
    source_path = source_root / name
    array = zarr.open_array(str(source_path), mode="r")
    shape = (len(times), array.shape[2], array.shape[3])
    destination = output / f"{name}.npy"
    fixture = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=array.dtype,
        shape=shape,
    )
    durations = []
    for output_index, time_index in enumerate(times):
        started = time.perf_counter()
        fixture[output_index] = array[time_index, face, :, :]
        fixture.flush()
        durations.append(time.perf_counter() - started)
    del fixture
    objects = source_objects(
        source_path,
        times=times,
        face=face,
        volume=False,
    )
    return {
        "name": name,
        "source_path": str(source_path.resolve()),
        "shape": shape,
        "dtype": str(array.dtype),
        "source_chunks": array.chunks,
        "read_seconds": durations,
        "source_objects": objects,
        "source_compressed_bytes": sum(item["bytes"] for item in objects),
        "fixture_path": str(destination),
        "fixture_bytes": destination.stat().st_size,
        "fixture_sha256": sha256(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--face", type=int, default=1)
    parser.add_argument("--times", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    zarr.config.set({"async.concurrency": 2})

    manifest: dict[str, Any] = {
        "source_root": str(args.source_root.resolve()),
        "face": args.face,
        "times": args.times,
        "region": REGION,
        "arrays": [],
    }
    for name in VOLUME_NAMES:
        manifest["arrays"].append(
            extract_volume(args.source_root, args.output, name, args.times, args.face)
        )
    for name in SURFACE_NAMES:
        manifest["arrays"].append(
            extract_surface(args.source_root, args.output, name, args.times, args.face)
        )

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    os.chmod(args.output, 0o555)
    for path in args.output.iterdir():
        os.chmod(path, 0o444)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
