# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Extract a bounded 24-hour LLC fixture for time-packing experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import zarr  # type: ignore[import-untyped]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-count", type=int, default=24)
    parser.add_argument("--face", type=int, default=1)
    parser.add_argument("--j-start", type=int, default=1440)
    parser.add_argument("--i-start", type=int, default=1440)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument(
        "--arrays", nargs="+", choices=("Theta", "Eta"), default=("Theta", "Eta")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    times = slice(args.time_start, args.time_start + args.time_count)
    j_slice = slice(args.j_start, args.j_start + args.width)
    i_slice = slice(args.i_start, args.i_start + args.width)
    zarr.config.set({"async.concurrency": 2})

    manifest = {
        "source": str(args.source.resolve()),
        "time_start": args.time_start,
        "time_count": args.time_count,
        "face": args.face,
        "j": [j_slice.start, j_slice.stop],
        "i": [i_slice.start, i_slice.stop],
        "arrays": [],
    }
    all_selections = {
        "Theta": (times, slice(None), args.face, j_slice, i_slice),
        "Eta": (times, args.face, j_slice, i_slice),
    }
    for name in args.arrays:
        selection = all_selections[name]
        array = zarr.open_array(str(args.source / name), mode="r")
        destination = args.output / f"{name}.npy"
        started = time.perf_counter()
        values = np.asarray(array[selection])
        np.save(destination, values)
        manifest["arrays"].append(
            {
                "name": name,
                "shape": values.shape,
                "dtype": str(values.dtype),
                "source_chunks": array.chunks,
                "read_and_write_seconds": time.perf_counter() - started,
                "fixture_bytes": destination.stat().st_size,
                "fixture_sha256": sha256(destination),
            }
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
