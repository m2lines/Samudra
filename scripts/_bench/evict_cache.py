#!/usr/bin/env python3
"""Drop this node's page cache for a zarr store's prognostic/boundary chunks over a row range.

    .venv/bin/python scripts/_bench/evict_cache.py --store S [--store S2] --rows 0:600
"""

import argparse
import itertools
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def chunk_files(store: Path, array: str, rows: range) -> list[Path]:
    meta = json.loads((store / array / ".zarray").read_text())
    sep = meta.get("dimension_separator", ".")
    grid = [math.ceil(s / c) for s, c in zip(meta["shape"], meta["chunks"])]
    rest = itertools.product(*(range(n) for n in grid[1:]))
    tails = [sep.join(str(i) for i in index) for index in rest]
    return [store / array / f"{row}{sep}{tail}" for row in rows for tail in tails]


def evict(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", action="append", required=True)
    parser.add_argument("--rows", default="0:600", help="start:stop store rows")
    args = parser.parse_args()
    start, stop = (int(v) for v in args.rows.split(":"))
    for store in map(Path, args.store):
        files = [f for array in ("prognostic", "boundary") for f in chunk_files(store, array, range(start, stop))]
        with ThreadPoolExecutor(32) as pool:
            done = sum(pool.map(evict, files))
        print(f"{os.uname().nodename}: evicted {done}/{len(files)} chunk files of {store}")


if __name__ == "__main__":
    main()
