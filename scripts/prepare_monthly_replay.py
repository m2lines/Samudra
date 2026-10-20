#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Materialize exact five-day replay intervals, using training dates only."""

import argparse
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from samudra.experiments.observation_data import (
    SPLITS,
    ObservationArchive,
    month_intervals,
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def worker(job):
    daily, output, first, lo, hi = job
    sources: dict[str, str] = {}

    class HashedArchive(ObservationArchive):
        @lru_cache(maxsize=32)
        def day(self, product, date):
            stamp = pd.Timestamp(date)
            path = (
                self.root
                / product
                / str(stamp.year)
                / (stamp.strftime("%Y-%m-%d") + ".npz")
            )
            payload = path.read_bytes()
            sources[str(path)] = hashlib.sha256(payload).hexdigest()
            with np.load(io.BytesIO(payload)) as z:
                return z["values"]

    archive = HashedArchive(daily)
    result = np.load(Path(output) / "values.npy", mmap_mode="r+")
    for index in range(lo, hi):
        start = pd.Timestamp(first) + pd.Timedelta(days=index)
        end = start + pd.Timedelta(days=5)
        result[index] = np.concatenate(
            [
                archive.interval("sst", start, end),
                archive.interval("adt", start, end)[:1],
                archive.interval("era5", start, end),
            ]
        )
        if not np.isfinite(result[index, 2:]).all():
            raise ValueError(f"Nonfinite ERA5 forcing: {start}")
    result.flush()
    return sources


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daily", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--producer", required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    first = pd.Timestamp(SPLITS["train"][0])
    last = month_intervals(SPLITS["train"][1])[0][-1]
    days = pd.date_range(first, last, freq="D")
    with np.load(a.samples / "grid.npz") as grid:
        shape = grid["mask"].shape[-2:]
    starts = days.to_numpy(dtype="datetime64[D]").astype(np.int64)
    np.save(a.out / "starts.npy", starts)
    target = np.lib.format.open_memmap(
        a.out / "values.npy", mode="w+", dtype=np.float32, shape=(len(days), 10, *shape)
    )
    del target
    jobs = [
        (str(a.daily), str(a.out), str(first.date()), lo, min(lo + 256, len(days)))
        for lo in range(0, len(days), 256)
    ]
    sources: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, part in enumerate(pool.map(worker, jobs)):
            for path, sha in part.items():
                if path in sources and sources[path] != sha:
                    raise ValueError("Daily source changed during preparation")
                sources[path] = sha
            print(
                json.dumps(dict(completed_chunks=i + 1, chunks=len(jobs))), flush=True
            )
    values = np.load(a.out / "values.npy", mmap_mode="r")
    checked = 0
    sample_hashes = {}
    for month in pd.period_range(*SPLITS["train"], freq="M"):
        path = a.samples / "train" / (str(month) + ".npz")
        start = month_intervals(str(month))[0][19:]
        indices = start.to_numpy(dtype="datetime64[D]").astype(np.int64) - starts[0]
        with np.load(path) as z:
            np.testing.assert_array_equal(values[indices, :2], z["surface"][19:])
            np.testing.assert_array_equal(values[indices, 2:], z["atmosphere"][19:])
        sample_hashes[str(path)] = digest(path)
        checked += 1
    (a.out / "source-sha256.json").write_text(
        json.dumps(sources, sort_keys=True) + "\n"
    )
    result = dict(
        producer=a.producer,
        training_split=list(SPLITS["train"]),
        first_bin_start=str(first.date()),
        last_bin_start=str(last.date()),
        last_raw_day=str((last + pd.Timedelta(days=4)).date()),
        intervals=len(days),
        bytes=(a.out / "values.npy").stat().st_size,
        exact_original_months=checked,
        samples=sample_hashes,
        contracts={
            str(a.samples / name): digest(a.samples / name)
            for name in ("grid.npz", "statistics.npz", "SHA256SUMS")
        },
        files={
            name: digest(a.out / name)
            for name in ("starts.npy", "values.npy", "source-sha256.json")
        },
    )
    (a.out / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
