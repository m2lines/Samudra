#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU audit/cache of ERA5 on every original A OM4 training interval."""

import argparse
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from samudra.experiments.monthly_model_data import ModelMonths
from samudra.experiments.observation_data import strict_mean


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    config = json.loads((a.root / "paths.json").read_text())
    samples = Path(config["samples"])
    grid = dict(np.load(samples / "grid.npz"))
    reader = ModelMonths(Path(config["om4"]) / "OM4.zarr", "om4", grid, config["om4"])
    needed = sorted(
        {i for start, stop, _, _ in reader.windows for i in range(start - 19, stop)}
    )
    early = a.root / "era5"
    gate = json.loads((early / "QUALIFIED.json").read_text())
    early_hashes = {}
    for year in range(1975, 1993):
        proof = json.loads((early / f"daily/era5/{year}/COMPLETE.json").read_text())
        if proof["contract"] != gate["contract"]:
            raise ValueError("Historical source contract mismatch")
        early_hashes.update(proof["files"])
    if len(early_hashes) != 6575:
        raise ValueError("Incomplete historical preparation")
    source_hashes = {}

    @lru_cache(maxsize=32)
    def day(stamp):
        base = early / "daily" if stamp.year < 1993 else Path(config["daily"])
        path = base / "era5" / str(stamp.year) / (stamp.strftime("%Y-%m-%d") + ".npz")
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if stamp.year < 1993 and checksum != early_hashes[path.name]:
            raise ValueError("Historical payload checksum changed")
        source_hashes[str(path)] = checksum
        with np.load(io.BytesIO(raw)) as saved:
            values = saved["values"]
            if pd.Timestamp(str(saved["time"])) != stamp:
                raise ValueError("Daily payload timestamp mismatch")
        if values.shape != (8, 180, 360) or not np.isfinite(values).all():
            raise ValueError("Invalid daily forcing")
        return values

    out = a.root / "forcing-cache"
    out.mkdir(exist_ok=True)
    values = np.lib.format.open_memmap(
        out / "values.npy", mode="w+", dtype="float32", shape=(len(needed), 8, 180, 360)
    )
    bounds = []
    for position, index in enumerate(needed):
        start, end = [pd.Timestamp(str(x)) for x in reader.bounds[index]]
        if end - start != pd.Timedelta(days=5):
            raise ValueError("OM4 interval is not five Gregorian days")
        values[position] = strict_mean(
            np.stack([day(t) for t in pd.date_range(start, periods=5)])
        )
        bounds.append([start.isoformat(), end.isoformat()])
        if position % 100 == 0:
            print(
                json.dumps(
                    dict(position=position, total=len(needed), start=str(start))
                ),
                flush=True,
            )
    values.flush()
    np.save(out / "indices.npy", needed)
    # Independent on-disk readback after the full write.
    readback = np.load(out / "values.npy", mmap_mode="r")
    overlap = 0
    for position, (start_text, _) in enumerate(bounds):
        expected = strict_mean(
            np.stack([day(t) for t in pd.date_range(start_text, periods=5)])
        )
        if not np.array_equal(readback[position], expected):
            raise ValueError("Forcing interval readback mismatch")
        overlap += int(pd.Timestamp(start_text).year >= 1993)
    proof = dict(
        producer=config["producer"],
        windows=len(reader.windows),
        intervals=len(needed),
        bounds=bounds,
        sources=source_hashes,
        overlap_intervals=overlap,
        all_intervals_readback_exact=True,
        files={n: digest(out / n) for n in ("values.npy", "indices.npy")},
        contracts={
            str(samples / n): digest(samples / n)
            for n in ("grid.npz", "statistics.npz")
        },
        qualification_sha256=digest(early / "QUALIFIED.json"),
    )
    (out / "COMPLETE.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps(
            dict(complete=True, windows=len(reader.windows), intervals=len(needed))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
