#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Stream public hourly ERA5 into the existing daily OBS forcing convention.

Run on cluster CPUs only. Qualification compares all eight fields against an
existing 1993 daily shard before any historical production preparation.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from samudra.experiments.observation_prepare import (
    FORCING,
    ConservativeRemap,
    save_atomic,
)

SOURCE = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
    "full_37-1h-0p25deg-chunk-1.zarr-v3"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_hours(day, accumulation):
    return pd.date_range(
        pd.Timestamp(day) + pd.Timedelta(hours=int(accumulation)), periods=24, freq="h"
    )


class Remote:
    def __init__(self, workers):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.metadata = {}

    def get(self, key):
        for attempt in range(5):
            try:
                with urllib.request.urlopen(SOURCE + "/" + key, timeout=120) as reply:
                    return reply.read()
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def meta(self, name):
        if name not in self.metadata:
            self.metadata[name] = json.loads(self.get(name + "/.zarray"))
        return self.metadata[name]

    def chunk(self, name, key):
        meta = self.meta(name)
        payload = self.get(name + "/" + key)
        raw = numcodecs.get_codec(meta["compressor"]).decode(payload)
        values = np.frombuffer(raw, dtype=meta["dtype"])
        if values.size != np.prod(meta["chunks"]):
            raise ValueError("Unexpected chunk length: " + name + "/" + key)
        return values.reshape(meta["chunks"])

    def coordinate(self, name):
        meta = self.meta(name)
        count = (meta["shape"][0] + meta["chunks"][0] - 1) // meta["chunks"][0]
        return np.concatenate([self.chunk(name, str(i)) for i in range(count)])[
            : meta["shape"][0]
        ]


def prepare_day(remote, day, times, order, remap):
    fields, coverages = [], []
    for name in FORCING:
        indices = times.get_indexer(expected_hours(day, name in FORCING[-3:]))
        if (indices < 0).any():
            raise ValueError("Missing source timestamps")
        meta = remote.meta(name)
        if meta["chunks"] != [1, 721, 1440]:
            raise ValueError("Unexpected ERA5 chunk layout")
        hourly = np.stack(
            list(remote.pool.map(lambda i: remote.chunk(name, f"{i}.0.0")[0], indices))
        )[:, order, :]
        valid = np.isfinite(hourly).all(0)
        raw = hourly.mean(0)
        raw[~valid] = np.nan
        if name in FORCING[-3:]:
            raw = raw / 3600
            if name == "total_precipitation":
                raw = raw * 1000
        field, coverage = remap(raw)
        if not np.isfinite(field).all():
            raise ValueError(f"Incomplete ERA5 coverage: {day} {name}")
        fields.append(field)
        coverages.append(coverage)
    return dict(values=np.stack(fields), coverage=np.stack(coverages))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--existing-daily", type=Path, required=True)
    p.add_argument("--year", type=int)
    p.add_argument("--qualify", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    if a.qualify == (a.year is not None):
        raise ValueError("Choose qualification or one production year")
    if a.year is not None and not 1975 <= a.year <= 1992:
        raise ValueError("Outside authorized historical preparation")
    a.root.mkdir(parents=True, exist_ok=True)
    remote = Remote(a.workers)
    lat, lon = remote.coordinate("latitude"), remote.coordinate("longitude")
    attrs = json.loads(remote.get("time/.zattrs"))
    if attrs["units"] != "hours since 1900-01-01 00:00:00":
        raise ValueError("Source time convention changed")
    times = pd.DatetimeIndex(
        pd.Timestamp("1900-01-01")
        + pd.to_timedelta(remote.coordinate("time"), unit="h")
    )
    if (
        not times.is_unique
        or not (
            np.diff(times.to_numpy(dtype="datetime64[ns]").astype(np.int64))
            == 3600 * 10**9
        ).all()
    ):
        raise ValueError("Noncontiguous source time")
    grid = dict(np.load(a.grid))
    order = np.argsort(lat)
    remap = ConservativeRemap(lat[order], lon, grid["lat"], grid["lon"])
    contract = dict(
        source=SOURCE,
        code_sha256=sha(__file__),
        grid_sha256=sha(a.grid),
        variables=list(FORCING),
        processing="same hourly support, float32 daily mean and conservative remap as OBS",
    )
    gate = a.root / "QUALIFIED.json"
    if a.qualify:
        day = pd.Timestamp("1993-01-05")
        result = prepare_day(remote, day, times, order, remap)
        existing_path = (
            a.existing_daily / "era5/1993" / (day.strftime("%Y-%m-%d") + ".npz")
        )
        with np.load(existing_path) as existing:
            for key in ("values", "coverage"):
                np.testing.assert_allclose(
                    result[key], existing[key], rtol=1e-6, atol=1e-7
                )
            errors = np.max(np.abs(result["values"] - existing["values"]), axis=(1, 2))
        early = prepare_day(remote, "1975-01-01", times, order, remap)
        save_atomic(a.root / "qualification-1975-01-01.npz", **early)
        gate.write_text(
            json.dumps(
                dict(
                    contract=contract,
                    matched_day=str(day.date()),
                    reference_sha256=sha(existing_path),
                    max_abs_errors=errors.tolist(),
                    early_finite=True,
                ),
                indent=2,
            )
            + "\n"
        )
        print(gate, flush=True)
        return
    proof = json.loads(gate.read_text())
    if proof["contract"] != contract or not proof["early_finite"]:
        raise ValueError("Producer-matched preparation qualification missing")
    out = a.root / "daily/era5" / str(a.year)
    out.mkdir(parents=True, exist_ok=True)
    signature = out / "manifest.json"
    if signature.exists() and json.loads(signature.read_text()) != contract:
        raise ValueError("Existing preparation signature changed")
    signature.write_text(json.dumps(contract, indent=2) + "\n")
    hashes = {}
    for day in pd.date_range(f"{a.year}-01-01", f"{a.year}-12-31"):
        path = out / (day.strftime("%Y-%m-%d") + ".npz")
        start = time.monotonic()
        if not path.exists():
            result = prepare_day(remote, day, times, order, remap)
            save_atomic(path, **result, time=np.array(day.isoformat()))
        with np.load(path) as saved:
            if (
                saved["values"].shape != (8, 180, 360)
                or not np.isfinite(saved["values"]).all()
            ):
                raise ValueError("Invalid prepared day")
            if str(saved["time"]) != day.isoformat():
                raise ValueError("Prepared day timestamp mismatch")
        hashes[path.name] = sha(path)
        print(
            json.dumps(dict(day=str(day.date()), seconds=time.monotonic() - start)),
            flush=True,
        )
    (out / "COMPLETE.json").write_text(
        json.dumps(dict(contract=contract, files=hashes), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
