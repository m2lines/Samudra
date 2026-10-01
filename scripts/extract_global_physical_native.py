# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact-bin native and training-grid references for fixed day-30 diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from samudra.experiments.observation_prepare import ConservativeRemap

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--root", type=Path, default=Path("/mnt/home/jrusak/data/obs_full_range")
)
parser.add_argument("--output", type=Path)
parser.add_argument("--lead-days", type=int, default=30)
parser.add_argument("--origins", nargs="+")
args = parser.parse_args()
if args.lead_days < 5:
    parser.error("lead-days must be at least five")
root = args.root
out = args.output or root / "d-observation-pilot/global-physical-focus-20260930"
out.mkdir(parents=True, exist_ok=True)
if (out / "NATIVE_COMPLETE.json").exists():
    raise ValueError("Preserve completed references; choose a new output directory")
origins = (
    [(str(value), pd.Timestamp(value)) for value in args.origins]
    if args.origins
    else [
        (str(month), month.start_time)
        for month in pd.period_range("2015-01", "2015-12", freq="M")
    ]
)
grid = dict(np.load(root / "d-observation-pilot/code/grid.npz"))
records: dict[str, Any] = {}
for product, variables in [("oisst", ["sst"]), ("duacs", ["adt"])]:
    source = root / "prepared" / (product + ".zarr")
    ds = xr.open_zarr(source, chunks=None)
    times = pd.DatetimeIndex(ds.time.values).normalize()
    lat = ds.lat.values
    lon = ds.lon.values
    remap = ConservativeRemap(lat, lon, grid["lat"], grid["lon"])
    records[product] = {
        "metadata_sha256": hashlib.sha256(
            (source / ".zmetadata").read_bytes()
        ).hexdigest(),
        "source": str(source),
        "lead_days": args.lead_days,
        "origins": [name for name, _ in origins],
        "dates": [],
        "files": {},
    }
    np.savez_compressed(out / (product + "-grid.npz"), lat=lat, lon=lon)
    for label, origin in origins:
        start = origin + pd.Timedelta(days=args.lead_days - 5)
        dates = pd.date_range(start, periods=5, freq="D")
        indices = times.get_indexer(dates)
        assert (indices >= 0).all()
        daily = (
            ds[variables[0]].isel(time=indices).transpose("time", "lat", "lon").values
        )
        valid = np.isfinite(daily).all(0)
        native = np.where(
            valid, np.where(np.isfinite(daily), daily, 0).mean(0), np.nan
        ).astype("float32")
        coarse_daily, _ = remap(daily)
        coarse = np.where(
            np.isfinite(coarse_daily).all(0),
            np.where(np.isfinite(coarse_daily), coarse_daily, 0).mean(0),
            np.nan,
        ).astype("float32")
        name = product + "-" + label + ".npz"
        np.savez_compressed(
            out / name,
            native=native,
            coarse=coarse,
            start=str(start.date()),
            end=str((start + pd.Timedelta(days=5)).date()),
        )
        records[product]["dates"].append(
            [str(start.date()), str((start + pd.Timedelta(days=5)).date())]
        )
        records[product]["files"][name] = hashlib.sha256(
            (out / name).read_bytes()
        ).hexdigest()
        print(product, label, flush=True)
records["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(out / "NATIVE_COMPLETE.json").write_text(json.dumps(records, indent=2) + "\n")
import shutil

for year in [2014, 2017, 2020] if not args.origins else []:
    source = (
        root / "d-observation-pilot/daily/iap" / str(year) / (str(year) + "-12.npz")
    )
    assert source.is_file()
    shutil.copy2(source, out / source.name)
