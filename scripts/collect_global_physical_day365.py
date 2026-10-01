#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only OM4 references matched to the final five days of annual forecasts."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--om4", type=Path, default=Path("/scratch/jr7309/data/om4_onedeg_v3/OM4.zarr")
    )
    parser.add_argument(
        "--grid", type=Path, default=Path("/scratch/jr7309/data/obs-d-pilot/grid.npz")
    )
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        raise ValueError("Preserve completed exports; choose a new output")
    grid = dict(np.load(args.grid))
    ds = xr.open_zarr(args.om4, chunks=None)
    np.testing.assert_allclose(ds.y.values, grid["lat"])
    np.testing.assert_allclose(ds.x.values, grid["lon"])
    stamps = pd.DatetimeIndex([str(v) for v in ds.time.values])
    centers = stamps.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9 / 86400
    origins = ["2015-01-01", "2018-01-01", "2021-01-01"]
    frames = []
    alignment = []
    for origin in origins:
        start = pd.Timestamp(origin) + pd.Timedelta(days=360)
        lo = start.value / 1e9 / 86400
        overlap = np.maximum(
            0, np.minimum(centers + 2.5, lo + 5) - np.maximum(centers - 2.5, lo)
        )
        indices = np.flatnonzero(overlap > 0)
        assert np.isclose(overlap.sum(), 5)
        weights = overlap[indices] / 5
        values = np.stack(
            [ds[k].isel(time=indices).values for k in ["thetao_0", "zos"]], axis=1
        )
        finite = np.isfinite(values).all(0)
        frame = np.sum(
            np.where(np.isfinite(values), values, 0) * weights[:, None, None, None],
            axis=0,
        )
        frames.append(np.where(finite, frame, np.nan))
        alignment.append(
            {
                "origin": origin,
                "lead_days": 365,
                "start": str(start.date()),
                "end": str((start + pd.Timedelta(days=5)).date()),
                "source_midpoints": [str(stamps[i]) for i in indices],
                "overlap_weights": weights.tolist(),
            }
        )
    np.savez_compressed(
        out / "om4-day365.npz",
        surface=frames,
        lat=grid["lat"],
        lon=grid["lon"],
        origins=origins,
    )
    report = {
        "script_sha256": digest(Path(__file__)),
        "om4_source": str(args.om4),
        "metadata_sha256": digest(args.om4 / ".zmetadata"),
        "grid_source": str(args.grid),
        "grid_sha256": digest(args.grid),
        "alignment": alignment,
        "files": {
            p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in out.glob("*.npz")
        },
    }
    (out / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
    archive = out.with_suffix(".tar")
    with tarfile.open(archive, "w") as tar:
        tar.add(out, arcname="om4")
    receipt = {"sha256": digest(archive), "bytes": archive.stat().st_size}
    (out.parent / (out.name + "-receipt.json")).write_text(json.dumps(receipt) + "\n")
    print("DAY365_REFERENCE_COMPLETE", receipt, flush=True)


if __name__ == "__main__":
    main()
