#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only OM4 samples matched to initialization and five-day forecast intervals."""

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
    parser.add_argument("--lead-days", type=int, nargs="+", default=[365])
    parser.add_argument("--map-fields", action="store_true")
    parser.add_argument(
        "--allow-partial-initial",
        action="store_true",
        help="Permit incomplete lead-zero reference coverage; record the gap explicitly.",
    )
    args = parser.parse_args()
    if any(lead != 0 and lead < 5 for lead in args.lead_days):
        parser.error("lead-days must be zero for initialization or at least five")
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
    fields = (
        ["thetao_9", "so_0", "uo_0", "vo_0"] if args.map_fields else ["thetao_0", "zos"]
    )
    alignment = []
    for lead in args.lead_days:
        frames = []
        for origin in origins:
            start = pd.Timestamp(origin) + pd.Timedelta(days=lead - 5)
            lo = start.value / 1e9 / 86400
            overlap = np.maximum(
                0, np.minimum(centers + 2.5, lo + 5) - np.maximum(centers - 2.5, lo)
            )
            indices = np.flatnonzero(overlap > 0)
            covered_days = float(overlap.sum())
            complete = bool(np.isclose(covered_days, 5))
            if not complete and not (
                args.allow_partial_initial and lead == 0 and 0 < covered_days < 5
            ):
                nearby = np.flatnonzero(np.abs(centers - lo) < 12)
                raise ValueError(
                    f"Incomplete OM4 interval: origin={origin}, lead={lead}, "
                    f"covered_days={overlap.sum()}, source_midpoints="
                    f"{[str(stamps[i]) for i in nearby]}, "
                    f"overlaps={overlap[nearby].tolist()}, "
                    f"time_encoding={ds.time.encoding}"
                )
            weights = overlap[indices] / covered_days
            values = np.stack([ds[k].isel(time=indices).values for k in fields], axis=1)
            finite = np.isfinite(values).all(0)
            frame = np.sum(
                np.where(np.isfinite(values), values, 0) * weights[:, None, None, None],
                axis=0,
            )
            frames.append(np.where(finite, frame, np.nan))
            alignment.append(
                {
                    "origin": origin,
                    "lead_days": lead,
                    "start": str(start.date()),
                    "end": str((start + pd.Timedelta(days=5)).date()),
                    "source_midpoints": [str(stamps[i]) for i in indices],
                    "overlap_weights": weights.tolist(),
                    "covered_days": covered_days,
                    "complete": complete,
                    "source_start": str(
                        (stamps[indices[0]] - pd.Timedelta(days=2.5)).date()
                    ),
                    "source_end": str(
                        (stamps[indices[-1]] + pd.Timedelta(days=2.5)).date()
                    ),
                    "sampling_caveat": (
                        None
                        if complete
                        else "Incomplete requested window: weighted mean of available OM4 "
                        "five-day averages, normalized by covered days; no gap filling."
                    ),
                }
            )
        np.savez_compressed(
            out / f"om4-day{lead}.npz",
            **{"values" if args.map_fields else "surface": frames},
            lat=grid["lat"],
            lon=grid["lon"],
            origins=origins,
            fields=fields,
        )
    report = {
        "script_sha256": digest(Path(__file__)),
        "om4_source": str(args.om4),
        "metadata_sha256": digest(args.om4 / ".zmetadata"),
        "grid_source": str(args.grid),
        "grid_sha256": digest(args.grid),
        "alignment": alignment,
        "fields": fields,
        "lead_days": args.lead_days,
        "allow_partial_initial": args.allow_partial_initial,
        "time_support_convention": "Stored midpoint plus/minus 2.5 days; source has no explicit time bounds",
        "field_attributes": {name: ds[name].attrs for name in fields},
        "reference_role": "Date-matched OM4 model-data sample, not observational ground truth",
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
