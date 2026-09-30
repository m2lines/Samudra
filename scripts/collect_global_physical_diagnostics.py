# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verified, CPU-only extraction of checkpoint maps, means and OM4 references."""

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("/scratch/jr7309/runs/2026-09-30-global-physical-diagnostics")
OLD = Path("/scratch/jr7309/runs/2026-09-29-observation-global")
DATA = Path("/scratch/jr7309/data/obs-d-pilot")
ANNUAL = Path("/scratch/jr7309/data/obs-d-annual-instance-v1/test")
OUT = ROOT / "compact-v2"
OUT.mkdir(exist_ok=True)
origins = ["2015-01-01", "2018-01-01", "2021-01-01"]
channels = [38, 47, 57, 66, 0, 19, 76]
grid = dict(np.load(DATA / "grid.npz"))
stats = dict(np.load(DATA / "statistics.npz"))
lat, lon = grid["lat"], grid["lon"]
mask = grid["mask"]
area = np.cos(np.deg2rad(lat))[:, None]
np.savez_compressed(
    OUT / "grid.npz",
    lat=lat,
    lon=lon,
    mask=mask[channels],
    channels=channels,
    names=grid["names"][channels],
)
np.savez_compressed(OUT / "climatology.npz", surface=stats["surface_climatology"])


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mean(z, support):
    w = np.where(support, area, 0)
    assert w.sum() > 0
    assert np.isfinite(z[..., support]).all()
    return (np.where(support, z, 0) * w).sum(axis=(-2, -1)) / w.sum()


def ohc(temp):
    edges = np.array(
        [
            0,
            5,
            15,
            30,
            50,
            80,
            130,
            200,
            300,
            450,
            650,
            900,
            1200,
            1600,
            2100,
            2700,
            3500,
            4500,
            5500,
            6750,
        ]
    )
    result = []
    for lo, hi in [(0, 700), (700, 2000)]:
        dz = np.maximum(0, np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo))
        keep = dz > 0
        valid = mask[38:57][keep].all(0)
        result.append(
            np.where(
                valid,
                np.sum(temp[keep] * dz[keep, None, None], 0) * 1035 * 3850,
                np.nan,
            )
        )
    return np.array(result)


# Native-layer observational OHC climatology from training months only.
total = np.zeros((12, 2, 180, 360))
count = np.zeros_like(total)
for p in sorted((DATA / "train").glob("*.npz")):
    values = np.load(p)["ohc"]
    i = int(p.stem[5:7]) - 1
    valid = np.isfinite(values)
    total[i] += np.where(valid, values, 0)
    count[i] += valid
climate_ohc = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
report = {
    "script_sha256": digest(__file__),
    "checkpoints": {},
    "om4": {},
    "sources": {},
    "annual": {},
}
for obs in [50, 500, 2000, 8000]:
    label = f"obs{obs:05d}"
    monthly = (
        OLD / "global-endpoint-monthly"
        if obs == 8000
        else ROOT / (label + "-monthly-retry1")
    )
    annual = (
        OLD / "global-endpoint-annual"
        if obs == 8000
        else ROOT / (label + "-annual-retry1")
    )
    complete = json.loads((annual / "COMPLETE.json").read_text())
    sig = complete["inputs"]
    assert sig["global_observations"]
    fixed = json.loads((monthly / "fixed-budget.json").read_text())
    lineage = fixed["fixed_budget_lineage"]
    assert sig["checkpoint_sha256"] == lineage["checkpoint_sha256"] == fixed["sha256"]
    assert lineage["task_counts"]["observation"] == obs
    m = dict(np.load(monthly / "fixed-budget-predictions.npz"))
    assert len(m["origins"]) == 96
    assert list(m["origins"][:12]) == [
        str(v) for v in pd.period_range("2015-01", "2015-12", freq="M")
    ]
    files = {
        k: monthly / (k + ".npz")
        for k in ["fixed-budget-inferred-persistence", "seasonal-climatology"]
    }
    baseline = {k: dict(np.load(p)) for k, p in files.items()}
    np.savez_compressed(
        OUT / (label + "-monthly.npz"),
        prediction=m["prediction"][:12, 5],
        reference=m["reference"][:12, 5],
        origins=m["origins"][:12],
        persistence=baseline["fixed-budget-inferred-persistence"]["prediction"][:12, 5],
        climatology=baseline["seasonal-climatology"]["prediction"][:12, 5],
    )
    record = {"lineage": lineage, "day30": {}, "annual": {}}
    for method, z in [
        ("forecast", m),
        ("persistence", baseline["fixed-budget-inferred-persistence"]),
        ("climatology", baseline["seasonal-climatology"]),
    ]:
        per = []
        for i in range(96):
            row = {}
            for c, name in enumerate(["sst", "adt"]):
                support = mask[38 if c == 0 else 76] & np.isfinite(
                    m["reference"][i, 5, c]
                )
                delta = z["prediction"][i, 5, c] - m["reference"][i, 5, c]
                row[name + "_rmse"] = float(np.sqrt(mean(delta**2, support)))
                row[name + "_bias"] = float(mean(delta, support))
            per.append(row)
        record["day30"][method] = {
            "mean": {k: float(np.mean([v[k] for v in per])) for k in per[0]},
            "per_origin": per,
        }
    for origin in origins:
        source = annual / (origin + ".npz")
        v = dict(np.load(source))
        raw = dict(np.load(ANNUAL / origin / "018.npz"))
        initial = v["initial"][-1]
        np.savez_compressed(
            OUT / (label + "-" + origin + ".npz"),
            initial=initial[channels],
            state=v["state_at_leads"][[2, 5]][:, channels],
            surface=v["surface"],
            reference=v["reference"],
            initial_reference=np.concatenate([raw["surface"], raw["velocity"]]),
            months=v["months"],
            predicted_ohc=v["predicted_ohc"],
            reference_ohc=v["reference_ohc"],
        )
        dates = pd.date_range(
            pd.Timestamp(origin) + pd.Timedelta(days=2.5), periods=73, freq="5D"
        )
        midmonths = dates.month.to_numpy() - 1
        means = {}
        support_info = {}
        for c, name, slot in [(0, "sst", 38), (1, "adt", 76)]:
            climate = stats["surface_climatology"][midmonths, c]
            support = (
                mask[slot]
                & np.isfinite(v["reference"][:, c]).all(0)
                & np.isfinite(climate).all(0)
            )
            means[name] = {
                "forecast": mean(v["surface"][:, c], support).tolist(),
                "observation": mean(v["reference"][:, c], support).tolist(),
                "persistence": np.repeat(mean(initial[slot], support), 73).tolist(),
                "climatology": mean(climate, support).tolist(),
            }
            support_info[name] = {
                "cells": int(support.sum()),
                "model_wet_cells": int(mask[slot].sum()),
                "cosine_area_fraction": float(
                    (support * area).sum() / (mask[slot] * area).sum()
                ),
            }
        month_indices = pd.PeriodIndex(v["months"], freq="M").month.to_numpy() - 1
        initial_ohc = ohc(initial[38:57])
        for c, name in enumerate(["ohc_0_700", "ohc_700_2000"]):
            climate = climate_ohc[month_indices, c]
            support = (
                np.isfinite(v["reference_ohc"][:, c]).all(0)
                & np.isfinite(v["predicted_ohc"][:, c]).all(0)
                & np.isfinite(initial_ohc[c])
                & np.isfinite(climate).all(0)
            )
            means[name] = {
                "forecast": mean(v["predicted_ohc"][:, c], support).tolist(),
                "observation": mean(v["reference_ohc"][:, c], support).tolist(),
                "persistence": np.repeat(
                    mean(initial_ohc[c], support), len(month_indices)
                ).tolist(),
                "climatology": mean(climate, support).tolist(),
            }
            support_info[name] = {
                "cells": int(support.sum()),
                "cosine_area_fraction_of_surface_wet": float(
                    (support * area).sum() / (mask[38] * area).sum()
                ),
            }
        record["annual"][origin] = {
            "dates": dates.strftime("%Y-%m-%d").tolist(),
            "months": v["months"].tolist(),
            "means": means,
            "support": support_info,
        }
        report["sources"][str(source)] = digest(source)
    report["checkpoints"][label] = record
    print("checkpoint_compact", label, flush=True)
# Match OM4 five-day averages by overlap, rather than nearest timestamp.
om4 = Path("/scratch/jr7309/data/om4_onedeg_v3/OM4.zarr")
ds = xr.open_zarr(om4, chunks=None)
stamps = pd.DatetimeIndex([str(v) for v in ds.time.values])
centers = stamps.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9 / 86400
frames = []
alignment = []
for month in pd.period_range("2015-01", "2015-12", freq="M"):
    start = month.start_time + pd.Timedelta(days=25)
    s = start.value / 1e9 / 86400
    overlap = np.maximum(
        0, np.minimum(centers + 2.5, s + 5) - np.maximum(centers - 2.5, s)
    )
    indices = np.flatnonzero(overlap > 0)
    assert np.isclose(overlap.sum(), 5)
    weights = overlap[indices] / 5
    om4_frames = np.stack(
        [ds[k].isel(time=indices).values for k in ["thetao_0", "zos"]], axis=1
    )
    valid = np.isfinite(om4_frames).all(0)
    frame = np.sum(
        np.where(np.isfinite(om4_frames), om4_frames, 0) * weights[:, None, None, None],
        axis=0,
    )
    frames.append(np.where(valid, frame, np.nan))
    alignment.append(
        {
            "origin": str(month),
            "start": str(start.date()),
            "end": str((start + pd.Timedelta(days=5)).date()),
            "source_midpoints": [str(stamps[i]) for i in indices],
            "overlap_weights": weights.tolist(),
        }
    )
np.testing.assert_allclose(ds.y.values, lat)
np.testing.assert_allclose(ds.x.values, lon)
np.savez_compressed(OUT / "om4-day30.npz", surface=frames, lat=lat, lon=lon)
report["om4"] = {
    "metadata_sha256": digest(om4 / ".zmetadata"),
    "alignment": alignment,
    "attrs": {k: ds[k].attrs for k in ["thetao_0", "zos"]},
}
report["sources"][str(DATA / "grid.npz")] = digest(DATA / "grid.npz")
report["sources"][str(DATA / "statistics.npz")] = digest(DATA / "statistics.npz")
report["files"] = {
    p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in OUT.glob("*.npz")
}
(OUT / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
archive = ROOT / "compact-v2.tar"
with tarfile.open(archive, "w") as tar:
    tar.add(OUT, arcname="compact")
(ROOT / "compact-v2-receipt.json").write_text(
    json.dumps({"sha256": digest(archive), "bytes": archive.stat().st_size}) + "\n"
)
print("COMPLETE", flush=True)
