#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare the six-hour LLC/OBS experiment without copying compact LLC fields.

Run grid, era5 (one job per year), then finalize. Requires the data environment.
Existing conservative daily surface shards are reused after grid equality checks.
ERA5 frame at t means states over [t-6h,t), and hourly accumulations ending in
(t-6h,t]. Source product timestamps and these conventions are retained in proofs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from samudra.observations.constants import MODEL_VARIABLES
from samudra.observations.prepare import FORCING, save_atomic

VERSION = "llc-obs-six-hour-v1"
SPLITS = {
    "llc": {
        "train": ["2011-09-13", "2012-09-14"],
        "validation": ["2012-09-14", "2012-11-15T15:00"],
    },
    "obs": {
        "train": ["1993-05-01", "2013-08-01"],
        "validation": ["2013-11-01", "2014-08-01"],
    },
}
EXCLUDED = [["2012-09-14", "2012-11-16"]]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, values):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def grid(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "grid.npz").exists():
        raise FileExistsError(
            "Grid already exists; do not overwrite a prepared archive"
        )
    ds = xr.open_zarr(args.llc, chunks=None)
    old = dict(np.load(args.daily / "grid.npz", allow_pickle=False))
    np.testing.assert_allclose(ds.lat.values, old["lat"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(ds.lon.values, old["lon"], rtol=0, atol=1e-8)
    geometry = xr.open_dataset(args.target_grid)
    lat = geometry.lat.values
    lon = geometry.lon.values
    np.testing.assert_allclose(
        lat if lat.ndim == 1 else lat[:, 0], old["lat"], atol=1e-8
    )
    np.testing.assert_allclose(lon if lon.ndim == 1 else lon[0], old["lon"], atol=1e-8)
    old["lat_b"], old["lon_b"] = geometry.lat_b.values, geometry.lon_b.values
    native = json.loads(args.depths.read_text())
    depth, thickness = -np.array(native["Z"]), np.array(native["drF"])
    if (
        depth.shape != (51,)
        or thickness.shape != (51,)
        or not (np.diff(depth) > 0).all()
        or not (thickness > 0).all()
    ):
        raise ValueError("Native depth metadata must contain 51 ordered layers")
    wet = ds.wet_area.values
    if not np.isfinite(wet).all() or (wet < 0).any():
        raise ValueError("Invalid LLC wet area")
    area = np.concatenate([wet] * 4 + [wet[:1]])
    names = [f"{v}_{k}" for v in MODEL_VARIABLES for k in range(51)] + ["zos"]
    means, stds = [], []
    for v in ("U", "V", "Theta", "Salt", "Eta"):
        means.extend(np.atleast_1d(ds[v + "_mean"].values).tolist())
        stds.extend(np.atleast_1d(ds[v + "_std"].values).tolist())
    stats_indices = np.array(json.loads(ds.attrs["stats_steps_json"]), dtype=int)
    stats_dates = pd.DatetimeIndex(ds.time.values[stats_indices])
    start, end = map(pd.Timestamp, SPLITS["llc"]["train"])
    if not ((stats_dates >= start) & (stats_dates < end)).all():
        raise ValueError("Embedded LLC normalization includes held-out times")
    save_atomic(
        args.output / "grid.npz",
        lat=old["lat"],
        lon=old["lon"],
        lat_b=old["lat_b"],
        lon_b=old["lon_b"],
        depth=depth,
        thickness=thickness,
        wet_area=area,
        names=np.array(names),
        llc_mean=np.array(means),
        llc_std=np.array(stds),
    )
    proof = dict(
        version=VERSION,
        llc=str(args.llc.resolve()),
        llc_metadata_sha256=digest(args.llc / ".zmetadata"),
        daily_root=str(args.daily.resolve()),
        daily_grid_sha256=digest(args.daily / "grid.npz"),
        native_depth_sha256=digest(args.depths),
        native_depth_source="engaging:/orcd/data/abodner/003/LLC4320/LLC4320/{Z,drF}",
        llc_statistics_timestamps=stats_dates.astype(str).tolist(),
        splits=SPLITS,
        excluded=EXCLUDED,
        forcing_fields=list(FORCING),
        forcing_support="frame t: state mean [t-6h,t), hourly accumulations ending (t-6h,t]; radiation /3600 W m-2, precipitation *1000/3600 kg m-2 s-1",
        observation_slots="same-date midnight analysis slot; OISST source nominal noon, DUACS nominal midnight; no subdaily interpolation",
        sha256={"grid.npz": digest(args.output / "grid.npz")},
    )
    save_json(args.output / "grid.json", proof)


def era5(args):
    from ocean_preprocessing.regridding import FiniteCoverageRemap, rectilinear_grid

    grid_values = dict(np.load(args.output / "grid.npz", allow_pickle=False))
    source = args.source / "era5-surface.zarr"
    ds = xr.open_zarr(source, chunks=None)
    source_geometry = xr.open_dataset(args.source_grid) if args.source_grid else ds
    np.testing.assert_allclose(source_geometry.lat.values, ds.lat.values)
    np.testing.assert_allclose(source_geometry.lon.values, ds.lon.values)
    remap = FiniteCoverageRemap(
        rectilinear_grid(
            ds.lat.values,
            ds.lon.values,
            source_geometry.lat_b.values,
            source_geometry.lon_b.values,
        ),
        rectilinear_grid(
            grid_values["lat"],
            grid_values["lon"],
            grid_values["lat_b"],
            grid_values["lon_b"],
        ),
    )
    out = args.output / "era5" / str(args.year)
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(
        version=VERSION,
        source_metadata_sha256=digest(source / ".zmetadata"),
        grid_sha256=digest(args.output / "grid.npz"),
        code_sha256=digest(__file__),
        year=args.year,
        source_grid_sha256=digest(args.source_grid) if args.source_grid else None,
    )
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("ERA5 preparation signature changed")
    save_json(manifest, signature)
    times = pd.DatetimeIndex(ds.time.values)
    days = pd.date_range(
        f"{args.year}-01-01",
        min(pd.Timestamp(f"{args.year + 1}-01-01"), pd.Timestamp("2014-08-01")),
        freq="D",
        inclusive="left",
    )
    if args.limit:
        days = days[: args.limit]
    hashes = {}
    for day in days:
        path = out / f"{day:%Y-%m-%d}.npz"
        if not path.exists():
            fields = []
            for i, name in enumerate(FORCING):
                expected = pd.date_range(
                    day + pd.Timedelta(hours=int(i >= 5)), periods=24, freq="h"
                )
                indices = times.get_indexer(expected)
                if (indices < 0).any():
                    raise ValueError(f"Missing ERA5 hourly support: {day} {name}")
                raw = ds[name].isel(time=indices).transpose("time", "lat", "lon").values
                raw = raw.reshape(4, 6, *raw.shape[-2:]).mean(1)
                if i >= 5:
                    raw = raw / 3600 * (1000 if i == 7 else 1)
                values, coverage = remap(raw)
                if not np.isfinite(values).all() or (coverage < 0.9 - 1e-6).any():
                    raise ValueError(f"Incomplete ERA5 spatial support: {day} {name}")
                fields.append(values)
            save_atomic(
                path,
                values=np.stack(fields, 1),
                interval_start=np.array(
                    pd.date_range(day, periods=4, freq="6h"), dtype="datetime64[ns]"
                ),
                interval_end=np.array(
                    pd.date_range(day + pd.Timedelta(hours=6), periods=4, freq="6h"),
                    dtype="datetime64[ns]",
                ),
            )
        hashes[str(path.relative_to(args.output))] = digest(path)
        if day.day == 1:
            print(
                json.dumps(dict(event="era5", day=str(day), count=len(hashes))),
                flush=True,
            )
    save_json(
        out / ("PILOT.json" if args.limit else "COMPLETE.json"),
        dict(signature=signature, shards=hashes),
    )


def finalize(args):
    proof = json.loads((args.output / "grid.json").read_text())
    grid_values = dict(np.load(args.output / "grid.npz", allow_pickle=False))
    area = grid_values["wet_area"][[102, 204]].astype(np.float64)
    surface_sum = np.zeros(2)
    surface_square = np.zeros(2)
    surface_weight = np.zeros(2)
    forcing_sum = np.zeros(8)
    forcing_square = np.zeros(8)
    forcing_weight = 0.0
    proof["surface_shards"] = {}
    proof["era5_shards"] = {}
    first, last = pd.Timestamp("1993-01-01"), pd.Timestamp("2014-08-01")
    training_start, training_end = map(pd.Timestamp, SPLITS["obs"]["train"])
    for year in range(1993, 2015):
        year_proof = json.loads(
            (args.output / "era5" / str(year) / "COMPLETE.json").read_text()
        )
        proof["era5_shards"].update(year_proof["shards"])
    for day in pd.date_range(first, last, freq="D", inclusive="left"):
        training = training_start <= day < training_end and not any(
            pd.Timestamp(a) <= day < pd.Timestamp(b) for a, b in EXCLUDED
        )
        for i, product in enumerate(("sst", "adt")):
            relative = f"{product}/{day.year}/{day:%Y-%m-%d}.npz"
            path = args.daily / relative
            proof["surface_shards"][relative] = digest(path)
            if training:
                with np.load(path, allow_pickle=False) as saved:
                    values = saved["values"][0].astype(np.float64)
                valid = np.isfinite(values)
                weight = area[i] * valid
                values = np.where(valid, values, 0)
                surface_weight[i] += weight.sum()
                surface_sum[i] += (values * weight).sum()
                surface_square[i] += (values**2 * weight).sum()
        relative = f"era5/{day.year}/{day:%Y-%m-%d}.npz"
        path = args.output / relative
        if digest(path) != proof["era5_shards"].get(relative):
            raise ValueError(f"ERA5 proof differs or lacks {relative}")
        if training:
            with np.load(path, allow_pickle=False) as saved:
                values = saved["values"].astype(np.float64)
            weight = area[0][None, None]
            forcing_sum += (values * weight).sum((0, 2, 3))
            forcing_square += (values**2 * weight).sum((0, 2, 3))
            forcing_weight += 4 * area[0].sum()
    mean = surface_sum / surface_weight
    std = np.sqrt(surface_square / surface_weight - mean**2)
    atmosphere_mean = forcing_sum / forcing_weight
    atmosphere_std = np.sqrt(forcing_square / forcing_weight - atmosphere_mean**2)
    state_mean, state_std = (
        grid_values["llc_mean"].copy(),
        grid_values["llc_std"].copy(),
    )
    datum = dict(
        llc_surface_mean=state_mean[[102, 204]].tolist(),
        observation_surface_mean=mean.tolist(),
        llc_minus_obs_mean=(state_mean[[102, 204]] - mean).tolist(),
        note="Training-distribution means use different dates/support; this is a datum sanity check, not an estimate of paired bias",
    )
    state_mean[[102, 204]], state_std[[102, 204]] = mean, std
    for values in (state_std, atmosphere_std):
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError("Invalid normalization variance")
    save_atomic(
        args.output / "statistics.npz",
        state_mean=state_mean.astype(np.float32),
        state_std=state_std.astype(np.float32),
        atmosphere_mean=atmosphere_mean.astype(np.float32),
        atmosphere_std=atmosphere_std.astype(np.float32),
    )
    proof.update(
        time_start="1993-01-01T06:00", time_end="2014-08-01T06:00", datum_audit=datum
    )
    proof["sha256"]["statistics.npz"] = digest(args.output / "statistics.npz")
    save_json(args.output / "COMPLETE.json", proof)
    print(json.dumps(dict(event="complete", datum_audit=datum)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["grid", "era5", "finalize"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llc", type=Path)
    parser.add_argument("--daily", type=Path)
    parser.add_argument("--depths", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--source-grid", type=Path)
    parser.add_argument("--target-grid", type=Path)
    parser.add_argument("--year", type=int)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
