# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Conservative observation coarsening, with explicit support and atomic shards.

This module deliberately imports no training machinery so it also runs on Grace CPU.
Daily surface/atmosphere shards retain enough timing information to assemble exact
five-day intervals around arbitrary monthly forecast origins.
"""

import argparse
import hashlib
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from samudra.observations.constants import OBSERVED_DEPTHS

FORCING = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
    "total_precipitation",
)


def interpolate_depth(
    values: NDArray, source_depth: ArrayLike, target_depth: ArrayLike
) -> NDArray[np.float32]:
    """Linear center interpolation; require both bracketing levels, never bridge gaps."""
    source_depth = np.asarray(source_depth)
    target_depth = np.asarray(target_depth)
    if not np.all(np.diff(source_depth) > 0):
        raise ValueError("Depth must increase")
    result = []
    for depth in target_depth:
        j = np.searchsorted(source_depth, depth)
        if j < len(source_depth) and source_depth[j] == depth:
            result.append(values[..., j, :, :])
        elif j == 0 or j == len(source_depth):
            result.append(np.full(values.shape[:-3] + values.shape[-2:], np.nan))
        else:
            weight = (depth - source_depth[j - 1]) / (
                source_depth[j] - source_depth[j - 1]
            )
            result.append(
                (1 - weight) * values[..., j - 1, :, :] + weight * values[..., j, :, :]
            )
    return np.stack(result, axis=-3).astype(np.float32)


def save_atomic(path: Path, **arrays: Any) -> None:
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def prepare(args: argparse.Namespace) -> None:
    from importlib.metadata import version

    # The separate data environment supplies this preprocessing package.
    from ocean_preprocessing import (  # type: ignore[import-not-found]
        regridding,
        vertical_grid,
    )

    target_depth = vertical_grid.OM4_DEPTH_LEVELS[:OBSERVED_DEPTHS]

    grid = np.load(args.grid)
    names = {
        "sst": ("oisst", ["sst"]),
        "adt": ("duacs", ["adt", "ugos", "vgos"]),
        "era5": ("era5-surface", list(FORCING)),
        "iap": ("argo-iap", ["temp", "salt"]),
    }
    product, variables = names[args.product]
    source = Path(args.source) / (product + ".zarr")
    ds = xr.open_zarr(source, chunks=None)
    remap = regridding.FiniteCoverageRemap(
        ds.lat.values, ds.lon.values, grid["lat"], grid["lon"]
    )
    out = Path(args.output) / args.product / str(args.year)
    out.mkdir(parents=True, exist_ok=True)
    times = pd.DatetimeIndex(ds.time.values)
    start = pd.Timestamp(f"{args.year}-01-01")
    end = pd.Timestamp(f"{args.year + 1}-01-01")
    selected = np.flatnonzero((times >= start) & (times < end))
    if args.limit:
        selected = selected[: args.limit * (24 if args.product == "era5" else 1)]
    if not len(selected):
        raise ValueError("Empty source period")
    meta_hash = hashlib.sha256((source / ".zmetadata").read_bytes()).hexdigest()
    signature = {
        "product": product,
        "variables": variables,
        "metadata_sha256": meta_hash,
        "year": args.year,
        "grid_sha256": hashlib.sha256(Path(args.grid).read_bytes()).hexdigest(),
        "min_coverage": 0.9,
        "source": str(source),
        "limit": args.limit,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "horizontal_backend": "ESMF conservative spherical-cell weights",
        "xesmf_version": version("xesmf"),
        "esmpy_version": version("esmpy"),
        "regridding_code_sha256": hashlib.sha256(
            Path(regridding.__file__).read_bytes()
        ).hexdigest(),
        "coverage_definition": "finite source area / full target cell area",
        "era5_accumulations": "hour ending at validity time; daily bins (00:00, next 00:00]",
    }
    manifest = out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Existing shard preparation signature differs")
    manifest.write_text(json.dumps(signature, indent=2) + "\n")
    started = time.monotonic()
    if args.product == "iap":
        # Preprocessing-only dependency, pinned separately on Grace.
        gsw = importlib.import_module("gsw")

        depth_name = next(n for n in ds.temp.dims if n not in ("time", "lat", "lon"))
        z = ds[depth_name].values
        pressure = gsw.p_from_z(-z[:, None, None], ds.lat.values[None, :, None])
        for index in selected:
            stamp = times[index].strftime("%Y-%m")
            path = out / (stamp + ".npz")
            if path.exists():
                continue
            fields = []
            ohc = None
            for name in variables:
                raw = (
                    ds[name]
                    .isel(time=int(index))
                    .transpose(depth_name, "lat", "lon")
                    .values
                )
                if name == "temp":
                    edges = np.r_[
                        0.0, (z[:-1] + z[1:]) / 2, z[-1] + (z[-1] - z[-2]) / 2
                    ]
                    heat = []
                    for low, high in [(0, 700), (700, 2000)]:
                        dz = np.maximum(
                            0, np.minimum(edges[1:], high) - np.maximum(edges[:-1], low)
                        )
                        keep = dz > 0
                        complete = np.isfinite(raw[keep]).all(0)
                        integral = (
                            np.nansum(raw[keep] * dz[keep, None, None], axis=0)
                            * 1035
                            * 3850
                        )
                        integral[~complete] = np.nan
                        heat.append(remap(integral)[0])
                    ohc = np.stack(heat)
                if name == "salt":
                    raw = gsw.SP_from_SA(
                        raw,
                        pressure,
                        ds.lon.values[None, None, :],
                        ds.lat.values[None, :, None],
                    )
                # Interpolate before horizontal averaging: missing depth brackets stay missing.
                raw = interpolate_depth(raw, z, target_depth)
                field, coverage = remap(raw)
                fields.append(field)
            save_atomic(
                path,
                values=np.stack(fields),
                time=np.array(stamp),
                depth=target_depth,
                ohc=ohc,
            )
            print(
                json.dumps(
                    {
                        "event": "prepared",
                        "path": str(path),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    else:
        dates = pd.DatetimeIndex(times[selected].normalize().unique())
        for day in dates:
            path = out / (day.strftime("%Y-%m-%d") + ".npz")
            if path.exists():
                continue
            fields, fractions = [], []
            for name in variables:
                if args.product == "era5":
                    # State fields at 00..23h; accumulations end at 01..24h.
                    accumulation = name in FORCING[-3:]
                    expected = pd.date_range(
                        day + pd.Timedelta(hours=int(accumulation)),
                        periods=24,
                        freq="h",
                    )
                    indices = times.get_indexer(expected)
                    if np.any(indices < 0):
                        raise ValueError(f"Missing hourly support for {name} {day}")
                    raw = (
                        ds[name]
                        .isel(time=indices)
                        .transpose("time", "lat", "lon")
                        .values
                    )
                    valid = np.isfinite(raw).all(0)
                    raw = raw.mean(0)
                    raw[~valid] = np.nan
                    if accumulation:
                        raw = raw / 3600
                        if name == "total_precipitation":
                            raw = raw * 1000  # m water/hour -> kg m-2 s-1
                else:
                    indices = np.flatnonzero(times.normalize() == day)
                    if len(indices) != 1:
                        raise ValueError("Expected one analysis per day")
                    raw = (
                        ds[name]
                        .isel(time=int(indices[0]))
                        .transpose("lat", "lon")
                        .values
                    )
                field, coverage = remap(raw)
                fields.append(field)
                fractions.append(coverage)
            save_atomic(
                path,
                values=np.stack(fields),
                coverage=np.stack(fractions),
                time=np.array(day.isoformat()),
            )
            print(
                json.dumps(
                    {
                        "event": "prepared",
                        "path": str(path),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    (out / "COMPLETE.json").write_text(
        json.dumps(
            {
                "signature": signature,
                "shards": len(list(out.glob("*.npz"))),
                "seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--product", choices=["sst", "adt", "era5", "iap"], required=True
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
