# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare evaluation-only August 2014 forcing; never modify training inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from ocean_preprocessing.regridding import (  # type: ignore[import-not-found]
    FiniteCoverageRemap,
    rectilinear_grid,
)

from samudra.observations.prepare import FORCING, save_atomic


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "source", "source-grid", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    proof = json.loads((args.prepared / "COMPLETE.json").read_text())
    for protected in (args.prepared, args.source, Path(proof["daily_root"])):
        left, right = args.output.resolve(), protected.resolve()
        if left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError(
                "Evaluation output must be separate from source/training stores"
            )
    if sha(args.prepared / "grid.npz") != proof["sha256"]["grid.npz"]:
        raise ValueError("Training grid checksum differs")
    grid = dict(np.load(args.prepared / "grid.npz"))
    ds = xr.open_zarr(args.source / "era5-surface.zarr", chunks=None)
    geometry = xr.open_dataset(args.source_grid)
    np.testing.assert_allclose(geometry.lat.values, ds.lat.values)
    np.testing.assert_allclose(geometry.lon.values, ds.lon.values)
    remap = FiniteCoverageRemap(
        rectilinear_grid(
            ds.lat.values, ds.lon.values, geometry.lat_b.values, geometry.lon_b.values
        ),
        rectilinear_grid(grid["lat"], grid["lon"], grid["lat_b"], grid["lon_b"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    signature = dict(
        grid_sha256=sha(args.prepared / "grid.npz"),
        training_proof_sha256=sha(args.prepared / "COMPLETE.json"),
        source_metadata_sha256=sha(args.source / "era5-surface.zarr/.zmetadata"),
        source_grid_sha256=sha(args.source_grid),
        script_sha256=sha(__file__),
    )
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Evaluation preparation identity changed")
    manifest.write_text(json.dumps(signature, indent=2) + "\n")
    completed = args.output / "COMPLETE.json"
    if completed.exists():
        previous = json.loads(completed.read_text())
        if previous["signature"] != signature:
            raise ValueError("Completed cache signature differs")
        for relative, expected in previous["era5_shards"].items():
            if sha(args.output / relative) != expected:
                raise ValueError("Completed evaluation forcing changed")
        for relative, expected in previous["surface_shards"].items():
            if sha(Path(proof["daily_root"]) / relative) != expected:
                raise ValueError("Completed observation reference changed")
        return
    progress_path = args.output / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    times = pd.DatetimeIndex(ds.time.values)
    hashes, surfaces = {}, {}
    for day in pd.date_range("2014-08-01", "2014-09-01", inclusive="left"):
        path = args.output / "era5" / str(day.year) / f"{day:%Y-%m-%d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        relative = str(path.relative_to(args.output))
        if relative in progress:
            if sha(path) != progress[relative]:
                raise ValueError("Prepared evaluation shard changed")
        else:
            fields = []
            for i, name in enumerate(FORCING):
                expected = pd.date_range(
                    day + pd.Timedelta(hours=int(i >= 5)), periods=24, freq="h"
                )
                ids = times.get_indexer(expected)
                if (ids < 0).any():
                    raise ValueError("Missing raw ERA5 support")
                raw = ds[name].isel(time=ids).transpose("time", "lat", "lon").values
                raw = raw.reshape(4, 6, *raw.shape[-2:]).mean(1)
                if i >= 5:
                    raw = raw / 3600 * (1000 if i == 7 else 1)
                values, coverage = remap(raw)
                if not np.isfinite(values).all() or (coverage < 0.9 - 1e-6).any():
                    raise ValueError("Incomplete ERA5 support")
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
        hashes[relative] = sha(path)
        progress[relative] = hashes[relative]
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2) + "\n")
        temporary.replace(progress_path)
        for product in ("sst", "adt"):
            relative = f"{product}/{day.year}/{day:%Y-%m-%d}.npz"
            surfaces[relative] = sha(Path(proof["daily_root"]) / relative)
        print(str(day.date()), flush=True)
    completed = dict(
        signature=signature,
        era5_shards=hashes,
        surface_shards=surfaces,
        forcing_support=proof["forcing_support"],
        daily_root=proof["daily_root"],
    )
    temporary = args.output / "COMPLETE.tmp"
    temporary.write_text(json.dumps(completed, indent=2) + "\n")
    temporary.replace(args.output / "COMPLETE.json")


if __name__ == "__main__":
    main()
