# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Materialize and read back eight-field ERA5 on the exact native OM4 intervals."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from samudra.observations.archive import ObservationArchive
from samudra.observations.model_data import ModelMonths
from samudra.observations.shared_era5 import digest, required_indices


def prepare(samples: Path, daily: Path, om4: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    grid = dict(np.load(samples / "grid.npz"))
    reader = ModelMonths(om4 / "OM4.zarr", "om4", grid, om4)
    archive = ObservationArchive(daily)
    for name in ("lat", "lon", "mask"):
        np.testing.assert_array_equal(grid[name], archive.grid[name])
    indices = required_indices(reader)
    values = np.lib.format.open_memmap(
        output / "values.npy",
        mode="w+",
        dtype="float32",
        shape=(len(indices), 8, *grid["mask"].shape[-2:]),
    )
    bounds = []
    files = {}
    for position, index in enumerate(indices):
        start, end = [pd.Timestamp(str(t)) for t in reader.bounds[index]]
        if start != start.normalize() or end != end.normalize():
            raise ValueError("ERA5 daily support requires midnight interval boundaries")
        raw = archive.interval("era5", start, end)
        if not np.isfinite(raw).all():
            raise ValueError(f"Missing ERA5 support at {start}")
        values[position] = raw
        values.flush()
        np.testing.assert_array_equal(
            np.load(output / "values.npy", mmap_mode="r")[position], raw
        )
        bounds.append([start.isoformat(), end.isoformat()])
        for day in pd.date_range(start, end - pd.Timedelta(days=1)):
            path = daily / "era5" / str(day.year) / (day.strftime("%Y-%m-%d") + ".npz")
            if str(path) not in files:
                files[str(path)] = digest(path)
        if position % 100 == 0:
            print(
                json.dumps({"intervals": position + 1, "total": len(indices)}),
                flush=True,
            )
    np.save(output / "indices.npy", np.asarray(indices, dtype=np.int64))
    proof = {
        "windows": len(reader.windows),
        "bounds": bounds,
        "all_intervals_readback_exact": True,
        "contracts": {
            str(samples / "grid.npz"): digest(samples / "grid.npz"),
            str(om4 / "OM4.zarr" / ".zmetadata"): reader.source_metadata_sha256,
        },
        "files": {
            name: digest(output / name) for name in ("indices.npy", "values.npy")
        },
        "source_sha256": files,
    }
    (output / "COMPLETE.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({"proof_sha256": digest(output / "COMPLETE.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("samples", "daily", "om4", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.samples, args.daily, args.om4, args.output)
