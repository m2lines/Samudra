# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare exact five-day observation planes for the shared Rust Zarr reader."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import zarr  # type: ignore[import-untyped]
from numcodecs import Blosc  # type: ignore[import-untyped]

from samudra.native_reader import PhysicalVariable
from samudra.observations.archive import ObservationArchive
from samudra.observations.training import archive_identity, digest
from samudra.rust_reader import RustIoRuntime


def prepare(daily: Path, output: Path, first: str, last: str) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to replace observation cache: {output}")
    archive = ObservationArchive(daily)
    starts = pd.date_range(first, last, freq="D")
    if not len(starts):
        raise ValueError("Empty interval range")
    source_identity = archive_identity(daily)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output.name + ".building-", dir=output.parent)
    )
    store = staging / "intervals.zarr"
    group = zarr.open_group(str(store), mode="w")
    shape = archive.day("sst", str(starts[0].date())).shape[-2:]
    array = group.create_dataset(
        "values",
        shape=(len(starts), 12, *shape),
        chunks=(1, 12, *shape),
        dtype="f4",
        fill_value=np.nan,
        compressor=Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE),
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "lev", "y", "x"]
    selectors = [PhysicalVariable("values", i) for i in range(12)]
    reader = RustIoRuntime(4).open(store, selectors)
    actual = np.empty((1, 12, *shape), np.float32)
    for i, start in enumerate(starts):
        end = start + pd.Timedelta(days=5)
        expected = np.concatenate(
            [
                archive.interval(product, start, end)
                for product in ("sst", "adt", "era5")
            ]
        )
        array[i] = expected
        reader.read_into([i], selectors, actual)
        np.testing.assert_array_equal(actual[0], expected, err_msg=str(start))
        if i % 100 == 0:
            print(
                json.dumps({"interval": i, "total": len(starts), "start": str(start)}),
                flush=True,
            )
    zarr.consolidate_metadata(str(store))
    if archive_identity(daily) != source_identity:
        raise ValueError("Observation source changed during preparation")
    proof = {
        "version": 1,
        "source": str(daily),
        "source_identity": source_identity,
        "source_grid_sha256": digest(daily / "grid.npz"),
        "first_start": starts[0].isoformat(),
        "last_start": starts[-1].isoformat(),
        "interval_days": 5,
        "stride_days": 1,
        "intervals": len(starts),
        "channels": ["sst", "adt", "ugos", "vgos", *[f"era5_{i}" for i in range(8)]],
        "all_intervals_rust_readback_exact": True,
        "files": {
            str(p.relative_to(staging)): digest(p)
            for p in sorted(store.rglob("*"))
            if p.is_file()
        },
    }
    (staging / "COMPLETE.json").write_text(json.dumps(proof, indent=2) + "\n")
    staging.rename(output)
    print(
        json.dumps(
            {"output": str(output), "proof_sha256": digest(output / "COMPLETE.json")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("daily", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first", default="1993-01-26")
    parser.add_argument("--last", default="2014-12-27")
    args = parser.parse_args()
    prepare(args.daily, args.output, args.first, args.last)
