# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild representative products/months and compare every array to training data."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from samudra.observations.archive import ObservationArchive
from samudra.observations.prepare import prepare


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(actual, path):
    with np.load(path) as reference:
        if set(actual) != set(reference.files):
            raise ValueError(f"Array keys differ: {path}")
        for key in reference.files:
            x, y = np.asarray(actual[key]), reference[key]
            if x.dtype != y.dtype:
                raise ValueError(f"Dtype differs: {path}:{key}")
            np.testing.assert_array_equal(x, y, err_msg=f"{path}:{key}", strict=True)
    return {
        "path": str(path),
        "sha256": digest(path),
        "arrays": sorted(actual),
        "comparison": "exact, including NaN support and timestamps",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--daily", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for product in ["sst", "adt", "era5", "iap"]:
        prepare(
            SimpleNamespace(
                source=args.source,
                grid=args.daily / "grid.npz",
                output=args.output / "regenerated",
                product=product,
                year=1993,
                limit=1,
            )
        )
        for f in (args.output / "regenerated" / product / "1993").glob("*.npz"):
            with np.load(f) as a:
                reports.append(compare(dict(a), args.daily / product / "1993" / f.name))
    archive = ObservationArchive(args.daily)
    for split, month in [
        ("train", "1993-05"),
        ("train", "2000-02"),
        ("validation", "2013-11"),
        ("test", "2022-12"),
    ]:
        reports.append(
            compare(archive.sample(month), args.samples / split / (month + ".npz"))
        )
    sources = {
        p.name: digest(p / ".zmetadata")
        for p in args.source.glob("*.zarr")
        if p.name in ["oisst.zarr", "duacs.zarr", "era5-surface.zarr", "argo-iap.zarr"]
    }
    result = {
        "comparisons": reports,
        "source_metadata_sha256": sources,
        "grid_sha256": digest(args.daily / "grid.npz"),
        "scope": "Four product shards from upstream stores plus four monthly examples from the original daily archive; not exhaustive reprocessing.",
    }
    (args.output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
