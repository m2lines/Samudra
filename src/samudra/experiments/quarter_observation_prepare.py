# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare quarter-degree surface inputs while retaining native IAP labels.

The existing daily preparation implements the scientific conversions and strict
temporal support. Only IAP's horizontal destination changes: monthly labels stay
on their native half-degree grid, for use with NativeIapOperator. This command
does not materialize training samples or fit normalization statistics.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

from samudra.experiments.observation_prepare import DEPTHS, prepare, save_atomic


def prepare_quarter(args):
    with np.load(args.grid) as grid:
        if (len(grid["lat"]), len(grid["lon"])) != (720, 1440):
            raise ValueError("Expected the OM4 quarter-degree 720x1440 grid")
    if args.product != "iap":
        return prepare(args)

    source = Path(args.source) / "argo-iap.zarr"
    with xr.open_zarr(source, chunks=None) as ds:
        lat, lon = ds.lat.values.copy(), ds.lon.values.copy()
    if (len(lat), len(lon)) != (360, 720):
        raise ValueError("Expected native IAP half-degree 360x720 source")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    native_grid = root / "iap-grid.npz"
    if native_grid.exists():
        with np.load(native_grid) as saved:
            np.testing.assert_array_equal(saved["lat"], lat)
            np.testing.assert_array_equal(saved["lon"], lon)
            np.testing.assert_array_equal(saved["depth"], DEPTHS)
    else:
        save_atomic(native_grid, lat=lat, lon=lon, depth=DEPTHS)
    contract = {
        "source": str(source),
        "source_metadata_sha256": hashlib.sha256(
            (source / ".zmetadata").read_bytes()
        ).hexdigest(),
        "model_grid_sha256": hashlib.sha256(Path(args.grid).read_bytes()).hexdigest(),
        "iap_grid_sha256": hashlib.sha256(native_grid.read_bytes()).hexdigest(),
        "preparation_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "horizontal_labels": "native IAP half-degree cells; no upsampling",
        "vertical_labels": "existing 14 OM4 depth centers; strict bracket interpolation",
        "operator": "NativeIapOperator: physical model means projected to IAP cells",
        "minimum_model_wet_area_coverage": 0.9,
        "normalization": "not fitted by this preparation command",
    }
    path = root / "iap-operator.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Existing native IAP preparation contract differs")
    path.write_text(json.dumps(contract, indent=2) + "\n")
    native_args = argparse.Namespace(**vars(args))
    native_args.grid = str(native_grid)
    return prepare(native_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--product", choices=["sst", "adt", "era5", "iap"], required=True
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    prepare_quarter(parser.parse_args())


if __name__ == "__main__":
    main()
