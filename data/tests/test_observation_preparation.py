# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Repository entrypoints run using the data environment, without importing Torch."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from ocean_preprocessing.vertical_grid import OM4_DEPTH_LEVELS

from tests import requires_xesmf

REPOSITORY = Path(__file__).parents[2]


def run_preparation(*arguments: str) -> None:
    environment = dict(os.environ, PYTHONPATH=str(REPOSITORY / "src"))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy,sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__'); assert 'torch' not in sys.modules",
            str(REPOSITORY / "data/prepare_observation_baseline.py"),
            *arguments,
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


@requires_xesmf
def test_grid_daily_and_months_entrypoints_in_data_environment(tmp_path):
    lat, lon = np.array([-45.0, 45.0]), np.array([45.0, 135.0, 225.0, 315.0])
    native = zarr.open_group(str(tmp_path / "OM4.zarr"), mode="w")
    native["y"], native["x"] = lat, lon
    for depth in range(19):
        native[f"mask_{depth}"] = np.ones((2, 4), dtype=bool)
    zarr.consolidate_metadata(str(tmp_path / "OM4.zarr"))
    output = tmp_path / "prepared"
    run_preparation(
        "grid",
        "--om4",
        str(tmp_path / "OM4.zarr"),
        "--output",
        str(output / "grid.npz"),
    )
    with np.load(output / "grid.npz") as grid:
        assert grid["depth"].shape == (19,)
        assert grid["depth"][-1] == 6000
        assert grid["mask"].shape == (77, 2, 4)
    products = tmp_path / "products"
    xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.full((2, 2, 4), 7.0, dtype=np.float32))},
        coords={"time": pd.date_range("2013-10-01", periods=2), "lat": lat, "lon": lon},
    ).to_zarr(products / "oisst.zarr", consolidated=True)
    run_preparation(
        "daily",
        "--source",
        str(products),
        "--grid",
        str(output / "grid.npz"),
        "--output",
        str(output / "daily"),
        "--product",
        "sst",
        "--year",
        "2013",
    )
    with np.load(output / "daily/sst/2013/2013-10-01.npz") as shard:
        np.testing.assert_allclose(shard["values"], 7)
        np.testing.assert_allclose(shard["coverage"], 1)
    xr.Dataset(
        {
            "temp": (
                ("time", "depth", "lat", "lon"),
                np.full((1, 14, 2, 4), 7.0, dtype=np.float32),
            ),
            "salt": (
                ("time", "depth", "lat", "lon"),
                np.full((1, 14, 2, 4), 35.0, dtype=np.float32),
            ),
        },
        coords={
            "time": [pd.Timestamp("2013-10-01")],
            "depth": OM4_DEPTH_LEVELS[:14],
            "lat": lat,
            "lon": lon,
        },
    ).to_zarr(products / "argo-iap.zarr", consolidated=True)
    run_preparation(
        "daily",
        "--source",
        str(products),
        "--grid",
        str(output / "grid.npz"),
        "--output",
        str(output / "daily"),
        "--product",
        "iap",
        "--year",
        "2013",
    )
    with np.load(output / "daily/iap/2013/2013-10.npz") as shard:
        np.testing.assert_array_equal(shard["depth"], OM4_DEPTH_LEVELS[:14])
        np.testing.assert_allclose(shard["values"][0], 7)
        assert np.isfinite(shard["values"][1]).any()
        assert not np.allclose(shard["values"][1], 35, equal_nan=True)
    # Populate the small common-grid input contract for the complete nine-month
    # validation materialization, including all 95-day input histories.
    for day in pd.date_range("2013-07-29", "2014-08-05"):
        for product, channels in (("sst", 1), ("adt", 3), ("era5", 8)):
            path = output / "daily" / product / str(day.year) / f"{day:%Y-%m-%d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, values=np.ones((channels, 2, 4), dtype=np.float32))
    for month in pd.period_range("2013-11", "2014-07", freq="M"):
        path = output / "daily/iap" / str(month.year) / f"{month}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            values=np.ones((2, 14, 2, 4), dtype=np.float32),
            ohc=np.ones((2, 2, 4), dtype=np.float32),
        )
    run_preparation("months", "--root", str(output), "--split", "validation")
    assert len(list((output / "validation").glob("*.npz"))) == 9
    with np.load(output / "validation/2013-11.npz") as sample:
        assert sample["surface"].shape == (25, 2, 2, 4)
        np.testing.assert_allclose(sample["month_weights"].sum(), 1)
