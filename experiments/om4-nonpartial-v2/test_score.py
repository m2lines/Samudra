# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Small analytical fixture verifies common support, scaling, and NaN rejection."""

import importlib.util
import json
from datetime import timedelta
from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr

spec = importlib.util.spec_from_file_location(
    "experiment_score", Path(__file__).with_name("score.py")
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_common_support_score_and_nonfinite_rejection(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "geometry").mkdir()
    masks = np.ones((19, 2, 2), dtype=bool)
    masks[:, 0, 0] = False
    masks[-1] = False
    np.savez(tmp_path / "geometry/geometry.npz", masks=masks, area=np.ones((2, 2)))
    (tmp_path / "preparation.json").write_text(json.dumps({"data": str(data)}))
    times = [
        cftime.DatetimeJulian(year, 1, 3) + timedelta(days=5 * i)
        for year in [2015, 2017, 2019, 2021]
        for i in range(74)
    ]
    channels = [
        f"{base}_{k}" for base in ["uo", "vo", "thetao", "so"] for k in range(19)
    ] + ["zos"]
    xr.Dataset(
        {
            name: (("time", "y", "x"), np.zeros((296, 2, 2), dtype="float32"))
            for name in channels
        },
        coords={"time": times, "y": [0.0, 1.0], "x": [0.0, 1.0]},
    ).to_zarr(data / "OM4.zarr", consolidated=True)
    xr.Dataset({name: xr.DataArray(2.0) for name in channels}).to_zarr(
        data / "OM4_stds.zarr", consolidated=True
    )
    for offset, year in enumerate([2015, 2017, 2019, 2021]):
        values = np.full((72, 19, 2, 2), 2.0, dtype="float32")
        values[:, :, 0, 0] = np.nan
        values[:, -1] = np.nan
        pred = xr.Dataset(
            {
                base: (("time", "lev", "y", "x"), values)
                for base in ["uo", "vo", "thetao", "so"]
            },
            coords={
                "time": times[offset * 74 + 2 : (offset + 1) * 74],
                "lev": np.arange(19),
                "y": [0.0, 1.0],
                "x": [0.0, 1.0],
            },
        )
        pred["zos"] = (("time", "y", "x"), values[:, 0])
        pred.to_zarr(
            tmp_path / "eval" / f"baseline-{year}" / "predictions.zarr",
            consolidated=True,
        )
    module.score(tmp_path, "baseline")
    summary = json.loads((tmp_path / "scores/baseline.json").read_text())
    assert summary["mean_normalized_rmse"] == pytest.approx(1.0)
    assert summary["mean_persistence_normalized_rmse"] == 0
    assert summary["empty_levels"] == [18]
    import zarr  # type: ignore[import-untyped]

    store = zarr.open(str(tmp_path / "eval/baseline-2015/predictions.zarr"), mode="r+")
    store["thetao"][0, 0, 1, 1] = np.nan
    with pytest.raises(AssertionError, match="invalid prediction"):
        module.score(tmp_path, "baseline")
