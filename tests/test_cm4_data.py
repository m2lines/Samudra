# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import datetime
from pathlib import Path

import cftime
import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from samudra.config import Cm4DataSourceConfig, Cm4TimeConfig, DataConfig
from samudra.data_backend import NativeOm4SourceBackend
from samudra.utils.location import LocalLocation


def _config() -> dict[str, str | dict[str, str]]:
    return {
        "type": "cm4",
        "train_time": {"start": "0151-01-06", "end": "0151-01-16"},
        "val_time": {"start": "0151-01-21", "end": "0151-01-26"},
        "inference_time": {"start": "0151-01-31", "end": "0151-02-05"},
        "data_location": "CM4.zarr",
        "data_means_location": "CM4_means.zarr",
        "data_stds_location": "CM4_stds.zarr",
    }


def test_cm4_dates_use_no_leap_midnight_and_roundtrip() -> None:
    data = DataConfig.model_validate({"sources": [_config()]})
    src = data.sources[0]
    assert isinstance(src, Cm4DataSourceConfig)
    assert isinstance(src.train_time, Cm4TimeConfig)
    assert src.train_time.start.datetime == cftime.DatetimeNoLeap(151, 1, 6)
    assert src.train_time.start.datetime.hour == 0
    assert src.inference_time is not None
    assert src.inference_time.end.datetime == cftime.DatetimeNoLeap(151, 2, 5)
    assert (
        DataConfig.model_validate_json(data.model_dump_json()).model_dump()
        == data.model_dump()
    )
    assert src.build_layout().boundary_var_names == ["tauuo", "tauvo", "hfds", "wfo"]


def test_cm4_dates_reject_leap_days() -> None:
    with pytest.raises(ValidationError):
        Cm4TimeConfig.model_validate({"start": "0200-02-29", "end": "0200-03-06"})


def test_cm4_rejects_overlapping_splits() -> None:
    cfg = _config()
    cfg["val_time"] = {"start": "0151-01-11", "end": "0151-01-26"}
    with pytest.raises(ValidationError, match="overlaps"):
        Cm4DataSourceConfig.model_validate(cfg)


def test_cm4_loads_compact_fields_and_slices_native_dates(tmp_path: Path) -> None:
    times = [
        cftime.DatetimeNoLeap(151, 1, 6) + datetime.timedelta(days=i * 5)
        for i in range(8)
    ]
    data = xr.Dataset(
        {
            name: (("time", "lev", "y", "x"), np.ones((8, 19, 2, 3), dtype="float32"))
            for name in ["thetao", "so", "uo", "vo"]
        }
        | {
            name: (("time", "y", "x"), np.ones((8, 2, 3), dtype="float32"))
            for name in ["zos", "tauuo", "tauvo", "hfds", "wfo"]
        },
        coords={
            "time": times,
            "lev": np.arange(19),
            "x": [0.0, 1.0, 2.0],
            "y": [-1.0, 1.0],
            "wetmask": (("lev", "y", "x"), np.ones((19, 2, 3), dtype=bool)),
        },
    )
    stats = data.mean(["time", "x", "y"])
    data.chunk({"time": 1, "lev": -1}).to_zarr(tmp_path / "CM4.zarr", consolidated=True)
    stats.to_zarr(tmp_path / "CM4_means.zarr", consolidated=True)
    stats.to_zarr(tmp_path / "CM4_stds.zarr", consolidated=True)
    cfg = DataConfig.model_validate({"sources": [_config()]})
    bundle = cfg.build(LocalLocation(path=tmp_path), include_inference=True)
    assert bundle.train_sources[0].time.size == 3
    assert bundle.val_sources[0].time.size == 2
    assert bundle.inference_source is not None
    assert bundle.inference_source.time.size == 2
    source = bundle.train_sources[0]
    assert source.time.values[0] == times[0]
    assert source.time.values[-1] == times[2]
    values = source.read(np.array([0]), ["thetao_0", "wfo"])
    np.testing.assert_array_equal(values, np.ones((1, 2, 2, 3)))

    # A native reader uses the same physical schema; source_type must not
    # reject CM4 before that reader is constructed.
    NativeOm4SourceBackend(
        "rust", lambda: pytest.fail("Validation must not open the reader")
    ).validate_source(
        data_location=LocalLocation(path=tmp_path / "CM4.zarr"),
        means_location=LocalLocation(path=tmp_path / "CM4_means.zarr"),
        stds_location=LocalLocation(path=tmp_path / "CM4_stds.zarr"),
        source_type="cm4",
        data_layout=cfg.sources[0].build_layout(),
    )


def test_cm4_rejects_a_dataset_on_the_wrong_calendar() -> None:
    cfg = Cm4DataSourceConfig.model_validate(_config())
    data = xr.Dataset(coords={"time": [cftime.DatetimeJulian(151, 1, 6)]})
    with pytest.raises(ValueError, match="no-leap calendar"):
        cfg.canonicalize_datasets(data, xr.Dataset(), xr.Dataset())
