# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from samudra.experiments.observation_data import SPLITS, month_intervals
from samudra.experiments.observation_prepare import prepare
from scripts.prepare_quarter_observations import (
    ANNUAL_ORIGINS,
    check_receipt,
    digest,
    execute,
    required_calendar,
    source_indices,
)


def test_calendar_exact_monthly_and_annual_union():
    calendar = required_calendar()
    expected: set[pd.Timestamp] = set()
    for bounds in SPLITS.values():
        for month in pd.period_range(*bounds, freq="M"):
            starts, ends, _ = month_intervals(str(month))
            for start, end in zip(starts, ends, strict=True):
                expected.update(pd.date_range(start, end, inclusive="left", freq="D"))
    monthly = expected.copy()
    for origin in pd.DatetimeIndex(ANNUAL_ORIGINS):
        for index in range(-19, 73):
            expected.update(
                pd.date_range(origin + pd.Timedelta(days=index * 5), periods=5)
            )
    assert expected == monthly  # Annual windows add no source days.
    assert set(pd.DatetimeIndex(calendar["daily_dates"])) == expected
    assert len(expected) == 10882
    assert {key: len(value) for key, value in calendar["months"].items()} == {
        "train": 243,
        "validation": 9,
        "test": 96,
    }
    assert "2014-08-05" not in calendar["daily_dates"]
    assert "2023-01-04" in calendar["daily_dates"]
    assert "2023-01-05" not in calendar["daily_dates"]


def test_source_audit_requires_ending_hour_and_unique_daily_analysis():
    hours = pd.date_range("2023-01-04", periods=25, freq="h")
    np.testing.assert_array_equal(
        source_indices(hours, "era5", ["2023-01-04"]), np.arange(25)
    )
    with pytest.raises(ValueError, match="ending-hour"):
        source_indices(hours[:-1], "era5", ["2023-01-04"])
    assert source_indices(["2023-01-04T12:00:00"], "sst", ["2023-01-04"]).tolist() == [
        0
    ]
    with pytest.raises(ValueError, match="one sst analysis"):
        source_indices(hours[:2], "sst", ["2023-01-04"])
    assert source_indices(["1993-05-15"], "iap", ["1993-05-01"]).tolist() == [0]


def test_selected_dates_preserve_daily_conversion_and_default(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    dates = pd.date_range("2001-01-01T12:00:00", periods=4)
    raw = np.arange(32, dtype=np.float32).reshape(4, 2, 4)
    raw[2, 0, 0] = np.nan
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), raw)},
        coords={"time": dates, "lat": [-45, 45], "lon": [45, 135, 225, 315]},
    )
    ds.to_zarr(source / "oisst.zarr", consolidated=True)
    grid = tmp_path / "grid.npz"
    np.savez(grid, lat=[-45, 45], lon=[45, 135, 225, 315])
    base = dict(source=str(source), grid=str(grid), product="sst", year=2001, limit=0)
    prepare(argparse.Namespace(**base, output=str(tmp_path / "legacy")))
    prepare(
        argparse.Namespace(
            **base, output=str(tmp_path / "selected"), selected_dates=["2001-01-03"]
        )
    )
    assert len(list((tmp_path / "legacy/sst/2001").glob("*.npz"))) == 4
    assert len(list((tmp_path / "selected/sst/2001").glob("*.npz"))) == 1
    with (
        np.load(tmp_path / "legacy/sst/2001/2001-01-03.npz") as old,
        np.load(tmp_path / "selected/sst/2001/2001-01-03.npz") as new,
    ):
        for name in old.files:
            np.testing.assert_array_equal(old[name], new[name])
    with pytest.raises(ValueError, match="Missing selected"):
        prepare(
            argparse.Namespace(
                **base, output=str(tmp_path / "missing"), selected_dates=["2001-01-05"]
            )
        )


def test_tasks_require_matching_smoke_receipt_and_detect_changed_shard(
    tmp_path, monkeypatch
):
    plan = {"output": str(tmp_path), "tasks": [{}]}
    monkeypatch.setattr(
        "scripts.prepare_quarter_observations.check_inputs", lambda plan: None
    )
    (tmp_path / "PLAN.json").write_text(json.dumps(plan))
    with pytest.raises(FileNotFoundError):
        execute(plan, 0)
    shard = tmp_path / "shard.npz"
    shard.write_bytes(b"original")
    ready = tmp_path / "SMOKE_READY.json"
    ready.write_text(
        json.dumps(
            {
                "plan_sha256": digest(tmp_path / "PLAN.json"),
                "records": [{"file": str(shard), "sha256": digest(shard)}],
            }
        )
    )
    check_receipt(ready, plan)
    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="shard changed"):
        check_receipt(ready, plan)


def test_audit_never_reads_field_payloads(tmp_path, monkeypatch):
    import zarr  # type: ignore[import-untyped]

    from scripts import prepare_quarter_observations as orchestration

    calendar = {
        "months": {"train": ["1993-05"]},
        "daily_dates": ["1993-05-01"],
        "annual_origins": [],
        "annual_additional_days": 0,
        "monthly_bins": 26,
    }
    monkeypatch.setattr(orchestration, "required_calendar", lambda: calendar)
    source = tmp_path / "source"
    source.mkdir()
    fields = set()
    for product, (store, variables) in orchestration.PRODUCTS.items():
        times = (
            pd.date_range("1993-05-01", periods=25, freq="h")
            if product == "era5"
            else pd.DatetimeIndex(["1993-05-01"])
        )
        ds = xr.Dataset(
            {
                name: (
                    ("time", "lat", "lon"),
                    np.ones((len(times), 2, 4), dtype=np.float32),
                )
                for name in variables
            },
            coords={"time": times, "lat": [-45, 45], "lon": [45, 135, 225, 315]},
        )
        ds.to_zarr(source / f"{store}.zarr", consolidated=True)
        fields.update(variables)
    original = zarr.storage.DirectoryStore.__getitem__

    def guarded(store, key):
        parts = key.split("/")
        if parts[0] in fields and not parts[-1].startswith("."):
            raise AssertionError(f"Temporal payload read: {key}")
        return original(store, key)

    monkeypatch.setattr(zarr.storage.DirectoryStore, "__getitem__", guarded)
    grid = tmp_path / "grid.npz"
    np.savez(grid, lat=np.linspace(-89, 89, 720), lon=np.arange(1440) / 4)
    plan = orchestration.audit(source, grid, tmp_path / "out")
    assert len(plan["tasks"]) == 4
    assert plan["sources"]["era5"]["required_source_timestamps"] == 25


def test_smoke_readback_rejects_nonfinite_forcing_and_bad_timestamp(tmp_path):
    from scripts.prepare_quarter_observations import verify_shard

    path = tmp_path / "era5.npz"
    values = np.ones((8, 720, 1440), dtype=np.float32)
    values[0, 1, 2] = np.nan
    np.savez_compressed(path, values=values, time=np.array("1993-05-01T00:00:00"))
    with pytest.raises(ValueError, match="Nonfinite forcing"):
        verify_shard(path, "era5", "1993-05-01")
    with pytest.raises(ValueError, match="Unexpected timestamp"):
        verify_shard(path, "era5", "1993-05-02")
