# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Failure and resume checks for no-raw-retention surface acquisition."""

import json

import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr
from ocean_preprocessing.obs_preprocessing import full_range as fr
from ocean_preprocessing.obs_preprocessing import surface as s


def fixture_source(tmp_path, monkeypatch, count=25):
    times = pd.date_range("1993-01-01", periods=count, freq="D")
    values = np.arange(count * 12, dtype="float64").reshape(count, 3, 4)
    values[:, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            name: (("time", "latitude", "longitude"), values.copy(), {"units": "m"})
            for name in ("adt", "sla")
        },
        coords=dict(
            time=times,
            latitude=[30.0, 0.0, -30.0],
            longitude=[-135.0, -45.0, 45.0, 135.0],
        ),
    )
    norm = s.prep._standardize_daily(ds)
    plan = dict(
        schema_version=1,
        product="duacs-ssh",
        variables=["adt", "sla"],
        source_uri="https://example.test/store",
        source_metadata={
            ".zattrs": {},
            "adt/.zattrs": {"units": "m"},
            "sla/.zattrs": {"units": "m"},
        },
        frequency="D",
        start_date=str(times[0]),
        end_date=str(times[-1]),
        time_count=count,
        coordinate_sha256=s._coordinates(ds),
        sizes=dict(norm.sizes),
        normalized_coordinates={
            n: s._array_hash(norm[n].values) for n in ("lat", "lon")
        },
        decoded_dtypes={n: str(norm[n].dtype) for n in ("adt", "sla")},
        processing="test",
    )
    manifest = tmp_path / "manifest.json"
    fr._atomic_json(manifest, plan)
    monkeypatch.setattr(s, "_open", lambda plan: ds.copy(deep=True))
    monkeypatch.setattr(s, "_verify_source", lambda plan: None)
    return manifest, norm, plan


def test_exact_values_resume_and_partial_final_chunk(tmp_path, monkeypatch):
    manifest, expected, _ = fixture_source(tmp_path, monkeypatch)
    output = tmp_path / "out"
    assert s.run(str(manifest), str(output), max_seconds=0) is False
    assert not (output / "duacs-ssh.zarr").exists()
    assert s.run(str(manifest), str(output)) is True
    with xr.open_zarr(output / "duacs-ssh.zarr") as actual:
        for name in expected.variables:
            xr.testing.assert_equal(actual[name], expected[name])
    assert s.run(str(manifest), str(output)) is True


def test_interrupted_write_replays_uncommitted_block(tmp_path, monkeypatch):
    manifest, expected, _ = fixture_source(tmp_path, monkeypatch)
    original = xr.Dataset.to_zarr
    calls = 0

    def interrupted(self, *args, **kwargs):
        nonlocal calls
        result = original(self, *args, **kwargs)
        if "region" in kwargs:
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption after write")
        return result

    monkeypatch.setattr(xr.Dataset, "to_zarr", interrupted)
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="interruption"):
        s.run(str(manifest), str(out))
    state = json.loads((out / ".duacs-ssh.progress.json").read_text())
    assert state["written"] == 24
    monkeypatch.setattr(xr.Dataset, "to_zarr", original)
    assert s.run(str(manifest), str(out))
    with xr.open_zarr(out / "duacs-ssh.zarr") as actual:
        xr.testing.assert_equal(actual.adt, expected.adt)


def test_changed_processing_code_rejects_resume(tmp_path, monkeypatch):
    manifest, _, _ = fixture_source(tmp_path, monkeypatch)
    out = tmp_path / "out"
    s.run(str(manifest), str(out), max_seconds=0)
    monkeypatch.setattr(s, "_code_hash", lambda: "changed")
    with pytest.raises(ValueError, match="processing code mismatch"):
        s.run(str(manifest), str(out))


def test_missing_chunk_raises_instead_of_fill(monkeypatch):
    def missing(url):
        raise OSError("HTTP 404")

    monkeypatch.setattr(s, "_fetch", missing)
    store = s.StrictHTTPStore("https://example.test", {})
    with pytest.raises(OSError, match="404"):
        store["adt/0.0.0"]
    with pytest.raises(KeyError):
        store[".missing_metadata"]


def test_http_403_not_retried(monkeypatch):
    class Session:
        calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            response = requests.Response()
            response.status_code = 403
            return response

    session = Session()
    monkeypatch.setattr(s._local, "session", session, raising=False)
    with pytest.raises(OSError, match="403"):
        s._fetch("https://example.test")
    assert session.calls == 1


def test_merge_preserves_velocity_files_and_rejects_grid_change(tmp_path, monkeypatch):
    manifest, expected, plan = fixture_source(tmp_path, monkeypatch)
    stage = tmp_path / "stage"
    s.run(str(manifest), str(stage))
    store = tmp_path / "prepared" / "duacs.zarr"
    base = xr.Dataset({"ugos": expected.adt.copy()}, coords=expected.coords)
    original = dict(
        schema_version=1,
        product="duacs",
        dataset_version="202411",
        variables=["ugos"],
        frequency="D",
        start_date=plan["start_date"],
        end_date=plan["end_date"],
        time_count=plan["time_count"],
    )
    base.attrs["m2lines/inventory_sha256"] = fr._digest(original)
    base.to_zarr(store, consolidated=True)
    velocity_bytes = {p.name: p.read_bytes() for p in (store / "ugos").iterdir()}
    original_path = tmp_path / "original.json"
    combined_path = tmp_path / "combined.json"
    fr._atomic_json(original_path, original)
    args = (
        str(manifest),
        str(stage / "duacs-ssh.zarr"),
        str(original_path),
        str(combined_path),
        str(store),
    )
    s.merge_duacs(*args)
    s.merge_duacs(*args)
    assert velocity_bytes == {
        p.name: p.read_bytes() for p in (store / "ugos").iterdir()
    }
    with xr.open_zarr(store) as actual:
        assert set(actual.data_vars) == {"ugos", "adt", "sla"}
        xr.testing.assert_equal(actual.adt, expected.adt)
    assert fr.validate(str(combined_path), str(store))["time_count"] == 25


def test_merge_rejects_mismatched_coordinates(tmp_path, monkeypatch):
    manifest, expected, plan = fixture_source(tmp_path, monkeypatch)
    stage = tmp_path / "stage"
    s.run(str(manifest), str(stage))
    original = dict(
        schema_version=1,
        product="duacs",
        dataset_version="202411",
        variables=["ugos"],
        frequency="D",
        start_date=plan["start_date"],
        end_date=plan["end_date"],
        time_count=plan["time_count"],
    )
    original_path = tmp_path / "original.json"
    fr._atomic_json(original_path, original)
    base = xr.Dataset({"ugos": expected.adt.copy()}, coords=expected.coords)
    base = base.assign_coords(lon=base.lon + 0.1)
    base.attrs["m2lines/inventory_sha256"] = fr._digest(original)
    store = tmp_path / "prepared" / "duacs.zarr"
    base.to_zarr(store, consolidated=True)
    with pytest.raises(AssertionError):
        s.merge_duacs(
            str(manifest),
            str(stage / "duacs-ssh.zarr"),
            str(original_path),
            str(tmp_path / "combined.json"),
            str(store),
        )
    assert not (store / "adt").exists()
