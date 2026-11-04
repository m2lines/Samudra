# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact native observation/OM4 values, including replay's shifted interval grid."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import zarr  # type: ignore[import-untyped]
from test_observation_baseline_data import model_store  # type: ignore[import-not-found]

from samudra.observations.archive import ObservationArchive, trajectory
from samudra.observations.model_data import ModelMonths
from samudra.observations.native import NativeObservationArchive
from samudra.observations.native_om4 import NativeOm4Planes, prepare_native_om4
from samudra.observations.shared_era5 import digest
from samudra.observations.training import Sources, archive_identity
from samudra.rust_reader import RustIoRuntime

pytest.importorskip("samudra_rust_loader")


def native_archive(tmp_path):
    daily = tmp_path / "daily"
    daily.mkdir()
    np.savez(daily / "grid.npz", lat=[-30, 30], lon=[0, 120, 240])
    for month in ("2000-01", "2000-02"):
        folder = daily / "iap/2000"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{month}.npz",
            values=np.ones((2, 14, 2, 3), np.float32),
            ohc=np.ones((2, 2, 3), np.float32),
        )
    cache = tmp_path / "cache"
    cache.mkdir()
    starts = pd.date_range("1999-09-28", "2000-03-01")
    values = np.arange(len(starts) * 12 * 2 * 3, dtype=np.float32).reshape(-1, 12, 2, 3)
    values[::3, 0, 0, 0] = np.nan
    group = zarr.open_group(str(cache / "intervals.zarr"), mode="w")
    array = group.create_dataset(
        "values", data=values, chunks=(1, 12, 2, 3), fill_value=np.nan
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time", "lev", "y", "x"]
    proof = {
        "version": 1,
        "interval_days": 5,
        "stride_days": 1,
        "intervals": len(starts),
        "channels": ["sst", "adt", "ugos", "vgos", *[f"era5_{i}" for i in range(8)]],
        "first_start": starts[0].isoformat(),
        "last_start": starts[-1].isoformat(),
        "source_grid_sha256": digest(daily / "grid.npz"),
        "source_identity": archive_identity(daily),
        "all_intervals_rust_readback_exact": True,
        "files": {
            str(p.relative_to(cache)): digest(p)
            for p in (cache / "intervals.zarr").rglob("*")
            if p.is_file()
        },
    }
    path = cache / "COMPLETE.json"
    path.write_text(json.dumps(proof))
    native = NativeObservationArchive(daily, cache, RustIoRuntime(2), digest(path))
    python = ObservationArchive(daily)
    slices = {"sst": slice(0, 1), "adt": slice(1, 4), "era5": slice(4, 12)}
    setattr(
        python,
        "interval",
        lambda product, a, b: values[(a - starts[0]).days, slices[product]],
    )
    return python, native, cache


def test_native_observation_fresh_and_shifted_replay_equal_python(tmp_path):
    python, native, _ = native_archive(tmp_path)
    for month in ("2000-01", "2000-02"):
        expected = trajectory(python, "2000-01", target_month=month)
        actual = trajectory(native, "2000-01", target_month=month)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    expected, actual = python.sample("2000-02"), native.sample("2000-02")
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    with pytest.raises(ValueError, match="outside"):
        native.interval("sst", "2000-01-01", "2000-01-05")


def test_native_observation_rejects_changed_chunk(tmp_path):
    python, _, cache = native_archive(tmp_path)
    array = zarr.open_group(str(cache / "intervals.zarr"), mode="r+")["values"]
    array[0, 1, 0, 0] += 1
    with pytest.raises(ValueError, match="content changed"):
        NativeObservationArchive(
            python.root, cache, RustIoRuntime(2), digest(cache / "COMPLETE.json")
        )


@pytest.mark.parametrize("packed", [False, True])
def test_native_om4_keeps_all_channels_and_native_replay_values(tmp_path, packed):
    path, reference = model_store(tmp_path, "om4")
    z = zarr.open_group(str(path), mode="r+")
    for index, name in enumerate(reference["names"]):
        z[name].attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]
        z[name][:] = np.broadcast_to(
            (np.arange(z[name].shape[0], dtype=np.float32) + index)[:, None, None],
            z[name].shape,
        )
    zarr.consolidate_metadata(str(path))
    python = ModelMonths(path, "om4", reference, tmp_path)
    native = ModelMonths(path, "om4", reference, tmp_path, runtime=RustIoRuntime(2))
    if packed:
        cache = tmp_path / "packed"
        prepare_native_om4(path, python, cache, RustIoRuntime(2))
        native.planes = NativeOm4Planes(
            cache, native, RustIoRuntime(2), digest(cache / "COMPLETE.json")
        )
    for month in (0, 1, 5):
        expected = python.continuation(0, month, "cpu")
        actual = native.continuation(0, month, "cpu")
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        sources = object.__new__(Sources)
        setattr(sources, "obs", SimpleNamespace(device="cpu"))
        sources.models = {"om4": native}
        sample = sources.load("om4", 0, month)
        assert "fields" not in sample
        np.testing.assert_array_equal(
            sample["surface"], expected["fields"][:, :, [38, 76]]
        )
        np.testing.assert_array_equal(
            sample["initial_fields"], expected["fields"][:, 17:19]
        )
        np.testing.assert_array_equal(
            sample["target_fields"], expected["fields"][:, 19:]
        )
    if packed:
        proof_path = cache / "COMPLETE.json"
        proof = json.loads(proof_path.read_text())
        proof["indices"].reverse()
        proof_path.write_text(json.dumps(proof))
        with pytest.raises(ValueError, match="cohort"):
            NativeOm4Planes(cache, native, RustIoRuntime(2), digest(proof_path))
        proof["indices"].reverse()
        proof_path.write_text(json.dumps(proof))
        array = zarr.open_group(str(cache / "frames.zarr"), mode="r+")["values"]
        array[0, 1, 0, 0] += 1
        with pytest.raises(ValueError, match="content changed"):
            NativeOm4Planes(cache, native, RustIoRuntime(2), digest(proof_path))
