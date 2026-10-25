# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Calendar and forcing invariants imported from the monthly latent setup."""

from types import SimpleNamespace

import numpy as np
import pytest
import zarr  # type: ignore[import-untyped]

from samudra.experiments.monthly_model_data import ModelMonths
from samudra.experiments.shared_era5 import SharedEra5


def test_native_om4_history_cutoff_calendar_and_gap_exclusion(tmp_path):
    root = zarr.open_group(str(tmp_path / "OM4.zarr"), mode="w")
    times = np.delete(np.arange(3000) * 5 + 2.5, 500)
    time = root.create_dataset("time", data=times)
    time.attrs.update(units="days since 1974-01-01", calendar="noleap")
    names = [f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)] + [
        "zos"
    ]
    lat, lon = np.array([-20.0, 20.0]), np.array([0.5, 1.5])
    root.create_dataset("y", data=lat)
    root.create_dataset("x", data=lon)
    for i in range(19):
        root.create_dataset(f"mask_{i}", data=np.ones((2, 2), bool))
    for name in names + ["tauuo", "tauvo", "hfds"]:
        root.create_dataset(name, data=np.full((len(times), 2, 2), 5, np.float32))
    zarr.consolidate_metadata(root.store)
    for suffix, value in (("means", 1), ("stds", 2)):
        store = zarr.open_group(str(tmp_path / f"OM4_{suffix}.zarr"), mode="w")
        for name in ("tauuo", "tauvo", "hfds"):
            store.create_dataset(name, data=np.array(value))
        zarr.consolidate_metadata(store.store)
    grid = dict(
        names=np.array(names), lat=lat, lon=lon, mean=np.ones(77), std=np.full(77, 2)
    )
    reader = ModelMonths(tmp_path / "OM4.zarr", "om4", grid, tmp_path)
    assert reader.excluded_gap_months
    for start, stop, weights, days in reader.windows:
        assert reader.bounds[start - 19, 0].year >= 1975
        stamp = reader.bounds[start, 1]
        assert (stamp.year, stamp.month) <= (2013, 7)
        assert sum(weights) == pytest.approx(1)
        assert days in (28, 30, 31)
        assert all(
            reader.bounds[i, 1] == reader.bounds[i + 1, 0]
            for i in range(start - 19, stop - 1)
        )
    sample = reader.sample(0, "cpu")
    assert sample["fields"].shape[1] == 19 + len(sample["weights"])
    assert sample["fields"].shape[2] == 77
    np.testing.assert_equal(sample["fields"].numpy(), 2)
    np.testing.assert_equal(sample["forcing"].numpy(), 2)


def test_shared_forcing_retains_history_and_observation_scales():
    cache = object.__new__(SharedEra5)
    cache.positions = {i: i for i in range(23)}
    cache.values = np.broadcast_to(np.arange(23)[:, None, None, None], (23, 8, 2, 3))
    cache.mean = np.ones((1, 8, 1, 1)) * 2
    cache.std = np.ones((1, 8, 1, 1)) * 3
    reader = SimpleNamespace(windows=[(19, 23, None, None)])
    actual = cache.load(reader, 0, SimpleNamespace(tensor=lambda x: x))
    assert actual.shape == (1, 23, 8, 2, 3)
    np.testing.assert_allclose(actual[0, :, 0, 0, 0], (np.arange(23) - 2) / 3)
