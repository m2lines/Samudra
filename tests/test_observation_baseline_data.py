# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Calendar/missingness contracts that make monthly replay scientifically valid."""

import importlib.util
import json
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import pytest
import zarr  # type: ignore[import-untyped]

from samudra.constants import build_om4_layout
from samudra.observations.archive import month_intervals, strict_mean, trajectory
from samudra.observations.intervals import calendar_weights, replay_month
from samudra.observations.model_data import ModelMonths
from samudra.observations.prepare import interpolate_depth


def test_replay_recomputes_straddling_bin_and_exact_month_weights():
    start, stop, next_start = replay_month("2000-01-01", "2000-02-01")
    assert (start, stop, next_start) == (6, 12, 12)
    starts = pd.date_range("2000-01-31", periods=6, freq="5D")
    weights = calendar_weights(
        starts,
        starts + pd.Timedelta(days=5),
        pd.Timestamp("2000-02-01"),
        pd.Timestamp("2000-03-01"),
    )
    np.testing.assert_allclose(weights, np.array([4, 5, 5, 5, 5, 5]) / 29)
    with pytest.raises(ValueError, match="complete"):
        calendar_weights(
            starts[1:],
            starts[1:] + pd.Timedelta(days=5),
            pd.Timestamp("2000-02-01"),
            pd.Timestamp("2000-03-01"),
        )


def test_fresh_month_matches_original_lattice():
    starts, ends, weights = month_intervals("1993-05")
    a, b, _ = replay_month("1993-05", "1993-05")
    assert (a, b) == (0, 7)
    assert starts[0] == pd.Timestamp("1993-01-26")
    np.testing.assert_allclose(weights, [5 / 31] * 6 + [1 / 31])
    assert ends[-1] == pd.Timestamp("1993-06-05")


def test_missing_temporal_and_depth_support_never_invents_labels():
    assert np.isnan(strict_mean(np.array([[1.0, np.nan], [3.0, 4.0]]))[1])
    np.testing.assert_allclose(strict_mean(np.array([[1.0], [3.0]])), [2.0])
    values = np.array([1.0, np.nan, 3.0])[:, None, None]
    actual = interpolate_depth(values, [0, 10, 20], [5, 20, 25])
    assert np.isnan(actual[0]).all() and np.isnan(actual[2]).all()
    assert actual[1, 0, 0] == 3


def test_train_and_validation_boundaries_rejected_before_io():
    class NoIO:
        def interval(self, *args):
            raise AssertionError("Boundary must reject before reading")

    with pytest.raises(ValueError, match="split boundary"):
        trajectory(NoIO(), "2013-07-01", target_month="2013-08-01")
    with pytest.raises(ValueError, match="split boundary"):
        trajectory(NoIO(), "2014-11-01", steps=73, split="validation")


def test_observation_normalization_deep_slots_and_surface_override():
    path = Path(__file__).parents[1] / "data/prepare_observation_normalization.py"
    spec = importlib.util.spec_from_file_location("baseline_normalization", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mean, std = module.state_statistics(
        dict(
            interior_mean=np.arange(28),
            interior_std=np.arange(28) + 1,
            surface_mean=[99, 100],
            surface_std=[10, 20],
        )
    )
    np.testing.assert_array_equal(mean[:38], 0)
    np.testing.assert_array_equal(std[:38], 1)
    np.testing.assert_array_equal(mean[52:57], 13)
    np.testing.assert_array_equal(std[71:76], 28)
    assert (mean[38], mean[76]) == (99, 100)


def model_store(tmp_path, kind):
    root = tmp_path / kind
    z = zarr.open_group(str(root), mode="w")
    lat, lon = np.array([-30.0, 30.0]), np.array([0.0, 120.0, 240.0])
    n = 219
    if kind == "cm4":
        z["lat"] = np.broadcast_to(lat[:, None], (2, 3))
        z["lon"] = np.broadcast_to(lon[None], (2, 3))
        z["dz"] = np.asarray(build_om4_layout().depth_thickness)
        z["wetmask"] = np.ones((19, 2, 3), dtype=bool)
        time = np.arange(1, n + 1, dtype=np.float64) * 5
        z["time_bnds"] = np.column_stack((time - 5, time))
        attrs = dict(units="days since 0001-01-01", calendar="noleap")
        z["time_bnds"].attrs.update(attrs)
        for name in ("uo", "vo", "thetao", "so"):
            z[name] = np.ones((n, 19, 2, 3), dtype=np.float32)
        z["zos"] = np.ones((n, 2, 3), dtype=np.float32)
    else:
        z["y"], z["x"] = lat, lon
        for i in range(19):
            z[f"mask_{i}"] = np.ones((2, 3), dtype=bool)
        time = np.arange(n) * 5 + 2.5
        attrs = dict(units="days since 1975-01-01", calendar="standard")
        for name in [
            f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)
        ] + ["zos"]:
            z[name] = np.ones((n, 2, 3), dtype=np.float32)
    z["time"] = time
    z["time"].attrs.update(attrs)
    for name in ("tauuo", "tauvo", "hfds"):
        z[name] = np.ones((n, 2, 3), dtype=np.float32)
    zarr.consolidate_metadata(str(root))
    for filename in ("OM4_means.zarr", "OM4_stds.zarr"):
        group = zarr.open_group(str(tmp_path / filename), mode="w")
        for name in ("tauuo", "tauvo", "hfds"):
            group[name] = np.array(1.0, dtype=np.float32)
        zarr.consolidate_metadata(str(tmp_path / filename))
    reference = dict(
        lat=lat,
        lon=lon,
        mean=np.zeros(77),
        std=np.full(77, 2.0),
        names=np.array(
            [f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)]
            + ["zos"]
        ),
    )
    return root, reference


@pytest.mark.parametrize("kind", ["om4", "cm4"])
def test_model_windows_keep_native_calendar_scales_and_replay(tmp_path, kind):
    path, reference = model_store(tmp_path, kind)
    data = ModelMonths(path, kind, reference, tmp_path)
    fresh = data.sample(0, "cpu")
    replay = data.continuation(0, 1, "cpu")
    assert replay["start_lead"] == fresh["next_start_lead"]
    np.testing.assert_allclose(fresh["fields"], 0.5)
    np.testing.assert_allclose(fresh["forcing"], 0)
    assert fresh["fields"].shape[1] == 19 + len(fresh["weights"])
    if kind == "cm4":
        assert isinstance(data.window_months[0], cftime.DatetimeNoLeap)
    with pytest.raises(ValueError, match="0..11"):
        data.continuation(0, 12, "cpu")


def test_trajectory_fresh_parity_and_replay_anchoring(tmp_path, monkeypatch):
    from samudra.observations.archive import ObservationArchive

    np.savez(
        tmp_path / "grid.npz",
        mask=np.ones((77, 2, 3)),
        lat=[-30, 30],
        lon=[0, 120, 240],
    )
    for month in ("2000-01", "2000-02"):
        folder = tmp_path / "iap/2000"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{month}.npz",
            values=np.ones((2, 14, 2, 3)),
            ohc=np.ones((2, 2, 3)),
        )
    archive = ObservationArchive(tmp_path)

    def interval(product, start, end):
        channels = {"sst": 1, "adt": 3, "era5": 8}[product]
        return np.full(
            (channels, 2, 3),
            (start - pd.Timestamp("1999-01-01")).days,
            dtype=np.float32,
        )

    monkeypatch.setattr(archive, "interval", interval)
    old = archive.sample("2000-01")
    new = trajectory(archive, "2000-01", target_month="2000-01")
    for key in ("surface", "atmosphere", "interior", "month_weights", "midpoints"):
        np.testing.assert_array_equal(new[key], old[key])
    replay = trajectory(archive, "2000-01", target_month="2000-02")
    assert replay["start_lead"] == new["next_start_lead"] == 6
    assert replay["interval_starts"][0] == np.datetime64("2000-01-31")
    # February rebasing would start Feb1 and silently change every five-day label.
    assert (
        archive.sample("2000-02")["surface"][19, 0, 0, 0]
        != replay["surface"][19, 0, 0, 0]
    )


def test_model_replay_rejects_a_gap_even_when_target_month_itself_is_valid(tmp_path):
    path, reference = model_store(tmp_path, "om4")
    store = zarr.open_consolidated(str(path), mode="r+")
    # A five-day missing interval; maintain valid lengths and strictly ordered time.
    store["time"][100:] += 5
    zarr.consolidate_metadata(str(path))
    data = ModelMonths(path, "om4", reference, tmp_path)
    origin = next(
        i
        for i, stamp in enumerate(data.window_months)
        if stamp.month == 12 and stamp.year == 1975
    )
    with pytest.raises(ValueError, match="gap"):
        data.continuation(origin, 10, "cpu")


def test_leap_year_replay_does_not_extend_to_370_days():
    with pytest.raises(ValueError, match="one year"):
        trajectory(None, "2000-01-01", target_month="2000-12-01")


@pytest.mark.parametrize("kind", ["om4", "cm4"])
def test_model_replay_metadata_can_be_logged_without_json_coercion(tmp_path, kind):
    path, reference = model_store(tmp_path, kind)
    data = ModelMonths(path, kind, reference, tmp_path)
    layout = data.continuation_layout(0, 1)
    assert layout is not None
    assert all(type(value) is int for value in layout.values())
    assert json.loads(json.dumps(layout)) == layout
    for sample in (data.sample(0, "cpu"), data.continuation(0, 1, "cpu")):
        metadata = {
            key: sample[key]
            for key in (
                "start_lead",
                "end_lead",
                "next_start_lead",
                "origin_index",
                "duration",
            )
        }
        assert all(type(value) is int for value in metadata.values())
        # This arithmetic feeds production training logs and qualification reports.
        metadata["max_start_lead_days"] = max(0, sample["start_lead"] * 5)
        assert json.loads(json.dumps(metadata)) == metadata
