# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from samudra.experiments.stochastic_state_evaluate import (
    temporal_statistics,
    validate_exposure,
)


def test_temporal_statistics_distinguishes_alternating_member_from_mean():
    values = np.tile([1.0, -1.0], 4)[None, :, None, None, None]
    members = np.concatenate([values, -values])
    mask = np.ones((1, 1, 1), dtype=bool)
    result = temporal_statistics(members, mask, np.array([0]))
    np.testing.assert_allclose(result["five_day_increment_rms"], [[2], [2]])
    np.testing.assert_allclose(result["lag_one_correlation"], [[-1], [-1]])
    mean_result = temporal_statistics(
        members.mean(0, keepdims=True), mask, np.array([0])
    )
    assert mean_result["five_day_increment_rms"] == [[0.0]]
    assert mean_result["lag_one_correlation"] == [[None]]


def test_temporal_statistics_masks_land_and_missing_cells():
    values = np.arange(5, dtype=float)[None, :, None, None, None]
    values = np.broadcast_to(values, (1, 5, 1, 1, 3)).copy()
    values[..., 1] = np.nan
    values[..., 2] = np.arange(5)[None, :, None, None] * 100
    result = temporal_statistics(
        values, np.array([[[True, True, False]]]), np.array([0])
    )
    np.testing.assert_allclose(result["five_day_increment_rms"], [[1]])


def test_checkpoint_exposure_and_arm_cannot_silently_change():
    saved = dict(
        step=12, counts=dict(observation=4, om4=8), contract=dict(arm="initial")
    )
    validate_exposure(saved, "initial", 4, 8)
    with pytest.raises(ValueError, match="exposure"):
        validate_exposure(saved, "initial", 8, 4)
    with pytest.raises(ValueError, match="arm"):
        validate_exposure(saved, "ar1", 4, 8)


def test_map_exports_use_actual_day30_365_and_latest_initial_bin(tmp_path, monkeypatch):
    from samudra.experiments import stochastic_state_evaluate as module
    from samudra.experiments.observation_pilot import atomic_json, digest

    origin = "2018-01-01"
    source = tmp_path / "source"
    source.mkdir()
    records: list[dict[str, str]] = [{}] * 17
    for i in (17, 18):
        path = source / f"{i}.npz"
        np.savez(path, surface=np.full((2, 2, 3), i, dtype=float))
        records.append(dict(file=path.name, sha256=digest(path)))
    atomic_json(dict(records=records), source / "COMPLETE.json")
    fields = (
        np.broadcast_to(np.arange(73)[None, :, None, None, None], (2, 73, 4, 2, 3))
        .copy()
        .astype(float)
    )
    initial = np.broadcast_to(
        np.array([11, 22])[None, :, None, None, None], (2, 2, 4, 2, 3)
    ).copy()
    np.savez(
        tmp_path / f"members-{origin}.npz",
        fields=fields,
        initial=initial,
        mask=np.ones((4, 2, 3), bool),
        lat=np.array([-30, 30]),
        lon=np.array([0, 120, 240]),
        channel_names=np.array(["thetao_0", "zos", "thetao_9", "so_9"]),
    )
    np.savez(tmp_path / f"{origin}.npz", reference=np.full((73, 4, 2, 3), 3.0))
    monkeypatch.setattr(module, "spatial_error", lambda *args: None)
    module.export_trajectory_diagnostics(tmp_path, origin, source)
    with np.load(tmp_path / f"maps-{origin}.npz") as saved:
        assert saved["mean"].shape == (2, 2, 2, 3)
        np.testing.assert_array_equal(saved["mean"][:, 0, 0, 0], [5, 72])
        np.testing.assert_array_equal(saved["mean_error"][:, 0, 0, 0], [2, 69])
        assert saved["initial_mean"].shape == (4, 2, 3)
        assert np.all(saved["initial_mean"] == 22)
        assert np.all(saved["initial_surface_reference"] == 18)
        assert np.all(saved["initial_surface_mean_error"] == 4)
