# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
import torch

from samudra.experiments.observation_data import SPLITS, fit_statistics
from samudra.experiments.observation_prepare import DEPTHS
from samudra.experiments.observation_training import Samples


def make_samples(tmp_path, *, native=False, deep_coast=False, cache_size=None):
    lat, lon = np.array([-67.5, -22.5, 22.5, 67.5]), np.arange(22.5, 360, 45)
    mask = np.ones((77, 4, 8), dtype=bool)
    if deep_coast:
        mask[49, :2, :2] = False  # 900-1200 m, affects only the lower OHC layer.
    np.savez(
        tmp_path / "grid.npz",
        lat=lat,
        lon=lon,
        mask=mask,
        mean=np.zeros(77),
        std=np.ones(77),
    )
    native_grid = dict(
        lat=np.array([-45, 45]), lon=np.arange(45, 360, 90), depth=DEPTHS
    )
    stats: dict[str, np.ndarray] = dict(
        interior_mean=np.arange(28, dtype=np.float32),
        interior_std=np.ones(28),
        surface_mean=np.array([20.0, 0.0]),
        surface_std=np.ones(2),
        atmosphere_mean=np.zeros(8),
        atmosphere_std=np.ones(8),
        surface_climatology=np.ones((12, 2, 4, 8), dtype=np.float32),
    )
    if native:
        stats.update(
            iap_lat=native_grid["lat"],
            iap_lon=native_grid["lon"],
            interior_climatology=np.full((12, 28, 2, 4), 3.0, dtype=np.float32),
            ohc_climatology=np.full((12, 2, 2, 4), 7.0, dtype=np.float32),
        )
    np.savez(tmp_path / "statistics.npz", **stats)
    return Samples(
        tmp_path,
        "cpu",
        global_observations=True,
        iap_grid=native_grid if native else None,
        cache_size=cache_size,
    )


def test_legacy_geometry_hooks_are_exact_and_optional(tmp_path):
    data = make_samples(tmp_path)
    state = torch.arange(77 * 4 * 8, dtype=torch.float32).reshape(1, 77, 4, 8) / 100
    target = torch.zeros((1, 28, 4, 8))
    projected = data.project_interior(state)
    expected = data.physical(state[:, None])[:, 0, data.ts_indices]
    torch.testing.assert_close(projected, expected, rtol=0, atol=0)
    channel, present = data.masked_channel_mse(
        expected, target, data.ts_mask, data.ts_scale
    )
    expected_loss = (
        channel[:, :14][present[:, :14]].mean()
        + channel[:, 14:][present[:, 14:]].mean()
    ) / 2
    torch.testing.assert_close(
        data.interior_loss(state, {"interior": target}), expected_loss, rtol=0, atol=0
    )
    np.testing.assert_array_equal(data.ohc(state), data.model_ohc(state))
    assert data.interior_area is data.area
    assert data.interior_support is data.ts_mask
    assert data.ohc_metric_kwargs == {}
    assert data.cache_size == 256


def test_dual_grid_supervision_and_complete_layer_coastal_support(tmp_path):
    data = make_samples(tmp_path, native=True, deep_coast=True)
    state = torch.ones((1, 77, 4, 8), requires_grad=True)
    target = torch.zeros((1, 28, 2, 4))
    target[:, 0] = 1e6  # Supplied SST channel remains excluded.
    loss = data.interior_loss(state, {"interior": target})
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert state.grad[:, 38].count_nonzero() == 0
    assert state.grad[:, 49, :2, :2].count_nonzero() == 0
    assert state.grad[:, 57].abs().sum() > 0
    assert data.ohc_support[0, 0, 0]
    assert not data.ohc_support[1, 0, 0]
    ohc = data.ohc(state.detach())
    assert ohc.shape == (1, 2, 2, 4)
    assert np.isfinite(ohc[0, 0]).all()
    assert np.isnan(ohc[0, 1, 0, 0])
    assert np.isfinite(ohc[0, 1, 1]).all()
    assert data.cache_size == 2


def test_native_climatology_stays_native_and_obs_only_normalization(tmp_path):
    data = make_samples(tmp_path, native=True)
    sample = dict(
        name="2015-01",
        month_weights=torch.ones(6) / 6,
        raw={"midpoints": np.array(["2014-12-01"] * 19 + ["2015-01-01"] * 6)},
    )
    fields = data.climatology_fields(sample)
    assert fields["surface"].shape == (1, 6, 2, 4, 8)
    torch.testing.assert_close(fields["interior"], torch.full((1, 28, 2, 4), 3.0))
    np.testing.assert_array_equal(fields["ohc"], np.full((1, 2, 2, 4), 7.0))
    data.use_observation_normalization()
    np.testing.assert_array_equal(data.grid["mean"][[38, 76]], [20.0, 0.0])
    np.testing.assert_array_equal(data.grid["mean"][52:57], np.full(5, 13))
    with pytest.raises(NotImplementedError, match="dual-grid"):
        data.persistence_anomaly(None, sample)
    with pytest.raises(NotImplementedError, match="observation-space"):
        data.climatology_prediction(sample)


def test_observation_caches_obey_explicit_bound(tmp_path):
    data = make_samples(tmp_path, native=True, cache_size=1)
    for index in range(3):
        path = tmp_path / f"cache-{index}.npz"
        np.savez(path, value=np.array(index))
        data.read_sample(path)
    assert data.read_sample.cache_info().currsize == 1


def test_statistics_native_shape_training_only_and_legacy_60_degree_policy(tmp_path):
    # Four rows include polar values large enough to detect accidental global
    # normalizer fitting. The native IAP grid is horizontally coarser.
    grid = dict(
        lat=np.array([-75.0, -25.0, 25.0, 75.0]),
        lon=np.arange(22.5, 360, 45),
        mask=np.ones((77, 4, 8), dtype=bool),
    )
    native = dict(lat=grid["lat"], lon=np.arange(45, 360, 90), depth=DEPTHS)
    paths = tmp_path / "train"
    paths.mkdir()
    months = list(pd.period_range(*SPLITS["train"], freq="M"))
    for index, month in enumerate(months):
        surface = np.full((1, 2, 4, 8), index, dtype=np.float32)
        atmosphere = np.full((1, 8, 4, 8), index, dtype=np.float32)
        interior = np.full((2, 14, 4, 4), index, dtype=np.float32)
        surface[:, :, [0, 3]] += 10000
        atmosphere[:, :, [0, 3]] += 10000
        interior[:, :, [0, 3]] += 10000
        ohc = np.full((2, 4, 4), index * 2.0, dtype=np.float32)
        np.savez(
            paths / f"{month}.npz",
            surface=surface,
            atmosphere=atmosphere,
            interior=interior,
            ohc=ohc,
            midpoints=np.array([str(month) + "-15"]),
        )
    output = tmp_path / "statistics.npz"
    fit_statistics(paths, grid, output, iap_grid=native)
    with np.load(output) as stats:
        for name in ("surface", "atmosphere", "interior"):
            np.testing.assert_allclose(
                stats[name + "_mean"], np.mean(np.arange(243)), rtol=1e-6
            )
            np.testing.assert_allclose(
                stats[name + "_std"], np.std(np.arange(243)), rtol=1e-6
            )
        january = [i for i, month in enumerate(months) if month.month == 1]
        np.testing.assert_allclose(
            stats["interior_climatology"][0, :, 0], np.mean(january) + 10000
        )
        np.testing.assert_allclose(stats["ohc_climatology"][0], np.mean(january) * 2)
        np.testing.assert_array_equal(
            stats["interior_climatology_count"][0], len(january)
        )
        assert "60S-60N" in str(stats["normalization_policy"])
    # Never accept a held-out file in the statistics inputs.
    np.savez(paths / "2015-01.npz", sentinel=np.array(1))
    with pytest.raises(ValueError, match="Non-training month"):
        fit_statistics(paths, grid, output, iap_grid=native)
