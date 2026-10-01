# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from samudra.experiments.observation_annual import inputs, month_weights
from samudra.experiments.observation_model import ObservationTransfer


def test_annual_month_weights_cover_calendar_without_phase_reset():
    starts = pd.date_range("2013-11-01", periods=73, freq="5D")
    for month in pd.period_range("2013-11", "2014-10", freq="M"):
        weights = month_weights(starts, str(month))
        assert weights.sum() == pytest.approx(1)
        assert np.count_nonzero(weights) in (6, 7)
    january = month_weights(starts, "2014-01")
    assert january[12] == pytest.approx(4 / 31)
    assert january[18] == pytest.approx(2 / 31)
    with pytest.raises(ValueError, match="Incomplete calendar"):
        month_weights(starts[:-1], "2014-10")


def test_future_observations_never_enter_annual_inputs():
    h, w = 4, 8
    mask = np.ones((77, h, w), dtype=bool)
    data = SimpleNamespace(
        grid=dict(
            mask=mask,
            mean=np.zeros(77),
            std=np.ones(77),
            lat=np.linspace(-80, 80, h),
            lon=np.arange(w) * 45,
        ),
        stats=dict(
            surface_climatology=np.zeros((12, 2, h, w)),
            atmosphere_mean=np.zeros(8),
            atmosphere_std=np.ones(8),
        ),
        mask=torch.tensor(mask),
        tensor=lambda a: torch.as_tensor(a, dtype=torch.float32),
    )
    raw = dict(
        surface=np.ones((92, 2, h, w)),
        atmosphere=np.zeros((92, 8, h, w)),
        midpoint=pd.date_range("2013-07-29", periods=92, freq="5D").to_numpy(),
    )
    before = inputs(data, raw)
    raw["surface"][19:] = np.nan
    after = inputs(data, raw)
    assert before[0].shape[1] == 19
    assert before[1].shape[1] == 92
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a, b)


def test_year_rollout_initializes_once_and_carries_state():
    count = []

    def initialize(surface, atmosphere, context, mask, validity):
        count.append(surface.shape[1])
        return torch.zeros(1, 2, 1, 2, 2)

    def evolution(states, forcing, context, mask, steps):
        assert steps == 1
        return states[:, -1] + 1

    model: Any = SimpleNamespace(
        initialize=initialize,
        adapt=lambda a: a[:, :, :3],
        evolution=evolution,
        call=lambda fn, *args: fn(*args),
    )
    forecast, _ = ObservationTransfer.forecast(
        model,
        torch.zeros(1, 19, 2, 2, 2),
        torch.zeros(1, 92, 8, 2, 2),
        torch.zeros(1, 92, 5, 2, 2),
        torch.ones(1, 2, 2),
        torch.ones(1, 19, 2, 2, 2),
    )
    assert count == [19]
    torch.testing.assert_close(
        forecast[0, :, 0, 0, 0], torch.arange(1, 74, dtype=torch.float32)
    )


def test_annual_velocity_uses_named_dimensions(monkeypatch):
    from samudra.experiments import observation_annual as annual
    from samudra.metrics import kernels

    lat, lon = np.linspace(-55, 55, 12), np.arange(36) * 10
    t = np.arange(73)[:, None, None]
    height = (
        np.sin(np.deg2rad(lon)[None, None] + t / 8)
        * np.cos(np.deg2rad(lat))[None, :, None]
    )
    surface = np.stack([height + 15, height], 1)
    u, v = kernels.geostrophic_velocity_from_zos(
        annual._field(height, lat, lon), "lat", "lon"
    )
    reference = np.concatenate(
        [
            surface,
            np.stack(
                [
                    u.transpose("time", "lat", "lon").values,
                    v.transpose("time", "lat", "lon").values,
                ],
                1,
            ),
        ],
        1,
    )
    data = SimpleNamespace(
        grid=dict(lat=lat, lon=lon, mask=np.ones((77, 12, 36), dtype=bool))
    )
    monkeypatch.setattr(annual, "spatial_error", lambda *args: None)
    result = annual.surface_metrics(surface, reference, data)
    assert result["annual_eke_rmse"] == pytest.approx(0)
    assert result["leads"]["365"]["velocity_rmse"] == pytest.approx(0)
