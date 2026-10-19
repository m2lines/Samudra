# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from samudra.experiments.monthly_annual import lead_metrics, month_layout
from samudra.experiments.observation_metrics import _field
from samudra.metrics import kernels


def test_month_layout_covers_calendar_and_keeps_absolute_lead():
    starts = pd.date_range("2015-01-01", periods=73, freq="5D")
    jan, w, d, p = month_layout(starts, "2015-01")
    np.testing.assert_array_equal(jan, np.arange(7))
    np.testing.assert_allclose(w, np.array([5] * 6 + [1]) / 31)
    assert d == 31 and p == 15.5
    dec, w, d, p = month_layout(starts, "2015-12")
    assert dec[-1] == 72 and p == 349.5
    assert w.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="Incomplete calendar month"):
        month_layout(starts[:6], "2015-01")


def test_fixed_lead_metrics_zero_for_perfect_case_predictions():
    lat = np.linspace(10, 70, 12)
    lon = np.arange(0, 360, 20)
    lonplane, latplane = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    frames = np.stack(
        [np.sin(lonplane) * np.cos(latplane) * scale for scale in [0.1, 0.2, 0.4]]
    )
    pu, pv = kernels.geostrophic_velocity_from_zos(
        _field(frames, lat, lon), "lat", "lon"
    )
    u, v = [a.transpose("time", "lat", "lon").values for a in [pu, pv]]
    surface = np.stack([frames + 15, frames], axis=1)
    prediction = np.repeat(surface[:, None], 73, axis=1)
    reference = np.repeat(
        np.concatenate([surface, u[:, None], v[:, None]], 1)[:, None], 73, axis=1
    )
    data = SimpleNamespace(
        grid={"lat": lat, "lon": lon, "mask": np.ones((77, len(lat), len(lon)), bool)}
    )
    result = lead_metrics(prediction, reference, data)
    assert len(result) == 73 and result[-1]["day"] == 365
    assert all(
        r[k] == pytest.approx(0)
        for r in result
        for k in ["sst_rmse", "adt_rmse", "velocity_rmse", "eke_rmse"]
    )


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_annual_history_accepts_samples_float_validity(mask_dtype):
    from samudra.experiments.monthly_annual import history_fields

    surface = torch.randn(1, 19, 2, 3, 4)
    valid = torch.ones_like(surface, dtype=mask_dtype)
    valid[:, :, 0, 0, 0] = 0
    fields, available = history_fields(surface, valid)
    assert fields.dtype == surface.dtype and available.dtype == torch.bool
    torch.testing.assert_close(fields[:, :, [38, 76]], surface, rtol=0, atol=0)
    torch.testing.assert_close(available[:, :, [38, 76]], valid.bool(), rtol=0, atol=0)
    assert not available[:, :, :38].any()
    with pytest.raises(ValueError, match="19-bin"):
        history_fields(surface[:, :18], valid[:, :18])
