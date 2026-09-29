# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import torch

from samudra.experiments import observation_annual, observation_metrics
from samudra.experiments.observation_training import Samples


def test_global_loss_supervises_poles_but_not_missing_or_land(tmp_path):
    lat = np.array([-80.0, -30.0, 30.0, 80.0])
    np.savez(
        tmp_path / "grid.npz",
        lat=lat,
        mask=np.ones((77, 4, 3), bool),
        mean=np.zeros(77),
        std=np.ones(77),
    )
    np.savez(
        tmp_path / "statistics.npz", interior_std=np.ones(28), surface_std=np.ones(2)
    )
    for global_domain in (False, True):
        data = Samples(tmp_path, "cpu", global_observations=global_domain)
        prediction = torch.ones(1, 1, 4, 3, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[0, 0, 0, 0] = float("nan")
        mask = torch.ones(1, 4, 3, dtype=torch.bool)
        mask[0, -1, -1] = False
        loss, _ = data.masked_channel_mse(prediction, target, mask, torch.ones(1))
        loss.sum().backward()
        assert prediction.grad is not None
        assert prediction.grad[0, 0, 0, 0] == 0
        assert prediction.grad[0, 0, -1, -1] == 0
        assert bool(prediction.grad[0, 0, 0, 1] > 0) == global_domain
        assert prediction.grad[0, 0, 1, 1] > 0


def test_polar_errors_enter_global_monthly_and_annual_scores(monkeypatch):
    monkeypatch.setattr(observation_metrics, "spatial_error", lambda *a, **k: None)
    monkeypatch.setattr(observation_annual, "spatial_error", lambda *a, **k: None)
    lat, lon = np.linspace(-85, 85, 18), np.arange(16) * 22.5
    mask = np.ones((18, 16), dtype=bool)
    reference = np.zeros((2, 6, 4, 18, 16))
    prediction = np.zeros((2, 6, 2, 18, 16))
    prediction[:, :, 0, np.abs(lat) > 60, :] = 2
    ohc = np.zeros((2, 2, 18, 16))
    ohc[:, :, np.abs(lat) > 60, :] = 3
    for global_domain in (False, True):
        result = observation_metrics.score(
            prediction,
            reference,
            ohc,
            np.zeros_like(ohc),
            lat,
            lon,
            mask,
            True,
            global_domain,
        )
        assert (result["metrics"]["sst_rmse"] > 0) == global_domain
        assert (result["metrics"]["ohc_0_700_rmse"] > 0) == global_domain
        data = SimpleNamespace(
            grid=dict(lat=lat, lon=lon, mask=mask[None]),
            global_observations=global_domain,
        )
        annual_prediction = np.repeat(prediction[0, :1], 73, axis=0)
        annual_reference = np.repeat(reference[0, :1], 73, axis=0)
        annual = observation_annual.surface_metrics(
            annual_prediction, annual_reference, data
        )
        assert (annual["leads"]["365"]["sst_rmse"] > 0) == global_domain
    assert observation_metrics.protocol(True)["version"] == 3
    assert observation_metrics.protocol(True, True)["version"] == 4
