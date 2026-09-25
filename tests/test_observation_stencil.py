# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from samudra.experiments import observation_metrics as metrics


def test_missing_adt_stencil_cannot_change_strict_velocity_error(monkeypatch):
    monkeypatch.setattr(metrics, "spatial_error", lambda *a, **k: None)
    lat, lon = np.linspace(-55, 55, 12), np.arange(16) * 22.5
    rng = np.random.default_rng(4)
    prediction = rng.normal(size=(3, 6, 2, 12, 16)) * 0.01
    reference = rng.normal(size=(3, 6, 4, 12, 16)) * 0.01
    reference[:, :, 1, 7, 8] = np.nan
    perturbed = prediction.copy()
    perturbed[:, :, 1, 7, 8] += np.arange(3)[:, None] * 100
    kwargs = dict(
        reference=reference,
        predicted_ohc=np.zeros((3, 2, 12, 16)),
        reference_ohc=np.ones((3, 2, 12, 16)),
        lat=lat,
        lon=lon,
        mask=np.ones((12, 16), dtype=bool),
    )
    clean = metrics.score(prediction, **kwargs, strict_velocity_support=True)
    changed = metrics.score(perturbed, **kwargs, strict_velocity_support=True)
    for name in ("velocity_rmse", "eke_rmse"):
        assert clean["metrics"][name] == pytest.approx(changed["metrics"][name])
    original = metrics.score(prediction, **kwargs)
    contaminated = metrics.score(perturbed, **kwargs)
    assert (
        contaminated["metrics"]["velocity_rmse"] > original["metrics"]["velocity_rmse"]
    )
    assert metrics.protocol()["version"] == 2
    assert metrics.protocol(True)["version"] == 3
