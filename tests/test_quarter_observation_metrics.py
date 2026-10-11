# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Surface and OHC metrics retain distinct, fixed observation supports."""

import numpy as np
import pytest

from samudra.experiments.observation_metrics import score


def inputs():
    lat, lon = np.linspace(-85, 85, 18), np.arange(16) * 22.5
    return dict(
        prediction=np.zeros((2, 6, 2, 18, 16)),
        reference=np.zeros((2, 6, 4, 18, 16)),
        lat=lat,
        lon=lon,
        mask=np.ones((18, 16), dtype=bool),
        global_observations=True,
        strict_velocity_support=True,
        compute_spectra=False,
    )


def test_explicit_same_grid_geometry_preserves_legacy_metrics():
    args = inputs()
    prediction = np.arange(2 * 2 * 18 * 16).reshape(2, 2, 18, 16).astype(float)
    reference = prediction * 0.9
    args.update(predicted_ohc=prediction, reference_ohc=reference)
    legacy = score(**args)
    explicit = score(
        **args,
        ohc_area=np.cos(np.deg2rad(args["lat"]))[:, None] * np.ones((1, 16)),
        ohc_support=np.ones((2, 18, 16), dtype=bool),
    )
    assert explicit == legacy


def test_native_ohc_grid_uses_layer_specific_support_and_area():
    prediction = np.array([[[[1.0, 2.0], [3.0, 100.0]], [[100.0, 2.0], [3.0, 4.0]]]])
    prediction = np.repeat(prediction, 2, axis=0)
    support = np.array([[[1, 1], [1, 0]], [[0, 1], [1, 1]]], dtype=bool)
    area = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = score(
        **inputs(),
        predicted_ohc=prediction,
        reference_ohc=np.zeros_like(prediction),
        ohc_area=area,
        ohc_support=support,
    )["metrics"]
    assert result["ohc_0_700_rmse"] == pytest.approx(np.sqrt((1 + 8 + 27) / 6))
    assert result["ohc_700_2000_rmse"] == pytest.approx(np.sqrt((8 + 27 + 64) / 9))
    assert result["sst_rmse"] == 0
    prediction[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite OHC"):
        score(
            **inputs(),
            predicted_ohc=prediction,
            reference_ohc=np.zeros_like(prediction),
            ohc_area=area,
            ohc_support=support,
        )


def test_native_ohc_geometry_cannot_be_partially_specified():
    with pytest.raises(ValueError, match="both area and fixed support"):
        score(
            **inputs(),
            predicted_ohc=np.zeros((2, 2, 2, 2)),
            reference_ohc=np.zeros((2, 2, 2, 2)),
            ohc_area=np.ones((2, 2)),
        )
