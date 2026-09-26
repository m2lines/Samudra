# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from samudra.experiments.observation_annual_decomposition import (
    temporal_decomposition,
    weighted_summary,
)


def test_temporal_identity_with_missing_reference_and_latitude_weights():
    r = np.arange(24, dtype=float).reshape(3, 2, 4)
    r[1, 0, 0] = np.nan
    p = np.nan_to_num(r) + np.arange(3)[:, None, None] + 2
    d = temporal_decomposition(p, r, np.array([[1], [0.5]]))
    assert d["mse"] == pytest.approx(
        d["within_year_temporal_mean_bias_mse"]
        + d["within_year_temporal_anomaly_error_mse"]
    )
    p[0, 1, 1] = np.nan
    with pytest.raises(ValueError):
        temporal_decomposition(p, r, np.ones((2, 4)))


def test_spatial_bias_and_anomaly_correlation_are_distinct():
    r = np.arange(8, dtype=float).reshape(2, 4)
    d = weighted_summary(r + 3, r, np.zeros_like(r), np.ones_like(r))
    assert d["rmse"] == pytest.approx(3)
    assert d["bias_removed_rmse"] == pytest.approx(0)
    assert d["centered_anomaly_correlation"] == pytest.approx(1)
