# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from samudra.experiments.observation_completion import scored_errors


def test_completion_score_separates_polar_support_from_missing_truth():
    lat = np.array([-75.0, 0.0, 75.0])
    target = np.full((2, 2, 3, 1), np.nan)
    prediction = np.zeros_like(target)
    valid = np.zeros_like(target, dtype=bool)
    target[:, 0, 2] = 4
    prediction[:, 0, 2] = 6
    valid[:, 0, 2] = True
    result = scored_errors(prediction, target, valid, lat)
    assert result["sst/north_polar"]["rmse"] == pytest.approx(2)
    assert result["sst/north_polar"]["bias"] == pytest.approx(2)
    assert result["sst/north_polar"]["count"] == 2
    assert result["sst/tropics_midlatitudes"]["rmse"] is None
    assert result["ssh/global"]["rmse"] is None
    assert result["ssh/global"]["count"] == 0
    valid[:, 1, 0] = True
    with pytest.raises(ValueError, match="Nonfinite target"):
        scored_errors(prediction, target, valid, lat)


def test_completion_score_weights_by_area_not_latitude_row_count():
    lat = np.array([0.0, 60.0])
    target = np.zeros((1, 2, 2, 1))
    prediction = target.copy()
    prediction[:, :, 1] = 3
    result = scored_errors(prediction, target, np.ones_like(target, bool), lat)
    assert result["sst/global"]["rmse"] == pytest.approx(np.sqrt(3))
    assert result["sst/global"]["bias"] == pytest.approx(1)
