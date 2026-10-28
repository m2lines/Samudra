# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Replay comparisons must preserve support and expose numerical differences."""

import numpy as np
import pytest
from export_annual_profiles import differences


def test_replay_difference_on_common_wet_support():
    a = np.array([np.nan, 3, 5], dtype=np.float32)
    b = np.array([np.nan, 2, 3], dtype=np.float64)
    assert differences(a, b) == dict(rms=np.sqrt(2.5), max_abs=2.0)


@pytest.mark.parametrize("other", [np.array([1, 2, 3]), np.array([np.nan, 2])])
def test_replay_difference_rejects_changed_support_or_shape(other):
    with pytest.raises(ValueError, match="support differs"):
        differences(np.array([np.nan, 2, 3]), other)
