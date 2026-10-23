# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from samudra.experiments.shared_era5 import SharedEra5
from scripts.run_monthly_replay import phase_settings


def test_shared_schedules_keep_main_uninterrupted_and_fork_obs_only():
    assert Counter(phase_settings(i, "shared-8k", "A")[1] for i in range(8000)) == dict(
        obs=4000, om4=4000
    )
    assert phase_settings(4000, "shared-8k", "A") == (False, "obs", 1e-4)
    for mode, start in [("shared-8k", 8000), ("shared-cool4k", 4000)]:
        assert all(
            phase_settings(i, mode, "A")[1] == "obs" for i in range(start, start + 1000)
        )
        assert phase_settings(start + 999, mode, "A")[2] == pytest.approx(1e-6)
    with pytest.raises(ValueError):
        phase_settings(3999, "shared-cool4k", "A")


def test_forcing_cache_uses_history_and_same_observation_scales():
    cache = object.__new__(SharedEra5)
    cache.positions = {i: i for i in range(23)}
    cache.values = np.broadcast_to(np.arange(23)[:, None, None, None], (23, 8, 2, 3))
    cache.mean = np.ones((1, 8, 1, 1)) * 2
    cache.std = np.ones((1, 8, 1, 1)) * 3
    reader = SimpleNamespace(windows=[(19, 23, None, None)])
    data = SimpleNamespace(tensor=lambda x: x)
    actual = cache.load(reader, 0, data)
    assert actual.shape == (1, 23, 8, 2, 3)
    np.testing.assert_allclose(actual[0, :, 0, 0, 0], (np.arange(23) - 2) / 3)
