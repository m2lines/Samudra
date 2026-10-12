# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from samudra.experiments.diffusion_interventions import observation_update_is_fast


def test_hybrid_uses_three_fast_then_full_for_entire_observation_updates():
    modes = [observation_update_is_fast("hybrid", count) for count in range(64)]
    assert modes == [True, True, True, False] * 16
    assert sum(modes) == 48
    assert [observation_update_is_fast("hybrid", count) for count in range(4)] == [
        True,
        True,
        True,
        False,
    ]
    assert all(observation_update_is_fast("fast", count) for count in range(64))
    assert not any(observation_update_is_fast("control", count) for count in range(64))


@pytest.mark.parametrize(("arm", "count"), [("other", 0), ("fast", -1)])
def test_invalid_observation_schedule_fails(arm, count):
    with pytest.raises(ValueError):
        observation_update_is_fast(arm, count)
