# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from samudra.utils.rollout_validation import RolloutValidationSpec


def test_rollout_validation_spec_resolves_day_horizon_to_model_steps():
    target_times = np.array(
        [
            np.datetime64("2026-01-02"),
            np.datetime64("2026-01-03"),
            np.datetime64("2026-01-04"),
            np.datetime64("2026-01-05"),
        ]
    )

    spec = RolloutValidationSpec.from_day_horizon(
        days=3,
        start_time=np.datetime64("2026-01-01"),
        target_times=target_times,
        output_steps=2,
    )

    assert spec == RolloutValidationSpec(
        label="3d",
        model_steps=1,
        target_timesteps=2,
    )


def test_rollout_validation_spec_rejects_uncovered_day_horizon():
    with pytest.raises(ValueError, match="val_time only covers"):
        RolloutValidationSpec.from_day_horizon(
            days=10,
            start_time=np.datetime64("2026-01-01"),
            target_times=np.array([np.datetime64("2026-01-02")]),
            output_steps=1,
        )
