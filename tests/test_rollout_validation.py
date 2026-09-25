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


@pytest.mark.parametrize("chunk_sizes", [[3], [1, 2]])
def test_rollout_score_averages_normalized_rmse_over_times_and_channels(chunk_sizes):
    from types import SimpleNamespace
    from typing import Any, cast

    import torch
    import xarray as xr

    from samudra.aggregator.validate.rollout import RolloutValidationAggregator
    from samudra.utils.output import ModelInferenceOutput

    # Different raw scales must not change equal weighting in normalized space.
    preprocessor = SimpleNamespace(
        prognostic_mask=torch.tensor([[[True, False]], [[True, False]]]),
        unnormalize_tensor_prognostic=lambda x, **kwargs: x
        * torch.tensor([2.0, 100.0]).reshape(1, 1, 2, 1, 1),
    )
    aggregator = RolloutValidationAggregator(
        output_steps=2,
        area_weights=torch.ones(1, 2),
        preprocessor=cast(Any, preprocessor),
        data_layout=cast(Any, None),  # Surface variables need no depth metadata.
        prognostic_var_names=cast(Any, ("zos", "hfds")),
        distributed_reduce=False,
    )
    prediction = (
        torch.arange(1.0, 13.0).reshape(3, 4, 1, 1).expand(-1, -1, -1, 2).clone()
    )
    prediction[..., 1] = float("nan")  # Land does not contribute.
    start = 0
    for size in chunk_sizes:
        chunk = prediction[start : start + size]
        aggregator.record_batch(
            ModelInferenceOutput(chunk, torch.zeros_like(chunk), xr.DataArray())
        )
        start += size
    logs = aggregator.get_logs("rollout_val/360d")
    assert logs["rollout_val/360d/normalized_rmse/channel_mean"] == pytest.approx(6.5)
    assert (
        logs["rollout_val/360d/weighted_rmse/zos"]
        != logs["rollout_val/360d/weighted_rmse/hfds"]
    )

    prediction[0, 0, 0, 0] = float("nan")  # Invalid ocean values invalidate the score.
    aggregator.record_batch(
        ModelInferenceOutput(
            prediction[:1], torch.zeros_like(prediction[:1]), xr.DataArray()
        )
    )
    assert not np.isfinite(
        aggregator.get_logs("rollout_val")["rollout_val/normalized_rmse/channel_mean"]
    )
