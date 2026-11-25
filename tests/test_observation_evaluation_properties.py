# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The streaming annual evaluator against a dense, independent scoring rule."""

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from samudra.observations.constants import (
    FORECAST_STEPS,
    PROFILE_NAMES,
    STATE_CHANNELS,
    STATE_NAMES,
    STEP_DAYS,
    SURFACE_CHANNELS,
)
from samudra.observations.evaluate import (
    complete_month_weights,
    evaluate_annual_predictions,
)
from samudra.observations.sample_types import ObservationSample

PROFILE = [STATE_NAMES.index(name) for name in PROFILE_NAMES]
METRIC = "rollout_val/365d/normalized_rmse/channel_mean"


def mean_rmse(error: np.ndarray, support: np.ndarray, area: np.ndarray) -> np.ndarray:
    """[time, channel, y, x] -> per-channel mean over supported times of spatial RMSE."""
    weight = support * area
    total = weight.sum((-2, -1))
    squared = np.where(support, error, 0) ** 2
    rmse = np.sqrt((squared * area).sum((-2, -1)) / np.where(total > 0, total, 1))
    present = total > 0
    return np.where(present, rmse, 0).sum(0) / present.sum(0)


@given(
    st.dates(
        min_value=np.datetime64("2013-08-05").item(),
        max_value=np.datetime64("2014-01-01").item(),
    ),
    st.integers(1, 3),
    st.integers(1, 3),
    st.integers(0, 2**31 - 1),
)
def test_streamed_score_matches_dense_scoring_rule(origin, height, width, seed):
    """Oracle: a dense NumPy restatement of the documented rule. Surface channels
    average five-day spatial RMSE over supported times; profile channels first
    average predictions into calendar months, then average monthly RMSE; each
    channel is divided by its state std; the score is the mean over SST, SSH and
    profile channels 1..27. Predictions are NaN wherever no target is supported,
    so a single unguarded zero weight on a NaN would poison a month."""
    rng = np.random.default_rng(seed)
    starts = np.datetime64(origin, "ns") + np.arange(FORECAST_STEPS) * np.timedelta64(
        STEP_DAYS, "D"
    )
    ends = starts + np.timedelta64(STEP_DAYS, "D")
    months, matrix = complete_month_weights(starts, ends)
    area = rng.uniform(0.1, 2, (height, width))
    std = rng.uniform(0.5, 3, STATE_CHANNELS)
    surface = rng.normal(size=(1, FORECAST_STEPS, 2, height, width))
    profile = rng.normal(size=(1, len(months), len(PROFILE), height, width))
    surface[rng.random(surface.shape) < 0.5] = np.nan
    profile[rng.random(profile.shape) < 0.5] = np.nan
    # Every scored channel needs at least one supported target.
    surface[0, 0, :, 0, 0] = 1.0
    profile[0, 0, :, 0, 0] = 1.0
    prediction = rng.normal(size=(1, FORECAST_STEPS, STATE_CHANNELS, height, width))
    # A prediction is needed where its own surface target exists, or where any
    # month it overlaps has a profile target (SST is both); all else may be NaN.
    needed = np.zeros(prediction.shape[1:], dtype=bool)
    needed[:, PROFILE] |= (
        np.einsum("mt,mcyx->tcyx", matrix > 0, np.isfinite(profile[0])) > 0
    )
    needed[:, SURFACE_CHANNELS] |= np.isfinite(surface[0])
    prediction[0] = np.where(needed, prediction[0], np.nan)

    trajectory = ObservationSample(
        interval_starts=starts,
        interval_ends=ends,
        months=months,
        month_matrix=torch.tensor(matrix),
        raw_surface=torch.tensor(surface),
        monthly_targets=torch.tensor(profile),
    )
    logs = evaluate_annual_predictions(
        torch.tensor(prediction), trajectory, torch.tensor(std), torch.tensor(area)
    )

    surface_score = (
        mean_rmse(
            prediction[0][:, SURFACE_CHANNELS] - surface[0],
            np.isfinite(surface[0]),
            area,
        )
        / std[SURFACE_CHANNELS]
    )
    monthly = np.einsum(
        "mt,tcyx->mcyx", matrix, np.nan_to_num(prediction[0][:, PROFILE])
    )
    profile_score = mean_rmse(monthly - profile[0], np.isfinite(profile[0]), area)
    profile_score = (profile_score / std[PROFILE])[1:]
    expected = np.concatenate((surface_score, profile_score)).mean()
    np.testing.assert_allclose(logs[METRIC], expected, rtol=1e-5)
