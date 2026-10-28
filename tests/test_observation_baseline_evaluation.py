# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
import xarray as xr
from numpy.typing import NDArray

from samudra.aggregator.validate.rollout import RolloutValidationAggregator
from samudra.observations.evaluate import (
    AnnualObservationEvaluator,
    ObservedChannelRMSE,
    complete_month_weights,
    evaluate_annual_predictions,
)
from samudra.utils.output import ModelInferenceOutput


def test_fully_observed_score_matches_existing_samudra_rollout_metric():
    std = torch.tensor([2.0, 100.0])
    area = torch.tensor([[1.0, 3.0]])
    normalized = torch.arange(1.0, 25.0).reshape(3, 4, 1, 2)
    preprocessor = SimpleNamespace(
        prognostic_mask=torch.ones(2, 1, 2, dtype=torch.bool),
        unnormalize_tensor_prognostic=lambda x, **kwargs: (
            x * std.reshape(1, 1, 2, 1, 1)
        ),
    )
    samudra = RolloutValidationAggregator(
        output_steps=2,
        area_weights=area,
        preprocessor=cast(Any, preprocessor),
        data_layout=cast(Any, None),
        prognostic_var_names=cast(Any, ("zos", "hfds")),
        distributed_reduce=False,
    )
    samudra.record_batch(
        ModelInferenceOutput(normalized, torch.zeros_like(normalized), xr.DataArray())
    )
    observed = ObservedChannelRMSE(("zos", "hfds"), std, area)
    physical = normalized.reshape(6, 2, 1, 2) * std.reshape(1, 2, 1, 1)
    observed.record(physical[:2], torch.zeros_like(physical[:2]))
    observed.record(physical[2:], torch.zeros_like(physical[2:]))
    expected = samudra.get_logs("annual")
    actual = observed.logs("annual")
    for key in (
        "normalized_rmse/channel_mean",
        "weighted_rmse/zos",
        "weighted_rmse/hfds",
    ):
        assert actual[f"annual/{key}"] == pytest.approx(expected[f"annual/{key}"])


def test_sparse_targets_average_times_within_channels_before_channel_mean():
    metric = ObservedChannelRMSE(
        ("sst", "temperature"), torch.ones(2), torch.tensor([[1.0, 3.0]])
    )
    predictions = torch.tensor(
        [[[[1.0, 3.0]], [[10.0, 1000.0]]], [[[4.0, 4.0]], [[999.0, 999.0]]]]
    )
    targets = torch.zeros_like(predictions)
    targets[0, 1, 0, 1] = float("nan")
    targets[1, 1] = float("nan")
    metric.record(predictions, targets)
    logs = metric.logs("annual")
    assert logs["annual/weighted_rmse/sst"] == pytest.approx((np.sqrt(7.0) + 4.0) / 2)
    assert logs["annual/weighted_rmse/temperature"] == 10
    assert logs["annual/normalized_rmse/channel_mean"] == pytest.approx(
        ((np.sqrt(7.0) + 4.0) / 2 + 10) / 2
    )
    assert logs["annual/target_times/temperature"] == 1
    assert logs["annual/target_cells/temperature"] == 1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_missing_prediction_at_observed_cell_fails_loudly(invalid):
    metric = ObservedChannelRMSE(("sst",), torch.ones(1), torch.ones(1, 2))
    prediction = torch.tensor([[[invalid, invalid]]])
    # Explicitly unsupported cells are not observations, even with a finite fill.
    metric.record(
        prediction,
        torch.zeros_like(prediction),
        torch.zeros_like(prediction, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="Nonfinite prediction"):
        metric.record(prediction, torch.zeros_like(prediction))
    with pytest.raises(ValueError, match="No observation support"):
        metric.logs("annual")


def test_calendar_month_weights_cross_boundary_and_drop_partial_months():
    starts: NDArray[np.datetime64] = np.arange(
        np.datetime64("2013-10-29"), np.datetime64("2013-12-08"), np.timedelta64(5, "D")
    )
    months, weights = complete_month_weights(starts, starts + np.timedelta64(5, "D"))
    assert months == ["2013-11"]
    assert weights.sum() == pytest.approx(1)
    # Oct29-Nov3 contributes only Nov1-Nov3; Nov28-Dec3 contributes Nov28-Dec1.
    assert weights[0, 0] == pytest.approx(2 / 30)
    assert weights[0, 6] == pytest.approx(3 / 30)
    assert weights[0, 7] == 0


def annual_fixture(origin="2013-11-01"):
    starts = np.datetime64(origin, "ns") + np.arange(73) * np.timedelta64(5, "D")
    ends = starts + np.timedelta64(5, "D")
    months, matrix = complete_month_weights(starts, ends)
    return dict(
        interval_starts=starts,
        interval_ends=ends,
        months=months,
        month_matrix=torch.tensor(matrix),
        raw_surface=torch.zeros(1, 73, 2, 1, 1),
        monthly_targets=torch.zeros(1, len(months), 28, 1, 1),
    )


def test_annual_score_uses_monthly_mean_not_average_of_step_rmse():
    trajectory = annual_fixture()
    prediction = torch.zeros(1, 73, 77, 1, 1)
    # November has six equal five-day bins: signed profile errors cancel.
    prediction[:, :3, 39] = 6
    prediction[:, 3:6, 39] = -6
    prediction[:, :, 38] = 2  # SST gets exactly one vote, monthly SST not a second.
    prediction[:, :, 76] = 3
    # Unsupervised U/V and deep T/S may be NaN and are not scored.
    prediction[:, :, :38] = float("nan")
    prediction[:, :, 52:57] = float("nan")
    logs = evaluate_annual_predictions(
        prediction, trajectory, torch.ones(77), torch.ones(1, 1)
    )
    assert logs["rollout_val/365d/normalized_rmse/channel_mean"] == pytest.approx(
        5 / 29
    )
    assert logs["rollout_val/365d/weighted_rmse/thetao_1"] == pytest.approx(0)
    assert logs["rollout_val/365d/monthly/weighted_rmse/thetao_0"] == pytest.approx(2)
    assert logs["rollout_val/365d/complete_months"] == 12
    assert logs["rollout_val/365d/target_times/thetao_0"] == 73
    assert logs["rollout_val/365d/monthly/target_times/thetao_1"] == 12


@pytest.mark.parametrize("origin", ["2013-07-01", "2014-02-01"])
def test_validation_rejects_training_or_heldout_target_overlap(origin):
    with pytest.raises(ValueError, match="after training and strictly before 2015"):
        AnnualObservationEvaluator(
            annual_fixture(origin), torch.ones(77), torch.ones(1, 1)
        )


def test_annual_evaluator_requires_complete_contiguous_coverage():
    trajectory = annual_fixture()
    evaluator = AnnualObservationEvaluator(trajectory, torch.ones(77), torch.ones(1, 1))
    evaluator.record(torch.zeros(1, 77, 1, 1))
    with pytest.raises(ValueError, match="incomplete annual"):
        evaluator.result()
    trajectory["interval_starts"][1] += np.timedelta64(1, "D")
    with pytest.raises(ValueError, match="positive and contiguous"):
        AnnualObservationEvaluator(trajectory, torch.ones(77), torch.ones(1, 1))


def test_annual_evaluator_rejects_incorrect_month_operator():
    trajectory = annual_fixture()
    trajectory["month_matrix"][0, 0] = 0
    with pytest.raises(ValueError, match="do not match calendar"):
        AnnualObservationEvaluator(trajectory, torch.ones(77), torch.ones(1, 1))
