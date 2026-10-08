# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from samudra.eval import Eval


@pytest.mark.parametrize("with_observations", [True, False])
def test_run_returns_and_logs_configured_metrics(with_observations):
    evaluator = Mock(spec=Eval)
    evaluator.standalone_inference = Mock(return_value={"inference/error": 1.0})
    evaluator.observations = object() if with_observations else None
    evaluator.report_observation_metrics = Mock(return_value={"obs/sst_rmse": 2.0})
    evaluator.wandb_logger = Mock()
    evaluator.finish = Mock()

    metrics = Eval.run(evaluator)

    assert metrics["inference/error"] == 1.0
    assert isinstance(metrics["eval_total_seconds"], float)
    assert metrics["eval_total_seconds"] >= 0
    assert ("obs/sst_rmse" in metrics) == with_observations
    if with_observations:
        evaluator.report_observation_metrics.assert_called_once_with(
            evaluator.observations
        )
    else:
        evaluator.report_observation_metrics.assert_not_called()
    # The rollout is logged before observation scoring mutates the returned map.
    evaluator.wandb_logger.log.assert_called_once()
    evaluator.finish.assert_called_once()


@pytest.mark.parametrize(
    "phase", ["standalone_inference", "report_observation_metrics"]
)
def test_run_propagates_failures_and_finishes(phase):
    evaluator = Mock(spec=Eval)
    evaluator.standalone_inference = Mock(return_value={"inference/error": 1.0})
    evaluator.observations = object()
    evaluator.report_observation_metrics = Mock(return_value={})
    evaluator.wandb_logger = Mock()
    evaluator.finish = Mock()
    getattr(evaluator, phase).side_effect = RuntimeError("scoring failed")

    with pytest.raises(RuntimeError, match="scoring failed"):
        Eval.run(evaluator)

    evaluator.finish.assert_called_once()
    assert evaluator.wandb_logger.log.call_count == (
        phase == "report_observation_metrics"
    )
