import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.aggregator.validate.rollout import (
    RolloutValidationAggregator,
    plan_rollout_windows,
)
from ocean_emulators.stepper import get_rollout_step_chunks
from ocean_emulators.utils.output import ModelInferenceOutput


def _plan(**kwargs):
    defaults = dict(label="short", total_timesteps=744, hist=0, seed=0)
    return plan_rollout_windows(**{**defaults, **kwargs})


def test_plan_rollout_windows_does_not_overlap_or_overrun():
    plan = _plan(num_steps=72, num_runs=5)

    assert plan.enabled
    assert plan.num_steps == 72
    assert len(plan.windows) == 5

    ends = []
    for window in plan.windows:
        # hist=0 means one step consumes one timestep, plus the initial condition.
        assert window.num_timesteps == 73
        ends.append(window.start_index + window.num_timesteps)

    starts = [window.start_index for window in plan.windows]
    assert starts == sorted(starts)
    for previous_end, next_start in zip(ends, starts[1:]):
        assert previous_end <= next_start
    assert ends[-1] <= 744


def test_plan_rollout_windows_accounts_for_history():
    plan = _plan(num_steps=10, num_runs=2, hist=2)

    # Each of the 10 steps consumes hist+1 timesteps, plus an input history.
    assert all(window.num_timesteps == 33 for window in plan.windows)
    assert plan.windows[-1].start_index + 33 <= 744


def test_plan_rollout_windows_reduces_runs_that_do_not_fit(caplog):
    plan = _plan(num_steps=480, num_runs=2)

    assert plan.num_steps == 480
    assert len(plan.windows) == 1
    assert "only 1 non-overlapping run(s) fit" in caplog.text


def test_plan_rollout_windows_shortens_horizon_that_does_not_fit(caplog):
    plan = _plan(total_timesteps=50, num_steps=480, num_runs=2)

    assert plan.enabled
    assert plan.num_steps == 49
    assert len(plan.windows) == 1
    assert "Shortening to 49 steps" in caplog.text


def test_plan_rollout_windows_disabled_and_impossible_cases():
    assert not _plan(num_steps=0, num_runs=5).enabled
    assert not _plan(num_steps=72, num_runs=0).enabled
    assert not _plan(total_timesteps=1, num_steps=72, num_runs=1).enabled


def test_plan_rollout_windows_is_reproducible_for_a_seed():
    first = _plan(num_steps=72, num_runs=5, seed=15)
    second = _plan(num_steps=72, num_runs=5, seed=15)
    other = _plan(num_steps=72, num_runs=5, seed=16)

    assert first.windows == second.windows
    assert first.windows != other.windows


@pytest.mark.parametrize(
    ("total_steps", "num_model_steps_forward", "expected"),
    [
        (5, 2, [2, 2, 1]),
        (4, 2, [2, 2]),
        (1, 2, [1]),
        (7, -1, [7]),
        (0, 2, []),
    ],
)
def test_get_rollout_step_chunks(total_steps, num_model_steps_forward, expected):
    assert (
        get_rollout_step_chunks(
            total_steps=total_steps,
            num_model_steps_forward=num_model_steps_forward,
        )
        == expected
    )


def _output(prediction: torch.Tensor, target: torch.Tensor) -> ModelInferenceOutput:
    time = xr.DataArray(np.arange(prediction.shape[0]), dims=["time"])
    return ModelInferenceOutput(prediction, target, time)


def _aggregator(num_steps: int) -> RolloutValidationAggregator:
    return RolloutValidationAggregator(
        num_steps=num_steps,
        area_weights=torch.ones(2, 2),
        wet=torch.ones(1, 2, 2, dtype=torch.bool),
        # A squared-error loss keeps the expected values easy to write down.
        loss_fn=lambda gen, target: (gen - target).square().mean(dim=(0, 2, 3)),
        device=torch.device("cpu"),
    )


def test_rollout_aggregator_averages_over_runs():
    aggregator = _aggregator(num_steps=2)
    target = torch.zeros(2, 1, 2, 2)

    for error in (1.0, 3.0):
        prediction = torch.full((2, 1, 2, 2), error)
        aggregator.record_run(_output(prediction, target))
        aggregator.finish_run()

    assert aggregator.n_runs == 2
    # Mean of the two runs' per-step squared errors: (1 + 9) / 2.
    torch.testing.assert_close(
        aggregator.loss_by_step(), torch.tensor([5.0, 5.0])
    )
    torch.testing.assert_close(
        aggregator.rmse_by_step(), torch.tensor([2.0, 2.0])
    )
    assert aggregator.mean_loss() == pytest.approx(5.0)


def test_rollout_aggregator_records_chunks_at_their_step_offset():
    aggregator = _aggregator(num_steps=4)
    target = torch.zeros(2, 1, 2, 2)

    aggregator.record_run(_output(torch.ones(2, 1, 2, 2), target), step_offset=0)
    aggregator.record_run(
        _output(torch.full((2, 1, 2, 2), 2.0), target), step_offset=2
    )
    aggregator.finish_run()

    assert aggregator.n_runs == 1
    torch.testing.assert_close(
        aggregator.rmse_by_step(), torch.tensor([1.0, 1.0, 2.0, 2.0])
    )


def test_rollout_aggregator_masks_land():
    wet = torch.tensor([[[True, False], [True, True]]])
    aggregator = RolloutValidationAggregator(
        num_steps=1,
        area_weights=torch.ones(2, 2),
        wet=wet,
        loss_fn=lambda gen, target: (gen - target).square().mean(dim=(0, 2, 3)),
        device=torch.device("cpu"),
    )
    # A huge error on the one land cell must not reach the reported RMSE.
    prediction = torch.tensor([[[[1.0, 1000.0], [1.0, 1.0]]]])
    aggregator.record_run(_output(prediction, torch.zeros(1, 1, 2, 2)))
    aggregator.finish_run()

    torch.testing.assert_close(aggregator.rmse_by_step(), torch.tensor([1.0]))


def test_rollout_aggregator_rejects_recording_past_the_plan():
    aggregator = _aggregator(num_steps=2)
    prediction = torch.zeros(3, 1, 2, 2)

    with pytest.raises(ValueError, match="exceeds the planned 2 rollout steps"):
        aggregator.record_run(_output(prediction, prediction))
