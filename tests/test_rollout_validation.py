import dataclasses
import math

import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.aggregator.validate.rollout import (
    RolloutValidationAggregator,
    plan_rollout_windows,
)
from ocean_emulators.stepper import get_rollout_step_chunks
from ocean_emulators.train import (
    INITIAL_BEST_VAL_LOSS,
    AutoregressiveValSpec,
    Trainer,
)
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


def test_plan_rollout_windows_overlaps_runs_that_do_not_fit(caplog):
    plan = _plan(num_steps=480, num_runs=2)

    assert plan.num_steps == 480
    # Both runs still happen: the first starts at the beginning of val_time and
    # the second ends at its end, which is the least overlap available.
    assert [window.start_index for window in plan.windows] == [0, 744 - 481]
    assert plan.windows[-1].start_index + plan.windows[-1].num_timesteps == 744
    assert "only 1 fit without overlapping" in caplog.text


def test_plan_rollout_windows_spreads_more_overlapping_runs_evenly():
    plan = _plan(num_steps=480, num_runs=3)

    last_start = 744 - 481
    assert [window.start_index for window in plan.windows] == [
        0,
        round(last_start / 2),
        last_start,
    ]


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


def test_rollout_aggregator_records_grouped_one_step_chunks():
    aggregator = RolloutValidationAggregator(
        num_steps=2,
        area_weights=torch.ones(2, 2),
        wet=torch.ones(1, 2, 2, dtype=torch.bool),
        loss_fn=lambda gen, target, sample_weight: (gen - target).square().mean(
            dim=(0, 2, 3)
        ),
        device=torch.device("cpu"),
        ownership=torch.ones(2, 1, 2, 2),
    )
    target = torch.zeros(1, 2, 1, 2, 2)

    for step, error in enumerate((1.0, 2.0)):
        prediction = torch.full((1, 2, 1, 2, 2), error)
        aggregator.record_run(_output(prediction, target), step_offset=step)
    aggregator.finish_run()

    torch.testing.assert_close(aggregator.loss_by_step(), torch.tensor([1.0, 4.0]))
    torch.testing.assert_close(
        aggregator.rmse_by_step(), torch.tensor([1.0, 2.0])
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


class _FakeTrainer:
    """Just enough Trainer for the pure combined-loss / spec-gating logic."""

    def __init__(self, long_start_epoch: int = 20):
        self.autoregressive_val_specs = (
            AutoregressiveValSpec(
                label="short",
                num_steps=72,
                num_runs=3,
                seed_offset=0,
                weight=0.35,
            ),
            AutoregressiveValSpec(
                label="long",
                num_steps=480,
                num_runs=1,
                seed_offset=1,
                weight=0.40,
                start_epoch=long_start_epoch,
            ),
        )
        self._combined_loss_contributors = frozenset()
        self.best_val_loss = INITIAL_BEST_VAL_LOSS

    combined_validation_loss = Trainer.combined_validation_loss
    _note_combined_loss_contributors = Trainer._note_combined_loss_contributors
    due_autoregressive_val_specs = Trainer.due_autoregressive_val_specs


def _ar_stats(short: float | None = None, long: float | None = None) -> dict:
    stats = {}
    if short is not None:
        stats["val/mean/short-autoregressive-loss"] = short
    if long is not None:
        stats["val/mean/long-autoregressive-loss"] = long
    return stats


def test_combined_loss_weights_all_three():
    combined, logs = _FakeTrainer().combined_validation_loss(
        1.0, _ar_stats(short=2.0, long=3.0)
    )

    expected = 0.25 * 1.0 + 0.35 * 2.0 + 0.40 * 3.0
    assert combined == pytest.approx(expected)
    assert logs["val/mean/combined-loss"] == pytest.approx(expected)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_combined_loss_drops_non_finite_long(bad, caplog):
    trainer = _FakeTrainer()
    combined, _ = trainer.combined_validation_loss(1.0, _ar_stats(short=2.0, long=bad))

    # Long drops out; the remaining weights are renormalized, not left short.
    expected = (0.25 * 1.0 + 0.35 * 2.0) / (0.25 + 0.35)
    assert combined == pytest.approx(expected)
    assert math.isfinite(combined)
    assert (
        "long-autoregressive val loss is nan, combined-loss is computed on "
        "one-step and short-autoregressive with weights 0.25 and 0.35"
    ) in caplog.text


def test_combined_loss_drops_two_non_finite_losses(caplog):
    trainer = _FakeTrainer()
    nan = float("nan")
    combined, _ = trainer.combined_validation_loss(1.0, _ar_stats(short=nan, long=nan))

    # Only one-step survives, so it is the score outright.
    assert combined == pytest.approx(1.0)
    assert "with weights 0.25" in caplog.text


def test_combined_loss_is_nan_only_when_nothing_is_finite(caplog):
    nan = float("nan")
    combined, logs = _FakeTrainer().combined_validation_loss(
        nan, _ar_stats(short=nan, long=nan)
    )

    assert math.isnan(combined)
    assert math.isnan(logs["val/mean/combined-loss"])
    assert "no best-validation checkpoint will be saved" in caplog.text
    # A nan score must never look like an improvement.
    assert not (combined <= INITIAL_BEST_VAL_LOSS)


def test_combined_loss_resets_best_when_a_validation_joins():
    trainer = _FakeTrainer()

    # Epochs before the long start epoch: only one-step and short contribute.
    trainer.combined_validation_loss(1.0, _ar_stats(short=2.0))
    trainer.best_val_loss = 1.35
    trainer.combined_validation_loss(1.0, _ar_stats(short=2.0))
    assert trainer.best_val_loss == 1.35

    # Long joins and shifts the scale, so the stale best must not stand.
    trainer.combined_validation_loss(1.0, _ar_stats(short=2.0, long=900.0))
    assert trainer.best_val_loss == INITIAL_BEST_VAL_LOSS


def test_combined_loss_does_not_reset_best_when_a_loss_goes_nan():
    trainer = _FakeTrainer()
    trainer.combined_validation_loss(1.0, _ar_stats(short=2.0, long=900.0))
    trainer.best_val_loss = 361.0

    # A transient nan drops long for one epoch; that is not a scale change.
    trainer.combined_validation_loss(1.0, _ar_stats(short=2.0, long=float("nan")))
    assert trainer.best_val_loss == 361.0

    # Nor is long coming back, since it already contributed once.
    trainer.combined_validation_loss(1.0, _ar_stats(short=2.0, long=800.0))
    assert trainer.best_val_loss == 361.0


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(1, ["short"]), (19, ["short"]), (20, ["short", "long"]), (21, ["short", "long"])],
)
def test_due_specs_gate_long_on_its_start_epoch(epoch, expected):
    trainer = _FakeTrainer(long_start_epoch=20)

    labels = [spec.label for spec in trainer.due_autoregressive_val_specs(epoch)]
    assert labels == expected


def test_due_specs_start_epoch_one_runs_long_immediately():
    trainer = _FakeTrainer(long_start_epoch=1)

    assert [s.label for s in trainer.due_autoregressive_val_specs(1)] == [
        "short",
        "long",
    ]


def test_due_specs_still_respects_disabled_rollouts():
    trainer = _FakeTrainer(long_start_epoch=1)
    trainer.autoregressive_val_specs = tuple(
        dataclasses.replace(spec, num_runs=0) for spec in trainer.autoregressive_val_specs
    )

    assert trainer.due_autoregressive_val_specs(50) == []
