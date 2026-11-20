# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the six-hour task, independent of monthly OBS archives."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from samudra.config import BlockConfig, UNetBackboneConfig
from samudra.models.surface_initialized import SurfaceInitializedConfig
from samudra.observations.sample_types import ForecastSample
from samudra.observations.snapshot import (
    LLC_STATE_NAMES,
    SnapshotTaskConfig,
    eligible_origins,
    masked_mse,
    snapshot_objective,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example():
    torch.manual_seed(21)
    cfg = SurfaceInitializedConfig(
        backbone=UNetBackboneConfig(
            ch_width=[4],
            dilation=[1],
            n_layers=[1],
            core_block=BlockConfig(norm="instance", instance_norm_affine=True),
        ),
        history_steps=29,
        forcing_mode="shared_era5",
        tasks=("llc", "observation"),
        activation_checkpointing=False,
    )
    model = cfg.build(LLC_STATE_NAMES)
    mask = torch.ones(205, 4, 8, dtype=torch.bool)
    mask[:, 0, 0] = False
    fields = torch.randn(1, 41, 205, 4, 8)
    surface = fields[:, :, [102, 204]].clone()
    validity = mask[[102, 204]].expand_as(surface).clone()
    sample: ForecastSample = dict(
        mask=mask,
        surface=surface,
        validity=validity,
        atmosphere=torch.randn(1, 41, 8, 4, 8),
        contexts=torch.randn(1, 41, 5, 4, 8),
        initial_fields=fields[:, 27:29],
        target_fields=fields[:, 29:],
    )
    return model, sample, mask.float()


def test_week_includes_both_endpoints_and_targets_are_daily():
    cfg = SnapshotTaskConfig()
    assert cfg.offsets().shape == (41,)
    assert cfg.offsets()[0] == np.timedelta64(-168, "h")
    assert cfg.offsets()[28] == np.timedelta64(0, "h")
    assert cfg.offsets()[-1] == np.timedelta64(72, "h")
    assert cfg.target_steps("obs").tolist() == [4, 8, 12]
    assert cfg.target_steps("llc").tolist() == list(range(1, 13))


def test_windows_reject_gaps_excluded_history_and_split_crossings():
    start, end = np.datetime64("2012-01-01"), np.datetime64("2012-02-01")
    times = np.arange(start, end, np.timedelta64(6, "h"))
    task = SnapshotTaskConfig()
    indices = eligible_origins(times, start, end, task, observation=True)
    assert times[indices[0]] == np.datetime64("2012-01-08")
    assert times[indices[-1]] == np.datetime64("2012-01-28")
    excluded = [(np.datetime64("2012-01-15"), np.datetime64("2012-01-16"))]
    for index in eligible_origins(
        times, start, end, task, observation=True, excluded=excluded
    ):
        window = times[index] + task.offsets()
        assert window[-1] < excluded[0][0] or window[0] >= excluded[0][1]
    gap_times = np.delete(times, 35)
    for index in eligible_origins(gap_times, start, end, task, observation=False):
        assert np.array_equal(
            gap_times[index - 28 : index + 13], gap_times[index] + task.offsets()
        )
    with pytest.raises(ValueError, match="unique"):
        eligible_origins(times[[0, 0]], start, end, task, observation=True)


def test_area_loss_ignores_missing_slots_without_diluting_channels():
    prediction = torch.zeros(1, 3, 2, 1, 2, requires_grad=True)
    target = torch.full_like(prediction, float("nan"))
    target[:, 0, 0] = torch.tensor([1.0, 3.0])
    target[:, 1:, 1] = 2
    valid = target.isfinite()
    area = torch.tensor([1.0, 3.0]).expand(2, 1, 2)
    loss = masked_mse(prediction, target, valid, area)
    # First channel is (1 + 3*9)/4 = 7; second is 4 regardless of two slots.
    torch.testing.assert_close(loss, torch.tensor(5.5))
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad[~valid] == 0).all()
    valid[:, :, 1] = False
    with pytest.raises(ValueError, match="Every scored channel"):
        masked_mse(prediction, target, valid, area)


def test_observation_loss_ignores_subdaily_targets_but_backpropagates_all_steps():
    model, sample, area = example()
    sample["validity"][:, :29] = False
    sample["validity"][:, :29:4] = sample["mask"][[102, 204]]
    outputs = []

    def retain(module, args, output):
        output.retain_grad()
        outputs.append(output)

    hook = model.evolution.register_forward_hook(retain)
    loss, parts, final = snapshot_objective(model, sample, area, "obs", 101)
    loss.backward()
    hook.remove()
    assert set(parts) == {"forecast", "completion"}
    assert final.shape == (1, 2, 205, 4, 8) and not final.requires_grad
    assert len(outputs) == 12
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in outputs)
    poisoned: ForecastSample = {**sample, "surface": sample["surface"].clone()}
    for index in range(29, 41):
        if (index - 28) % 4:
            poisoned["surface"][:, index] = float("nan")
    changed, _, _ = snapshot_objective(model, poisoned, area, "obs", 101)
    torch.testing.assert_close(changed, loss)


def test_dense_llc_supervises_every_depth_and_no_future_surface_leaks():
    model, sample, area = example()
    loss, parts, _ = snapshot_objective(model, sample, area, "llc", 101)
    loss.backward()
    head = model.evolution.net[-1]
    assert (head.weight.grad.flatten(1).abs().sum(1) > 0).all()
    assert set(parts) == {"forecast", "completion", "reconstruction"}
    sample["surface"][:, 29:] = float("nan")
    changed, _, _ = snapshot_objective(model, sample, area, "llc", 101)
    torch.testing.assert_close(changed, loss)


def test_29_frame_model_copies_only_observed_final_surface_slots():
    model, sample, _ = example()
    sample["validity"][:, :29] = False
    sample["validity"][:, :29:4] = sample["mask"][[102, 204]]
    forecast, states = model.forecast(
        sample["surface"],
        sample["atmosphere"],
        sample["contexts"],
        sample["mask"],
        sample["validity"],
        "observation",
    )
    assert forecast.shape == (1, 12, 205, 4, 8)
    available = sample["validity"][:, 27:29]
    torch.testing.assert_close(
        states[:, :, [102, 204]][available], sample["surface"][:, 27:29][available]
    )
    assert not available[:, 0].any() and available[:, 1].any()
    with pytest.raises(ValueError, match="Task must"):
        model.forcing(sample["atmosphere"], "om4-family")


def test_experimental_model_and_task_presets():
    root = Path(__file__).resolve().parents[1] / "experiments/llc-observation-six-hour"
    config = SurfaceInitializedConfig.model_validate(
        yaml.safe_load((root / "model.yaml").read_text())
    )
    task = SnapshotTaskConfig.model_validate(
        yaml.safe_load((root / "task.yaml").read_text())
    )
    assert config.history_steps == task.history_steps == 29
    assert config.initializer_backbone is not None
    assert config.evolution_backbone is not None
    assert config.initializer_backbone.ch_width == [280, 380, 480, 520]
    assert config.evolution_backbone.ch_width == [280, 380, 480, 520]
    assert config.tasks == ("llc", "observation")


def test_monthly_trainer_rejects_six_hour_model_before_opening_archives():
    from samudra.observations.config import ObservationTrainConfig

    path = (
        Path(__file__).resolve().parents[1]
        / "src/samudra/configs/observations/deterministic_mixed_large.yaml"
    )
    values = yaml.safe_load(path.read_text())
    values["model"]["history_steps"] = 29
    with pytest.raises(ValueError, match="monthly observation trainer"):
        ObservationTrainConfig.model_validate(values)


@pytest.mark.parametrize(
    "bad_bound", ["start", "end", "exclusion_start", "exclusion_end"]
)
def test_nonfinite_split_or_exclusion_bounds_fail_closed(bad_bound):
    start, end = np.datetime64("2012-01-01"), np.datetime64("2012-02-01")
    times = np.arange(start, end, np.timedelta64(6, "h"))
    bounds = dict(
        start=start,
        end=end,
        exclusion_start=np.datetime64("2012-01-15"),
        exclusion_end=np.datetime64("2012-01-16"),
    )
    bounds[bad_bound] = np.datetime64("NaT")
    with pytest.raises(ValueError, match="bounds must be finite"):
        eligible_origins(
            times,
            bounds["start"],
            bounds["end"],
            SnapshotTaskConfig(),
            observation=True,
            excluded=[(bounds["exclusion_start"], bounds["exclusion_end"])],
        )
