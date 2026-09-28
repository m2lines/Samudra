# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from samudra.experiments import initializer_models
from samudra.experiments.missingness import (
    TaskView,
    completion_loss,
    corrupt_sample,
    identity_adapters,
    structured_visibility,
)
from samudra.experiments.task_schedule import TaskSchedule, sample_indices


def initializer(monkeypatch, policy):
    monkeypatch.setattr(
        initializer_models,
        "make_unet",
        lambda inputs, outputs, widths: nn.Conv2d(inputs, outputs, 1),
    )
    return initializer_models.HistoryInitializer(
        ["thetao_0", "hidden", "zos"], "unet", True, surface_policy=policy
    )


def test_missing_surface_is_learned_but_known_observations_are_exact(monkeypatch):
    model = initializer(monkeypatch, "observed-only")
    nn.init.zeros_(model.net.weight)
    nn.init.constant_(model.net.bias, 3)
    surface = torch.full((1, 19, 2, 4, 6), 7.0)
    validity = torch.ones_like(surface, dtype=torch.bool)
    validity[:, :, :, 1, 2] = False
    surface[:, :, :, 1, 2] = 0
    mask = torch.ones(3, 4, 6)
    mask[:, 0, 0] = 0
    result = model(
        surface, torch.zeros(1, 19, 3, 4, 6), torch.zeros(1, 5, 4, 6), mask, validity
    )
    assert (result[:, :, [0, 2], 1, 2] == 3).all()
    assert (result[:, :, [0, 2], 2, 2] == 7).all()
    assert (result[:, :, :, 0, 0] == 0).all()
    result[:, :, [0, 2], 1, 2].sum().backward()
    assert (model.net.bias.grad[[0, 2, 3, 5]] != 0).all()
    model.zero_grad()
    result = model(
        surface, torch.zeros(1, 19, 3, 4, 6), torch.zeros(1, 5, 4, 6), mask, validity
    )
    result[:, :, [0, 2], 2, 2].sum().backward()
    assert model.net.bias.grad.count_nonzero() == 0


def test_legacy_copy_and_checkpoint_keys_preserved(monkeypatch):
    old = initializer(monkeypatch, "legacy-copy")
    new = initializer(monkeypatch, "observed-only")
    assert old.state_dict().keys() == new.state_dict().keys()
    surface = torch.rand(1, 19, 2, 4, 6)
    result = old(
        surface,
        torch.zeros(1, 19, 3, 4, 6),
        torch.zeros(1, 5, 4, 6),
        torch.ones(3, 4, 6),
        torch.zeros_like(surface),
    )
    torch.testing.assert_close(result[:, :, [0, 2]], surface[:, -2:], atol=0, rtol=0)


def test_structured_masks_are_reproducible_persistent_and_do_not_mutate_labels():
    surface = torch.randn(1, 25, 2, 30, 60)
    valid = torch.ones_like(surface, dtype=torch.bool)
    valid[..., 0, :] = False
    sample = {
        "surface": surface,
        "validity": valid,
        "raw_surface": surface[:, 19:].clone(),
    }
    before = surface.clone()
    rng = torch.get_rng_state()
    changed = corrupt_sample(sample, 71)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    torch.testing.assert_close(sample["surface"], before)
    assert changed["raw_surface"] is sample["raw_surface"]
    assert not changed["validity"][..., 0, :].any()
    assert not changed["completion_valid"][..., 0, :].any()
    assert changed["completion_valid"].any()
    assert (changed["surface"][~changed["validity"]] == 0).all()
    torch.testing.assert_close(changed["validity"][:, 0], changed["validity"][:, -1])
    torch.testing.assert_close(changed["validity"], structured_visibility(valid, 71))
    assert not torch.equal(changed["validity"], structured_visibility(valid, 72))
    # Future labels cannot affect the corruption of forecast history inputs.
    alternative = dict(sample, raw_surface=torch.full_like(sample["raw_surface"], 1000))
    torch.testing.assert_close(
        corrupt_sample(alternative, 71)["surface"], changed["surface"]
    )


def test_completion_loss_ignores_unobserved_targets_and_keeps_polar_labels():
    prediction = torch.zeros(1, 2, 2, 3, 4, requires_grad=True)
    target = torch.ones_like(prediction)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., 1, :] = False
    target[..., 1, :] = float("nan")
    loss = completion_loss(prediction, target, valid, torch.tensor([-80.0, 0.0, 80.0]))
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad[..., 0, :] != 0).all()
    assert (prediction.grad[..., 1, :] == 0).all()


def test_task_adapters_start_identity_preserve_rng_and_route_gradients():
    rng = torch.get_rng_state()
    adapters = identity_adapters(3)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    value = torch.randn(2, 3, 4, 5)
    for task in ("om4", "observation"):
        torch.testing.assert_close(adapters[task](value), value, atol=0, rtol=0)
    adapters["observation"](value).square().mean().backward()
    assert all(p.grad is None for p in adapters["om4"].parameters())
    assert all(p.grad is not None for p in adapters["observation"].parameters())


def test_task_binding_is_explicit():
    class Recording(nn.Module):
        surface = [0, 2]

        def forward(self, value, task="observation"):
            return task, value

    binding = TaskView(Recording(), "om4")
    assert binding.surface == [0, 2]
    assert binding(3) == ("om4", 3)


def test_finish_schedule_has_exact_exposure_and_unbroken_observation_tail():
    schedule = TaskSchedule(8000, 8000, "mixed-finish", 2000)
    assert schedule.counts(14000) == {"om4": 8000, "observation": 6000}
    assert schedule.counts(16000) == {"om4": 8000, "observation": 8000}
    assert all(schedule.task(i) == "observation" for i in range(14000, 16000))
    tasks = [schedule.task(i) for i in range(schedule.total)]
    assert tasks.count("om4") == tasks.count("observation") == 8000
    for completed in (0, 123, 13999, 14000, 15789):
        restored = TaskSchedule(8000, 8000, "mixed-finish", 2000)
        assert restored.counts(completed) == schedule.counts(completed)
        count = schedule.counts(completed)["observation"]
        assert sample_indices(243, 1729, count, 8) == sample_indices(
            243, 1729, restored.counts(completed)["observation"], 8
        )


@pytest.mark.parametrize(
    "ordering,tail",
    [("mixed", 1), ("scratch", 2), ("mixed-finish", 0), ("mixed-finish", 8)],
)
def test_invalid_finish_contract_rejected(ordering, tail):
    with pytest.raises(ValueError):
        TaskSchedule(8 if ordering != "scratch" else 0, 8, ordering, tail)


def test_gap_intervention_changes_only_supported_missing_cells():
    from samudra.experiments.observation_gaps import replace_missing

    value = torch.full((1, 2, 2, 3, 4), 7.0)
    valid = torch.ones_like(value, dtype=torch.bool)
    valid[..., 1, :] = False
    replacement = torch.full_like(value, 2.0)
    replacement[..., 1, 0] = float("nan")
    wet = torch.ones(2, 3, 4, dtype=torch.bool)
    wet[:, 1, 1] = False
    result, support = replace_missing(value, valid, replacement, wet)
    assert torch.equal(result[valid], value[valid])
    assert (result[support] == 2).all()
    assert (result[~support] == 7).all()


def test_initial_gap_intervention_preserves_hidden_state_history_intervention_does_not():
    from samudra.experiments.observation_gaps import forecast

    class Frozen:
        def initialize(self, surface, atmosphere, context, mask, validity):
            state = surface.new_ones((1, 2, 77, 3, 4)) * surface.mean()
            state[:, :, [38, 76]] = surface[:, -2:]
            return state

        def adapt(self, forcing):
            return forcing

        def evolution(self, states, forcing, context, mask, lead):
            return states[:, -1]

    surface = torch.full((1, 19, 2, 3, 4), 7.0)
    valid = torch.ones_like(surface, dtype=torch.bool)
    valid[..., 1, :] = False
    arguments = (
        surface,
        torch.zeros(1, 20, 3, 3, 4),
        torch.zeros(1, 20, 5, 3, 4),
        torch.ones(77, 3, 4),
        valid,
    )
    _, original = forecast(Frozen(), arguments, torch.zeros_like(surface), "baseline")
    _, initial = forecast(
        Frozen(), arguments, torch.zeros_like(surface), "initial-zero"
    )
    _, history = forecast(
        Frozen(), arguments, torch.zeros_like(surface), "history-zero"
    )
    hidden = [i for i in range(77) if i not in (38, 76)]
    torch.testing.assert_close(
        initial[:, :, hidden], original[:, :, hidden], rtol=0, atol=0
    )
    assert not torch.equal(history[:, :, hidden], original[:, :, hidden])
    assert (initial[:, :, [38, 76], 1, :] == 0).all()
    assert (history[:, :, [38, 76], 1, :] == 0).all()
