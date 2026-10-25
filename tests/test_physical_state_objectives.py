# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from samudra.experiments import diffusion_spatial
from samudra.experiments.physical_state_objectives import (
    native_objective,
    observation_objective,
)
from samudra.experiments.samudra_physical_diffusion import SamudraPhysicalDiffusion


def test_native_teacher_forcing_uses_matching_history_forcing_and_targets():
    fields = torch.arange(5.0).reshape(1, 5, 1, 1, 1).expand(1, 5, 4, 2, 3)
    forcing = 10 + fields[:, :, :1].expand(1, 5, 8, 2, 3)
    contexts = 20 + fields[:, :, :1].expand(1, 5, 5, 2, 3)
    mask = torch.ones(4, 2, 3)
    seen: list = []

    def initial(surface, past, context, truth, observed_mask, weights, generator):
        torch.testing.assert_close(surface, fields[:, :3, [0, 3]])
        torch.testing.assert_close(past, forcing[:, :3])
        torch.testing.assert_close(context, contexts[:, 2])
        torch.testing.assert_close(truth, fields[:, 1:3])
        torch.testing.assert_close(weights, mask * 2)
        return torch.tensor(2.0)

    def transition(history, force, context, truth, observed_mask, weights, generator):
        step = 3 + len(seen)
        torch.testing.assert_close(history, fields[:, step - 2 : step])
        torch.testing.assert_close(force, forcing[:, step])
        torch.testing.assert_close(context, contexts[:, step])
        torch.testing.assert_close(truth, fields[:, step])
        seen.append(step)
        return torch.tensor(float(step))

    model = SimpleNamespace(
        history=3, surface=[0, 3], pretraining_loss=initial, transition_loss=transition
    )
    batch = dict(fields=fields, atmosphere=forcing, contexts=contexts, mask=mask)
    loss = native_objective(
        model,
        SimpleNamespace(area=torch.tensor(2.0)),
        batch,
        torch.Generator().manual_seed(1),
    )
    assert seen == [3, 4]
    assert loss.item() == 5.5


def fixtures():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    h, w = 17, 25
    model = SamudraPhysicalDiffusion(
        SimpleNamespace(channels=77, history=19, surface=[38, 76]),
        widths=(4, 8),
        sampling_steps=2,
    )
    mask = torch.ones(77, h, w)
    mask[:, 0] = 0
    indices = list(range(38, 52)) + list(range(57, 71))
    data = SimpleNamespace(
        mask=mask,
        area=torch.ones(h, w),
        mean=torch.zeros(1, 1, 77, 1, 1),
        std=torch.ones(1, 1, 77, 1, 1),
        ts_indices=indices,
        ts_mask=mask[indices],
        ts_scale=torch.ones(28, 1, 1),
        surface_scale=torch.ones(2, 1, 1),
    )
    sample = dict(
        surface=torch.randn(1, 19, 2, h, w),
        validity=mask[[38, 76]].expand(1, 19, 2, h, w),
        atmosphere=torch.randn(1, 21, 8, h, w, requires_grad=True),
        contexts=torch.randn(1, 21, 5, h, w),
        interior=torch.randn(1, 28, h, w),
        raw_surface=torch.randn(1, 2, 2, h, w),
        month_weights=torch.tensor([0.25, 0.75]),
    )
    return model, data, sample


def assert_gradients(module):
    gradients = [p.grad for p in module.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum() for g in gradients) > 0


def test_actual_observation_objective_trains_both_diffusers_and_all_forcing_channels():
    model, data, sample = fixtures()
    original = {key: value.clone() for key, value in sample.items()}
    loss = observation_objective(model, data, sample, torch.Generator().manual_seed(4))
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.initializer, model.processor):
        assert_gradients(module)
    gradients = sample["atmosphere"].grad
    assert gradients is not None and torch.isfinite(gradients).all()
    assert (gradients.abs().sum((0, 1, 3, 4)) > 0).all()
    for key in original:
        torch.testing.assert_close(sample[key], original[key])
    with pytest.raises(ValueError, match="at least two"):
        observation_objective(model, data, sample, torch.Generator(), members=1)


def test_native_all_eight_era5_channels_receive_denoising_gradients():
    model, data, sample = fixtures()
    batch = dict(
        fields=torch.randn(1, 21, 77, 17, 25) * data.mask,
        atmosphere=sample["atmosphere"],
        contexts=sample["contexts"],
        mask=data.mask,
    )
    loss = native_objective(model, data, batch, torch.Generator().manual_seed(4))
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.initializer, model.processor):
        assert_gradients(module)
    gradients = sample["atmosphere"].grad
    assert gradients is not None and torch.isfinite(gradients).all()
    assert (gradients.abs().sum((0, 1, 3, 4)) > 0).all()


def test_monthly_averaging_preserves_each_member_before_interior_scoring(monkeypatch):
    _, data, sample = fixtures()
    members = torch.zeros(2, 1, 2, 77, 17, 25)
    members[0, :, 0] = 1
    members[0, :, 1] = 5
    members[1, :, 0] = 8
    members[1, :, 1] = 0
    captured = []

    def interior(data, monthly, sample, *args):
        captured.append(monthly.clone())
        return monthly.sum() * 0

    monkeypatch.setattr(diffusion_spatial, "interior_crps", interior)
    diffusion_spatial.forecast_crps(data, members, sample)
    assert len(captured) == 1
    assert captured[0].shape == (2, 1, 77, 17, 25)
    torch.testing.assert_close(captured[0][0], torch.full_like(captured[0][0], 4))
    torch.testing.assert_close(captured[0][1], torch.full_like(captured[0][1], 2))


def test_observation_batch_preserves_equal_example_losses_with_missing_targets():
    from samudra.experiments.physical_state_objectives import (
        observation_batch_objective,
    )

    _, data, first = fixtures()
    second = {key: value.detach().clone() for key, value in first.items()}
    second["month_weights"] = torch.tensor([0.9, 0.1])
    second["atmosphere"] += 0.7
    second["surface"] *= 2
    # Unequal support must not turn the batch loss into a pooled-cell average.
    second["interior"][..., 3:, :] = torch.nan
    second["raw_surface"][..., 4:, :] = torch.nan
    second["validity"][..., 5:, :] = 0

    class FixedPrediction(torch.nn.Module):
        surface = [38, 76]

        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(0.8))

        def forecast(
            self, surface, atmosphere, contexts, mask, validity, *, generator, members
        ):
            base = atmosphere[:, 19:, :1].expand(-1, -1, 77, -1, -1) * self.gain
            prediction = torch.stack([base + 0.1 * m for m in range(members)])
            initial = base[:, :2] + surface[:, -2:, :1]
            return prediction, torch.stack([initial + 0.1 * m for m in range(members)])

    model = FixedPrediction()
    seeds = [17, 93]
    batch = observation_batch_objective(
        model, data, [first, second], torch.Generator().manual_seed(12), seeds=seeds
    )
    batch.backward()
    assert model.gain.grad is not None
    batch_gradient = model.gain.grad.clone()
    model.zero_grad(set_to_none=True)
    serial = torch.stack(
        [
            observation_batch_objective(
                model, data, [sample], torch.Generator().manual_seed(12), seeds=[seed]
            )
            for sample, seed in zip((first, second), seeds, strict=True)
        ]
    ).mean()
    serial.backward()
    torch.testing.assert_close(batch, serial, rtol=0, atol=0)
    torch.testing.assert_close(batch_gradient, model.gain.grad, rtol=1e-6, atol=1e-6)
