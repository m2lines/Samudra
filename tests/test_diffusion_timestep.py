# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Endpoint alignment, checkpoint compatibility, and gradient paths."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from samudra.experiments.diffusion_latent import PersistentLatentProcessor
from samudra.experiments.diffusion_timestep import (
    TimestepLatentProcessor,
    auxiliary_loss,
    endpoint_consistency,
    install_timestep_processor,
)


def _example():
    torch.manual_seed(19)
    base = PersistentLatentProcessor(8, 1, (4, 6), (8, 12))
    state = torch.randn(1, 2, 8, 4, 6)
    forcing = torch.randn(1, 2, 3, 8, 12)
    contexts = torch.randn(1, 2, 5, 8, 12)
    return base, state, forcing, contexts


def test_five_day_behavior_and_ordinary_checkpoint_keys_unchanged():
    base, state, forcing, contexts = _example()
    base.input_adapters = nn.ModuleDict({"om4": nn.Conv2d(24, 24, 1)})
    upgraded = TimestepLatentProcessor(base)
    expected = base(state, forcing[:, 0], contexts[:, 0])
    actual = upgraded(state, forcing[:, 0], contexts[:, 0])
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    ordinary = {
        key: value
        for key, value in upgraded.state_dict().items()
        if not key.startswith("timestep_adapter.")
    }
    base.load_state_dict(ordinary, strict=True)
    # Learned auxiliary adapter must never perturb inherited five-day behavior.
    nn.init.normal_(upgraded.timestep_adapter.weight)
    torch.testing.assert_close(
        expected, upgraded(state, forcing[:, 0], contexts[:, 0]), rtol=0, atol=0
    )
    holder = SimpleNamespace(processor=upgraded)
    assert install_timestep_processor(holder) is upgraded


def test_ten_day_uses_ordered_forcing_and_retains_origin_slot():
    base, state, forcing, contexts = _example()
    processor = TimestepLatentProcessor(base)
    direct = processor.advance_ten_days(state, forcing, contexts)
    torch.testing.assert_close(direct[:, 0], state[:, -1])
    five = processor(state, forcing[:, 0], contexts[:, 0])
    # Zero adapter starts with a doubled shared residual, not a second head.
    torch.testing.assert_close(
        direct[:, -1], state[:, -1] + 2 * (five[:, -1] - state[:, -1])
    )
    nn.init.normal_(processor.timestep_adapter.weight, std=0.1)
    forcing = forcing.requires_grad_()
    output = processor.advance_ten_days(state, forcing, contexts)
    output[:, -1].square().mean().backward()
    assert forcing.grad is not None
    assert forcing.grad[:, 0].abs().sum() > 0
    assert forcing.grad[:, 1].abs().sum() > 0
    swapped = processor.advance_ten_days(state, forcing.flip(1), contexts.flip(1))
    assert not torch.equal(output[:, -1], swapped[:, -1])
    with pytest.raises(ValueError, match="two ordered forcing"):
        processor.advance_ten_days(state, forcing[:, :1], contexts)


def test_consistency_only_compares_endpoint_and_both_branches_receive_gradients():
    short = torch.randn(2, 2, 8, 4, 6, requires_grad=True)
    direct = (short.detach() + 0.3).requires_grad_()
    origin = torch.randn_like(short)
    value = endpoint_consistency(short, direct, origin)
    changed_history = direct.detach().clone()
    changed_history[:, 0] += 1000
    torch.testing.assert_close(
        value, endpoint_consistency(short, changed_history, origin)
    )
    value.backward()
    for branch in (short, direct):
        assert branch.grad is not None
        assert branch.grad[:, 0].count_nonzero() == 0
        assert branch.grad[:, -1].abs().sum() > 0
        assert torch.isfinite(branch.grad).all()
    zero = torch.zeros_like(short)
    assert torch.isfinite(endpoint_consistency(zero, direct, zero))


def test_auxiliary_supervises_matching_physical_pair_without_interior_inputs(
    monkeypatch,
):
    import samudra.experiments.diffusion_timestep as module

    base, state, forcing, contexts = _example()
    origin = state.clone().requires_grad_()
    targets = torch.stack(
        (torch.full((1, 4, 8, 12), 5.0), torch.full((1, 4, 8, 12), 10.0)), 1
    )
    surface = torch.zeros(1, 19, 2, 8, 12)
    wave = SimpleNamespace(
        trainset=None,
        mask=torch.ones(4, 8, 12),
        weights=torch.ones(4, 8, 12),
        model_sample=lambda *_: (surface, None, contexts[:, 0], None, forcing, targets),
    )
    model = SimpleNamespace(
        processor=TimestepLatentProcessor(base),
        surface=[0, 1],
        decoder=None,
        training=True,
        encode_native=lambda *args, **kwargs: (origin, None, None),
    )
    supplied = {}

    def fake_loss(decoder, latent, target, *args, **kwargs):
        supplied["latent"] = latent
        supplied["target"] = target
        return latent.square().mean()

    monkeypatch.setattr(module, "denoising_loss", fake_loss)
    loss, metrics = auxiliary_loss(model, wave, 0, torch.ones_like(surface), 8)
    planned_contexts = torch.stack(
        [module.advance_season(contexts[:, 0], day) for day in (5, 10)], 1
    )
    short = model.processor.rollout(origin, forcing, planned_contexts)
    direct = model.processor.advance_ten_days(origin, forcing, planned_contexts)
    torch.testing.assert_close(
        supplied["latent"],
        torch.stack((short[1][:, -1], direct[:, -1]), 1).flatten(1, 2),
    )
    torch.testing.assert_close(supplied["target"], targets.flatten(1, 2))
    assert all(not value.requires_grad for value in metrics.values())
    loss.backward()
    assert origin.grad is not None and origin.grad.abs().sum() > 0
    assert model.processor.timestep_adapter.weight.grad is not None
    assert model.processor.timestep_adapter.weight.grad.abs().sum() > 0
    assert model.processor.head.weight.grad is not None
    assert model.processor.head.weight.grad.abs().sum() > 0
