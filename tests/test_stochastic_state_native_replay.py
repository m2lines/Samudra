# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Native replay boundary, split, age, and serialization invariants."""

import io

import numpy as np
import pytest
import torch

from samudra.experiments.stochastic_state_native_replay import OM4ReplayPool


def trajectory():
    return [
        torch.full((2, 1, 2, 3, 1, 1), float(i), requires_grad=True) for i in range(7)
    ]


def test_native_capture_uses_last_state_and_keeps_member_history():
    pool = OM4ReplayPool(100)
    states = trajectory()
    states[6].data[1] += 100
    pool.add_next(0, 0, states, 10)
    entry = pool.draw(np.random.default_rng(1), 10)
    assert entry["start_lead"] == 6
    assert not entry["latent"].requires_grad
    torch.testing.assert_close(entry["latent"], states[6])
    states[6].data.zero_()
    assert entry["latent"][0].mean() == 6 and entry["latent"][1].mean() == 106


def test_native_chain_reaches_day360_but_never_crosses_split_or_resets_age():
    pool = OM4ReplayPool(100)
    pool.add_next(28, 0, trajectory(), 0)
    assert not pool.entries
    for lead in range(0, 72, 6):
        pool.add_next(0, lead, trajectory(), lead, origin_step=0)
    assert sorted(pool.entries) == list(range(6, 72, 6))
    assert (max(pool.entries) + 6) * 5 == 360
    assert pool.entries[66][0]["origin_step"] == 0
    assert pool.draw(np.random.default_rng(1), 250) is not None
    assert pool.draw(np.random.default_rng(1), 251) is None


def test_native_serialization_preserves_sampling_and_configuration():
    pool = OM4ReplayPool(100)
    for step in range(20):
        pool.add_next(0, 0, trajectory(), step)
    assert len(pool.entries[6]) == 8
    buffer = io.BytesIO()
    torch.save(pool.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    restored = OM4ReplayPool(100)
    restored.load_state_dict(saved)
    a, b = np.random.default_rng(11), np.random.default_rng(11)
    for _ in range(10):
        assert pool.draw(a, 20)["created_step"] == restored.draw(b, 20)["created_step"]
    with pytest.raises(ValueError, match="configuration"):
        OM4ReplayPool(101).load_state_dict(saved)
    with pytest.raises(ValueError, match="boundaries"):
        pool.add_next(0, 1, trajectory(), 0)


def test_native_loss_continues_cache_and_confines_fresh_fields_to_auxiliary(
    monkeypatch,
):
    from types import SimpleNamespace

    import samudra.experiments.stochastic_state_native_replay as module

    monkeypatch.setattr(module, "advance_season", lambda context, days: context)
    monkeypatch.setattr(module, "structured_visibility", lambda mask, seed: mask)
    monkeypatch.setattr(
        module, "excess_latent_loss", lambda states, initial: states[-1].sum() * 0
    )
    monkeypatch.setattr(
        module,
        "spatial_crps",
        lambda prediction, target, support, area, scale: (
            (prediction - target).square().mean().reshape(1),
            torch.tensor([True]),
        ),
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class Model:
        arm = "initial"
        surface = [0, 2]

        def advance_members(self, state, forcing, context, *, generator, task):
            assert task == "om4"
            return state + parameter, None

        def encode_native(self, surface, past, context, mask, visible, task):
            return torch.full((1, 2, 3, 1, 1), -100.0), None, None

        def sample_initial_members(self, initial, generator, members):
            return initial[None].expand(members, *initial.shape)

        def decode_members(self, state, mask, known=None, anchor=None):
            return state

    def sample(dataset, indices):
        assert indices == [8]
        return (
            torch.ones(1, 19, 2, 1, 1),
            None,
            None,
            torch.zeros(1, 2, 3, 1, 1),
            torch.zeros(1, 6, 1, 1, 1),
            torch.zeros(1, 6, 3, 1, 1),
        )

    wave = SimpleNamespace(
        trainset=range(100),
        device="cpu",
        model_sample=sample,
        mask=torch.ones(3, 1, 1),
        weights=torch.ones(3, 1, 1),
    )
    entry = {"origin_index": 2, "start_lead": 6, "latent": trajectory()[6]}
    captured = []
    loss = module.native_replay_objective(
        Model(),
        wave,
        entry,
        torch.ones(1, 19, 2, 1, 1),
        3,
        capture=lambda states: captured.extend(states),
    )
    assert captured[0].mean() == 6 and captured[-1].mean() == 12
    loss.backward()
    assert parameter.grad is not None and parameter.grad > 0
    assert entry["latent"].grad is None
