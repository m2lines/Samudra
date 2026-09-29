# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from samudra.experiments.diffusion_extend import extend
from samudra.experiments.diffusion_fit import bounded_fit
from samudra.experiments.observation_pilot import atomic_json, digest


def test_completed_budget_extension_matches_uninterrupted_optimizer_and_rng(tmp_path):
    def run(path, updates, seconds):
        torch.manual_seed(71)
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        signature = dict(max_updates=updates, max_seconds=seconds, seed=71)
        atomic_json(signature, path / "protocol.json")
        state = bounded_fit(
            model,
            optimizer,
            lambda step: model(torch.randn(3, 2)).square().mean(),
            lambda step: model(torch.ones(3, 2)).square().mean().item(),
            path,
            signature,
            max_updates=updates,
            max_seconds=seconds,
            checkpoint_every=1,
            validate_every=2,
        )
        return model, optimizer, state

    parent = tmp_path / "parent"
    _, _, state = run(parent, 2, 10)
    atomic_json(
        dict(state=state, best_checkpoint_sha256=digest(parent / "best.pt")),
        parent / "OBSERVATION_COMPLETE.json",
    )
    parent_hash = digest(parent / "last.pt")
    receipt = extend(parent, tmp_path / "continued", updates=4, hours=20 / 3600)
    assert receipt["initial_state"]["step"] == 2
    assert not receipt["initial_state"]["complete"]
    continued, optimizer, state = run(tmp_path / "continued", 4, 20)
    expected, expected_optimizer, _ = run(tmp_path / "whole", 4, 20)
    assert state["step"] == 4 and state["complete"]
    assert digest(parent / "last.pt") == parent_hash
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(continued.state_dict()[name], tensor, rtol=0, atol=0)
    for key, record in expected_optimizer.state_dict()["state"].items():
        for name, tensor in record.items():
            torch.testing.assert_close(
                optimizer.state_dict()["state"][key][name], tensor, rtol=0, atol=0
            )
