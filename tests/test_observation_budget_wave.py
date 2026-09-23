# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from samudra.experiments import observation_budget_wave as wave
from samudra.experiments import observation_pilot as pilot


def test_child_failure_does_not_create_completion(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("failed fitting")

    monkeypatch.setattr(wave.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="failed fitting"):
        wave.child("module", [], tmp_path, "TRAIN_COMPLETE.json")
    assert not (tmp_path / "TRAIN_COMPLETE.json").exists()
    with pytest.raises(ValueError, match="Changed command"):
        wave.child("module", ["different"], tmp_path, "TRAIN_COMPLETE.json")


def test_fixed_budget_ignores_plateau_and_preserves_prefix(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.initializer = torch.nn.Linear(1, 1)
            self.evolution = torch.nn.Linear(1, 1)
            self.adapter = torch.nn.Linear(1, 1)

        def set_phase(self, phase):
            self.train()

    p: Any = pilot.Pilot.__new__(pilot.Pilot)
    p.args = SimpleNamespace(
        from_scratch=True,
        adapter_only=False,
        core_lr=0.01,
        warmup_steps=2,
        seed=1729,
        accumulate=1,
        validate_every=2,
        patience=1,
        fixed_updates=True,
        milestone_steps=[2, 4],
    )
    p.out, p.device = tmp_path, torch.device("cpu")
    p.model = Model()
    p.training = [1, 2]
    p.validation = []
    p.data = SimpleNamespace(load=lambda x: x)
    p.control, p.spectral_keys, p.global_best = {}, [], 1.0
    p.evaluate = lambda paths: {"metrics": {}}
    p.objective = lambda sample, phase: sum(
        x.square().sum() for x in p.model.parameters()
    )
    p.emit = lambda value: None
    pilot.atomic_torch({"model": p.model.state_dict()}, tmp_path / "best.pt")
    pilot.atomic_json({"score": 1.0}, tmp_path / "best.json")
    monkeypatch.setattr(pilot, "selection_score", lambda *args: 1.0)
    monkeypatch.setattr(torch, "autocast", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    p.phase("joint", 4, 1)
    assert json.loads((tmp_path / "joint-complete.json").read_text())["step"] == 4
    prefix = torch.load(tmp_path / "joint-00002.pt", weights_only=False)
    final = torch.load(tmp_path / "joint-00004.pt", weights_only=False)
    assert prefix["step"] == 2 and final["step"] == 4
    assert any(
        not torch.equal(prefix["model"][k], final["model"][k]) for k in prefix["model"]
    )
    assert (tmp_path / "joint-00002-best.json").exists()
