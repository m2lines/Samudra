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


@pytest.mark.parametrize(
    "phase,dense", [("joint", False), ("joint", True), ("reconstruction", True)]
)
def test_fixed_budget_ignores_plateau_and_preserves_prefix(
    tmp_path, monkeypatch, phase, dense
):
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
        milestone_steps=[1, 3, 4] if dense else [2, 4],
        dense_checkpoints=dense,
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
    p.phase(phase, 4, 1)
    assert json.loads((tmp_path / f"{phase}-complete.json").read_text())["step"] == 4
    first = 1 if dense else 2
    prefix = torch.load(tmp_path / f"{phase}-{first:05d}.pt", weights_only=False)
    final = torch.load(tmp_path / f"{phase}-00004.pt", weights_only=False)
    assert prefix["step"] == first and final["step"] == 4
    assert any(
        not torch.equal(prefix["model"][k], final["model"][k]) for k in prefix["model"]
    )
    assert (tmp_path / f"{phase}-{first:05d}-best.json").exists()
    if dense:
        assert (tmp_path / f"{phase}-00000.pt").exists()
        assert (tmp_path / f"{phase}-00003.pt").exists()
    frozen = (tmp_path / f"{phase}-{first:05d}.pt").read_bytes()
    p.phase(phase, 4, 1)
    assert frozen == (tmp_path / f"{phase}-{first:05d}.pt").read_bytes()


def test_submitted_dependencies_bound_concurrency_and_budget(tmp_path, monkeypatch):
    import importlib.util
    import itertools
    import sys
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts/submit_observation_budget_wave.py"
    spec = importlib.util.spec_from_file_location("budget_submission_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requests = {}

    def capture(args, name, target, arguments, hours, dependencies):
        requests[name] = {"dependencies": set(dependencies), "hours": hours}
        return name

    monkeypatch.setattr(module.submission, "submit", capture)
    monkeypatch.setattr(
        sys, "argv", ["submit", "--root", str(tmp_path), "--code-commit", "producer"]
    )
    module.main()
    assert sum(x["hours"] for x in requests.values()) == 81
    assert requests["random"]["dependencies"] == {"calibration"}
    maximum = 0
    for count in range(len(requests) + 1):
        for subset in itertools.combinations(requests, count):
            complete = set(subset)
            if any(not requests[name]["dependencies"] <= complete for name in complete):
                continue
            runnable = [
                name
                for name, request in requests.items()
                if name not in complete and request["dependencies"] <= complete
            ]
            maximum = max(maximum, len(runnable))
    assert maximum == 4
