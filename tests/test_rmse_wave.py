# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Selection and exposure contracts for the longer RMSE wave."""

import copy

import numpy as np
import pytest
import torch

from samudra.experiments import observation_metrics as metrics
from samudra.experiments.observation_joint import JointPilot
from samudra.experiments.task_schedule import TaskSchedule


def test_rmse_blend_is_dimensionless_complete_and_ignores_spectra():
    names = metrics.PROTOCOL["integrated"]
    control = {"metrics": dict(zip(names, [2, 4, 8, 16, 32], strict=True))}
    result = {
        "metrics": dict(zip(names, [1, 2, 4, 8, 16], strict=True)),
        "spectra": {"bad": {"error_dex": np.nan}},
    }
    assert metrics.selection_score(result, control, [], metric="rmse-only") == 0.5
    assert metrics.selection_score(control, control, [], metric="rmse-only") == 1
    for key in names:
        invalid = copy.deepcopy(result)
        invalid["metrics"][key] = np.nan
        with pytest.raises(ValueError):
            metrics.selection_score(invalid, control, [], metric="rmse-only")
    with pytest.raises(ValueError):
        metrics.selection_score(result, control, [])
    assert metrics.protocol(True, True)["version"] == 4
    assert metrics.protocol(True, True, "rmse-only")["version"] == 5


def test_stretched_schedule_exposure_and_prefix_difference():
    old, new = TaskSchedule(2000, 2000, "mixed"), TaskSchedule(3000, 3000, "mixed")
    assert new.counts(6000) == {"om4": 3000, "observation": 3000}
    assert new.counts(4000) != old.counts(4000)
    assert [new.counts(i)["observation"] for i in [1500, 3000, 4500, 6000]] == [
        328,
        937,
        1828,
        3000,
    ]


def test_diagnostic_snapshot_keeps_lineage_and_resume_keeps_optimizer(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.get_rng_state)
    pilot = JointPilot.__new__(JointPilot)
    pilot.model = torch.nn.Linear(2, 1)
    pilot.optimizer = torch.optim.AdamW(pilot.model.parameters())
    pilot.model(torch.ones(1, 2)).sum().backward()
    pilot.optimizer.step()
    pilot.manifest = {"test": "lineage"}
    pilot.completed = 2
    pilot.schedule = TaskSchedule(2, 2, "mixed")
    pilot.elapsed_seconds = 0
    pilot.started = 0
    pilot.checkpoint(tmp_path / "resume.pt")
    pilot.checkpoint(tmp_path / "snapshot.pt", model_only=True)
    resume = torch.load(tmp_path / "resume.pt", weights_only=False)
    snapshot = torch.load(tmp_path / "snapshot.pt", weights_only=False)
    assert resume["optimizer"]["state"]
    assert snapshot["optimizer"] is None
    for key in ["manifest", "global_step", "step", "phase", "task_counts"]:
        assert snapshot[key] == resume[key]
    for key in resume["model"]:
        torch.testing.assert_close(
            snapshot["model"][key], resume["model"][key], rtol=0, atol=0
        )


def test_rmse_qualification_uses_mode_without_spectral_requirement():
    from samudra.experiments.observation_pilot import Pilot

    pilot = Pilot.__new__(Pilot)
    from types import SimpleNamespace

    pilot.args = SimpleNamespace(selection_metric="rmse-only")
    assert pilot.selection_kwargs == {"metric": "rmse-only"}


def test_wave_runner_passes_seed_budget_and_recent_only(tmp_path, monkeypatch):
    import json
    import sys

    from scripts import run_early_fine_wave as runner

    arm = "U-global-seed1730"
    paths = dict(
        producer="test",
        runs={arm: dict(seed=1730, updates=6000, recent_only=True)},
        observations="obs",
        global_om4="om4",
        early_data="early",
        selection_metric="rmse-only",
    )
    (tmp_path / "paths.json").write_text(json.dumps(paths))
    probe = tmp_path / ("probe-" + arm)
    probe.mkdir()
    (probe / "THROUGHPUT.json").write_text(
        json.dumps({"training_hours_for_4000_updates": 1})
    )
    calls = []

    def command(module, args):
        args = list(map(str, args))
        calls.append((module, args))
        out = __import__("pathlib").Path(args[args.index("--output") + 1])
        out.mkdir(parents=True, exist_ok=True)
        if module.endswith("observation_joint"):
            (out / "TRAIN_COMPLETE.json").write_text("{}")

    monkeypatch.setattr(runner, "command", command)
    monkeypatch.setenv("SAMUDRA_CODE_COMMIT", "test")
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--root", str(tmp_path), "--arm", arm, "--stage", "train"],
    )
    runner.main({arm: ("unet", 0)})
    args = calls[0][1]
    for flag, value in [
        ("--seed", "1730"),
        ("--om4-updates", "3000"),
        ("--joint-steps", "3000"),
        ("--selection-metric", "rmse-only"),
        ("--core-lr", "0.0001"),
    ]:
        assert args[args.index(flag) + 1] == value
    assert "--recent-only" in args and "--model-only-milestones" in args
    assert args[args.index("--milestone-steps") + 1 :] == [
        "500",
        "1000",
        "1500",
        "2000",
        "2500",
        "3000",
    ]
