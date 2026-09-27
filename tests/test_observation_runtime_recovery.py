# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Operational recovery must not rewrite frozen checkpoint or science contracts."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location(
    "runtime_recovery",
    Path(__file__).parents[1] / "scripts/resume_observation_joint_runtime.py",
)
assert spec is not None and spec.loader is not None
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def pilot():
    args = SimpleNamespace(
        joint_hours=27,
        joint_steps=16000,
        core_lr=1e-4,
        milestone_steps=[8000, 12000, 16000],
    )
    return SimpleNamespace(
        args=args,
        manifest={"arguments": vars(args)},
        elapsed_seconds=27 * 3600,
        model=object(),
        optimizer=object(),
        completed=15000,
        deadline=object(),
    )


def test_budget_override_preserves_aliasing_manifest_and_training_state():
    run = pilot()
    manifest = copy.deepcopy(run.manifest)
    state = (run.model, run.optimizer, run.completed, run.deadline, run.elapsed_seconds)
    assert recovery.extend_runtime(run, 30) == 27
    assert run.args.joint_hours == 30
    assert run.manifest == manifest
    assert run.manifest["arguments"]["joint_hours"] == 27
    assert {**vars(run.args), "joint_hours": 27} == manifest["arguments"]
    assert state == (
        run.model,
        run.optimizer,
        run.completed,
        run.deadline,
        run.elapsed_seconds,
    )
    run.args.milestone_steps.append(17000)
    assert run.manifest == manifest


@pytest.mark.parametrize("hours", [0, 27, 30.01, float("nan"), float("inf")])
def test_invalid_extension_cannot_modify_runner(hours):
    run = pilot()
    original_args = run.args
    with pytest.raises(ValueError):
        recovery.extend_runtime(run, hours)
    assert run.args is original_args
    assert run.args.joint_hours == 27


def test_exhausted_allowance_is_not_restarted():
    run = pilot()
    run.elapsed_seconds = 30 * 3600
    with pytest.raises(ValueError, match="exhausted"):
        recovery.extend_runtime(run, 30)


def test_original_loop_completes_after_operational_extension(tmp_path):
    import datetime
    import json
    import time

    from samudra.experiments.observation_joint import JointPilot
    from samudra.experiments.task_schedule import TaskSchedule

    run: Any = JointPilot.__new__(JointPilot)
    run.args = SimpleNamespace(
        joint_hours=27, milestone_steps=[2], validate_every=1, joint_probe=False
    )
    run.manifest = {"arguments": vars(run.args)}
    frozen = copy.deepcopy(run.manifest)
    run.elapsed_seconds = 27 * 3600
    run.started = time.monotonic()
    run.completed = 1
    run.schedule = TaskSchedule(0, 2, "scratch")
    run.deadline = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)
    run.out = tmp_path
    run.resume = tmp_path / "joint-last.pt"
    run.resume.touch()
    (tmp_path / "best.pt").write_bytes(b"unchanged selected model")
    run.global_best = 0.5
    run.wandb = None
    run.qualify_selection = lambda: None
    run.validate = lambda: None
    checkpoints = []
    run.checkpoint = lambda path: checkpoints.append(
        (path.name, copy.deepcopy(run.manifest))
    )
    run.train_update = lambda: setattr(run, "completed", run.completed + 1)
    # Without the override the original loop must stop and never claim success.
    run.run_joint()
    assert (tmp_path / "TRAIN_PARTIAL.json").exists()
    assert not (tmp_path / "TRAIN_COMPLETE.json").exists()
    recovery.extend_runtime(run, 30)
    run.run_joint()
    completed = json.loads((tmp_path / "TRAIN_COMPLETE.json").read_text())
    assert completed["task_counts"] == {"om4": 0, "observation": 2}
    assert all(manifest == frozen for _, manifest in checkpoints)
    assert ("joint-00002.pt", frozen) in checkpoints
