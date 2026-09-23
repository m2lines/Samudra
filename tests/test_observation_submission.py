# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "observation_submission",
    Path(__file__).parents[1] / "scripts/submit_observation_pilot.py",
)
assert spec is not None and spec.loader is not None
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)


def test_live_dependency_is_retained(monkeypatch):
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, "123|RUNNING\n", ""),
    )
    pending, checks = submission.pending_dependencies(["123"])
    assert pending == ["123"]
    assert checks[0]["proof"] == "live scheduler dependency"


@pytest.mark.parametrize(
    "accounting,successful",
    [
        ("123|COMPLETED|0:0|\n", True),
        ("123|COMPLETED|1:0|\n", False),
        ("123|CANCELLED|0:0|\n", False),
        ("", False),
    ],
)
def test_expired_dependency_requires_successful_accounting(
    monkeypatch, accounting, successful
):
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 1, "", "Invalid job id"),
    )
    monkeypatch.setattr(
        submission.subprocess, "check_output", lambda *a, **k: accounting
    )
    if successful:
        pending, checks = submission.pending_dependencies(["123"])
        assert pending == []
        assert "COMPLETED 0:0" in checks[0]["proof"]
    else:
        with pytest.raises(ValueError, match="lacks successful completion proof"):
            submission.pending_dependencies(["123"])


@pytest.mark.parametrize("parallel,limit", [(False, 2), (True, 3)])
def test_dag_requires_completed_own_training_and_bounds_gpu_concurrency(
    tmp_path, monkeypatch, parallel, limit
):
    import itertools
    import json
    import sys

    (tmp_path / "DATA_READY.json").write_text("{}")
    contract = tmp_path / "qualification-h200"
    contract.mkdir()
    (contract / "QUALIFIED.json").write_text(
        json.dumps(
            {"strict_load": True, "zero_adapter_source_equivalence": "bitwise exact"}
        )
    )
    dependencies = {}

    def submit(args, name, module, module_args, hours, predecessors=()):
        dependencies[name] = set(predecessors)
        return name

    monkeypatch.setattr(submission, "submit", submit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "submit",
            "--root",
            str(tmp_path),
            "--data",
            str(tmp_path),
            "--code-commit",
            "pinned",
        ]
        + (["--parallel-scratch"] if parallel else []),
    )
    submission.main()

    def ancestors(name):
        result = set(dependencies[name])
        for predecessor in dependencies[name]:
            result |= ancestors(predecessor)
        return result

    prior = {name: ancestors(name) for name in dependencies}
    for evaluation, training in (
        ("primary-evaluation", "primary"),
        ("adapter-evaluation", "adapter-only"),
        ("scratch-evaluation", "scratch"),
    ):
        assert training in prior[evaluation]
    assert "scratch-fitting" in prior["scratch"]
    if parallel:
        assert "adapter-only" not in prior["primary-evaluation"]
    # Any simultaneously eligible/running jobs must be an antichain of this DAG.
    for group in itertools.combinations(dependencies, limit + 1):
        assert any(
            a in prior[b] or b in prior[a] for a, b in itertools.combinations(group, 2)
        )
