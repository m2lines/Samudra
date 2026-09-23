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
