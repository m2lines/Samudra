# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

SUBMIT = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/submit_initializer_wave.py")
)["submit"]


def setup(tmp_path, monkeypatch):
    for key in ("GHCR_USERNAME", "GHCR_TOKEN", "WANDB_API_KEY"):
        monkeypatch.setenv(key, "test-placeholder")
    proof = tmp_path / "qualification.json"
    proof.write_text(
        json.dumps(
            dict(code_commit="qualified", arm="A", max_steps=3, phase="reconstruction")
        )
    )
    args = SimpleNamespace(
        root=str(tmp_path / "run"),
        name="A",
        arm="A",
        stage="production",
        phase="reconstruction",
        deadline="2099-01-01T00:00:00Z",
        wall_hours=10,
        hours=8,
        qualification=str(proof),
        code_commit="qualified",
        code_layer="layer",
        container_hash="container",
        seed=1729,
        learning_rate=1e-4,
        gpus=2,
        gpu="h200",
        initial_checkpoint=None,
        evaluate_only=False,
        after=None,
        wrapper="wrapper",
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="12345")

    monkeypatch.setattr("subprocess.run", run)
    return args, calls


def test_h200_routing_checkpoint_recovery_and_no_duplicate_submission(
    tmp_path, monkeypatch
):
    args, calls = setup(tmp_path, monkeypatch)
    SUBMIT(args)
    command = calls[0][0]
    assert "--comment=preemption=yes;requeue=true" in command
    assert "--requeue" in command and "--signal=B:USR1@300" in command
    assert "--constraint=h200" in command
    assert not any(x.startswith("--partition") for x in command)
    record = json.loads((Path(args.root) / "A-submission.json").read_text())
    assert record["id"] == "12345"
    assert record["module_args"][record["module_args"].index("--accumulate") + 1] == "2"
    with pytest.raises(FileExistsError):
        SUBMIT(args)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("code_commit", "different"),
        ("arm", "B"),
        ("phase", "joint"),
        ("deadline", "2000-01-01T00:00:00Z"),
    ],
)
def test_mismatched_proof_or_expired_deadline_prevents_submission(
    tmp_path, monkeypatch, field, value
):
    args, calls = setup(tmp_path, monkeypatch)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        SUBMIT(args)
    assert not calls
