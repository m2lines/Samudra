# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Reject false scheduler-resume receipts and mismatched non-model state."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


@pytest.fixture
def resume_audit(tmp_path, monkeypatch):
    path = (
        Path(__file__).parents[1]
        / "experiments/deterministic_obs_v2/scheduler_resume.py"
    )
    spec = importlib.util.spec_from_file_location("scheduler_resume_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "1")
    request = {"job_id": "777", "restart_count_before": 0, "wandb_id": "same-run"}
    (tmp_path / "requeue-request.json").write_text(json.dumps(request))
    state = {
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"moment": torch.tensor([0.5])},
        "updates": 18,
        "cooled": True,
        "ema": {"num_updates": 2},
        "pending_plan": None,
        "rank_rng": [{"torch_rng": torch.tensor([3], dtype=torch.uint8)}],
        "exposure": {"obs": {"updates": 10}, "om4": {"updates": 8}},
    }
    for name in ("control", "resumed"):
        (tmp_path / name / "saved_nets").mkdir(parents=True)
        (tmp_path / name / "wandb-id.txt").write_text("same-run")
        torch.save(state, tmp_path / name / "saved_nets/endpoint.pt")
    return module


def test_real_restart_and_optimizer_payload_are_required(
    resume_audit, tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")
    with pytest.raises(ValueError, match="did not restart"):
        resume_audit.audit(tmp_path)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    endpoint = tmp_path / "resumed/saved_nets/endpoint.pt"
    state = torch.load(endpoint, weights_only=False)
    state["optimizer"]["moment"].add_(1)
    torch.save(state, endpoint)
    with pytest.raises(AssertionError):
        resume_audit.audit(tmp_path)
    assert not (tmp_path / "SCHEDULER_QUALIFIED.json").exists()


def test_exact_endpoint_and_tracking_identity(resume_audit, tmp_path):
    identity = tmp_path / "resumed/wandb-id.txt"
    identity.write_text("new-run")
    with pytest.raises(ValueError, match="W&B run identity"):
        resume_audit.audit(tmp_path)
    identity.write_text("same-run")
    resume_audit.audit(tmp_path)
    receipt = json.loads((tmp_path / "SCHEDULER_QUALIFIED.json").read_text())
    assert receipt["exact_full_checkpoint_payload"]
    assert receipt["restart_count_after"] == 1
