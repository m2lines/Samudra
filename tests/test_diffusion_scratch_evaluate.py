# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from samudra.experiments.diffusion_scratch_evaluate import (
    VALIDATION_ORIGINS,
    checkpoint_signature,
    commands,
    verify_output,
)
from samudra.experiments.observation_pilot import digest
from samudra.experiments.task_schedule import TaskSchedule


def test_quadratic_observation_milestone_keeps_actual_om4_exposure(tmp_path):
    schedule = TaskSchedule(10160, 10160, "mixed")
    assert schedule.counts(7560)["observation"] == 1999
    counts = schedule.counts(7561)
    assert counts == {"om4": 5561, "observation": 2000}
    checkpoint = tmp_path / "snapshot.pt"
    checkpoint.write_bytes(b"immutable checkpoint")
    saved = dict(counts=counts, step=7561, model={}, contract={})
    signature = checkpoint_signature(saved, checkpoint, "residual", 5561, 2000)
    annual = commands(tmp_path, checkpoint, tmp_path / "eval", signature)["annual"]
    assert annual[annual.index("--expected-om4-updates") + 1] == "5561"
    assert annual[annual.index("--expected-observation-updates") + 1] == "2000"
    with pytest.raises(ValueError, match="task exposure"):
        checkpoint_signature(saved, checkpoint, "residual", 2000, 2000)


def test_direct_marker_and_requested_arm_must_both_agree(tmp_path):
    checkpoint = tmp_path / "snapshot.pt"
    checkpoint.write_bytes(b"checkpoint")
    saved = dict(
        model={"processor.direct_prediction_version": torch.tensor(1)},
        processor_prediction="direct",
        counts={"om4": 10160, "observation": 10160},
        step=20320,
        contract={},
    )
    assert (
        checkpoint_signature(saved, checkpoint, "direct", 10160, 10160)[
            "processor_prediction"
        ]
        == "direct"
    )
    with pytest.raises(ValueError, match="requested arm"):
        checkpoint_signature(saved, checkpoint, "residual", 10160, 10160)
    saved["processor_prediction"] = "residual"
    with pytest.raises(ValueError, match="marker disagree"):
        checkpoint_signature(saved, checkpoint, "residual", 10160, 10160)


def test_resume_rejects_stale_checkpoint_and_corrupt_payload(tmp_path):
    output = tmp_path / "validation"
    output.mkdir()
    payload = output / "members.npz"
    payload.write_bytes(b"verified members")
    signature = dict(
        checkpoint_sha256="checkpoint",
        counts={"om4": 5561, "observation": 2000},
        sampling_steps=32,
    )
    receipt = dict(
        **signature,
        split="validation",
        metrics={"origins": VALIDATION_ORIGINS},
        files={payload.name: digest(payload)},
    )
    complete = output / "COMPLETE.json"
    complete.write_text(json.dumps(receipt))
    assert verify_output(tmp_path, "validation", signature)
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        verify_output(
            tmp_path, "validation", dict(signature, checkpoint_sha256="another")
        )
    payload.write_bytes(b"broken members")
    with pytest.raises(ValueError, match="hash/path"):
        verify_output(tmp_path, "validation", signature)


def test_incomplete_annual_members_do_not_skip_remaining_work(tmp_path):
    output = tmp_path / "annual"
    output.mkdir()
    signature = dict(
        checkpoint_sha256="checkpoint",
        counts={"om4": 5561, "observation": 2000},
        sampling_steps=32,
        members=8,
    )
    receipt = dict(
        inputs=signature,
        origins={origin: {} for origin in ["2015-01-01", "2018-01-01", "2021-01-01"]},
    )
    (output / "COMPLETE.json").write_text(json.dumps(receipt))
    assert not verify_output(tmp_path, "annual", signature)
