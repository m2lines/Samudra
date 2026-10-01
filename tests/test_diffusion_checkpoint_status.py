# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from samudra.experiments.diffusion_checkpoint_status import verify_checkpoint
from samudra.experiments.observation_pilot import digest


def test_stopped_checkpoint_requires_explicit_opt_in_and_unchanged_bytes(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"selected weights")
    receipt = dict(
        status="stopped_by_user",
        reason="User stop",
        checkpoint_sha256={"best.pt": digest(checkpoint)},
    )
    (tmp_path / "TRAINING_STOPPED.json").write_text(json.dumps(receipt))
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(checkpoint, "OBSERVATION_COMPLETE.json")
    assert (
        verify_checkpoint(checkpoint, "OBSERVATION_COMPLETE.json", stopped=True)[
            "status"
        ]
        == "stopped_by_user"
    )
    with pytest.raises(ValueError):
        verify_checkpoint(checkpoint, "PRETRAIN_COMPLETE.json", stopped=True)
    checkpoint.write_bytes(b"changed weights")
    with pytest.raises(ValueError):
        verify_checkpoint(checkpoint, "OBSERVATION_COMPLETE.json", stopped=True)


def test_incomplete_training_marker_is_not_completion(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"selected weights")
    marker = tmp_path / "OBSERVATION_COMPLETE.json"
    receipt = dict(
        state=dict(complete=False), best_checkpoint_sha256=digest(checkpoint)
    )
    marker.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_checkpoint(checkpoint, marker.name)
    receipt["state"]["complete"] = True
    marker.write_text(json.dumps(receipt))
    assert verify_checkpoint(checkpoint, marker.name)["status"] == "budget_complete"
