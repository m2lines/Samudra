# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify evaluation checkpoints without claiming stopped training completed."""

import json
from pathlib import Path

from samudra.experiments.observation_pilot import digest


def verify_checkpoint(path: Path, marker: str, *, stopped: bool = False):
    if stopped:
        receipt = json.loads((path.parent / "TRAINING_STOPPED.json").read_text())
        if (
            marker != "OBSERVATION_COMPLETE.json"
            or receipt["status"] != "stopped_by_user"
            or not receipt["reason"]
            or receipt["checkpoint_sha256"][path.name] != digest(path)
        ):
            raise ValueError("Stopped checkpoint receipt differs")
        return dict(status="stopped_by_user", receipt=receipt)
    completed = json.loads((path.parent / marker).read_text())
    if not completed["state"]["complete"] or completed[
        "best_checkpoint_sha256"
    ] != digest(path):
        raise ValueError("Training stage/checkpoint not verified")
    return dict(status="budget_complete", receipt=completed)
