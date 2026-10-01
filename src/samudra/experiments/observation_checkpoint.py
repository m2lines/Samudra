# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify explicitly declared fixed-budget checkpoints without relabeling selection."""

import json
from pathlib import Path

import torch

from samudra.experiments.observation_pilot import digest


def load_fixed_checkpoint(run, name, split, om4_updates, observation_updates):
    run = Path(run)
    if om4_updates < 0 or observation_updates < 0:
        raise ValueError("Fixed-budget counts must be nonnegative")
    if name != f"joint-{observation_updates:05d}.pt":
        raise ValueError("Fixed-budget filename does not match observation count")
    if split == "test" and not (run / "TRAIN_COMPLETE.json").exists():
        raise ValueError("Held-out fixed-budget evaluation requires completed training")
    manifest = json.loads((run / "manifest.json").read_text())
    path = run / name
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"om4": om4_updates, "observation": observation_updates}
    if (
        saved.get("manifest") != manifest
        or saved.get("task_counts") != expected
        or saved.get("global_step") != sum(expected.values())
        or saved.get("step") != observation_updates
        or saved.get("phase") != "joint"
    ):
        raise ValueError("Fixed-budget checkpoint lineage/counts differ")
    lineage = {
        "checkpoint_policy": "fixed-budget, not validation-selected",
        "checkpoint": name,
        "checkpoint_sha256": digest(path),
        "global_step": saved["global_step"],
        "task_counts": expected,
        "training_manifest_sha256": digest(run / "manifest.json"),
    }
    # Do not retain optimizer tensors during evaluation.
    return saved["model"], lineage
