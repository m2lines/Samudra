# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Start a spike replay or counterfactual from one of #1037's milestone checkpoints.

A replay resumes the parent exactly: model, optimizer, EMA, sampler, pending
sample plan and every rank's RNG. The parent's manifest must match this run's,
except for the differences listed here, and each one is recorded in
``lineage.json``:

- Code may differ only in the files that add the probe, the loss guard and the
  configurable Adam betas (``ALLOWED_CODE``). ``tests/test_spike_probe.py`` checks
  on CPU that this code reproduces the original training exactly when the probe is
  on and no intervention is configured. On GPU, the replay control has to
  reproduce the W&B record of the original run before any counterfactual counts.
- Config may differ in the interventions (``INTERVENTIONS``) and in save cadence.
  Config keys the parent predates must hold their neutral defaults, unless they
  are interventions. The planned duration (``DURATION``) may differ only when the
  whole replay lies in the child's constant-LR main phase, where it cannot affect
  any update; this lets the update-3308 replay start from the 2,000-update run.
- ``--reseed`` replaces the sampler state, so the replay draws a different
  sequence of training windows and corruption masks from the same weights.

Learning rate and Adam betas live in the optimizer state as well as the config;
the fork writes the new values into the restored parameter groups.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from samudra.observations.config import ObservationTrainConfig
from samudra.observations.snapshot_data import snapshot_manifest
from samudra.observations.state import atomic_save, write_json
from samudra.utils.train import CheckpointPaths

ALLOWED_CODE = frozenset(
    {
        "observations/config.py",
        "observations/pipeline.py",
        "observations/probes.py",
        "observations/snapshot.py",
        "observations/state.py",
        "observations/train.py",
        "observations/training.py",
    }
)
INTERVENTIONS = frozenset(
    {"gradient_clip", "adam_betas", "learning_rate", "loss_guard"}
)
OPERATIONAL = frozenset({"save_freq", "milestone_updates"})
DURATION = frozenset(
    {"main_updates", "post_training_updates", "post_training_scheduler"}
)
NEUTRAL = {"adam_betas": [0.9, 0.999], "loss_guard": None}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fork(
    parent: dict[str, Any], child: dict[str, Any], *, stop_after: int | None = None
) -> dict[str, dict[str, Any]]:
    """Return the differing config values; raise on any unpermitted difference."""
    permitted = INTERVENTIONS | OPERATIONAL
    if (
        stop_after is not None
        and child["config"]["scheduler"] is None
        and stop_after <= child["config"]["main_updates"]
    ):
        permitted |= DURATION
    for key in ("version", "selection_metric", "preparation_sha256", "runtime"):
        if parent[key] != child[key]:
            raise ValueError(f"Replay {key} differs from the parent")
    old_code, new_code = parent["code_sha256"], child["code_sha256"]
    changed_code = {
        name
        for name in set(old_code) | set(new_code)
        if old_code.get(name) != new_code.get(name)
    }
    if changed_code - ALLOWED_CODE:
        raise ValueError(
            f"Code differs beyond the probe: {sorted(changed_code - ALLOWED_CODE)}"
        )
    old_cfg, new_cfg = parent["config"], child["config"]
    changes = {}
    for key in sorted(set(old_cfg) | set(new_cfg)):
        if key not in new_cfg:
            raise ValueError(f"Replay config lacks the parent's {key}")
        if key not in old_cfg:
            if key not in INTERVENTIONS and new_cfg[key] != NEUTRAL.get(key):
                raise ValueError(f"New config key {key} must hold its neutral default")
            if new_cfg[key] != NEUTRAL.get(key):
                changes[key] = {"parent": NEUTRAL.get(key), "child": new_cfg[key]}
            continue
        if old_cfg[key] == new_cfg[key]:
            continue
        if key not in permitted:
            raise ValueError(f"Replay may not change {key}")
        changes[key] = {"parent": old_cfg[key], "child": new_cfg[key]}
    return changes


def fork(
    saved: dict[str, Any],
    child: dict[str, Any],
    cfg: ObservationTrainConfig,
    reseed: int | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return the checkpoint rewritten for the replay, and the config changes."""
    if saved["in_post_training"] or saved["scheduler"] is not None:
        raise ValueError("Replays start from constant-LR main-phase checkpoints")
    if cfg.stop_after is None or cfg.stop_after <= saved["updates"]:
        raise ValueError("A replay needs stop_after beyond the parent's update")
    changes = validate_fork(saved["manifest"], child, stop_after=cfg.stop_after)
    result = copy.copy(saved)
    result["manifest"] = child
    result["best_score"] = float("inf")
    result["loss_guard"] = {}
    optimizer = copy.deepcopy(saved["optimizer"])
    for group in optimizer["param_groups"]:
        group["lr"] = group["initial_lr"] = cfg.learning_rate
        group["betas"] = tuple(cfg.adam_betas)
    result["optimizer"] = optimizer
    if reseed is not None:
        result["sampler_rng"] = np.random.default_rng(reseed).bit_generator.state
        result["pending_plan"] = None
        result["replan_pool_entries"] = None
        changes["sampler_seed"] = {"parent": "checkpoint state", "child": reseed}
    return result, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument(
        "--parent-sha256", help="Expected checksum; computed and recorded if omitted"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reseed", type=int)
    args = parser.parse_args()
    parent_sha256 = sha256(args.parent)
    if args.parent_sha256 is not None and parent_sha256 != args.parent_sha256:
        raise ValueError("Parent checkpoint checksum differs")
    cfg = ObservationTrainConfig.model_validate(yaml.safe_load(args.config.read_text()))
    if cfg.resume_ckpt_path is not None or cfg.fork_ckpt_path is not None:
        raise ValueError("A replay resumes from the checkpoint this tool writes")
    saved = torch.load(args.parent, map_location="cpu", weights_only=False)
    result, changes = fork(saved, snapshot_manifest(cfg), cfg, args.reseed)
    lineage = dict(
        parent=str(args.parent),
        parent_sha256=parent_sha256,
        parent_updates=saved["updates"],
        stop_after=cfg.stop_after,
        changes=changes,
        parent_manifest=saved["manifest"],
        child_manifest=result["manifest"],
    )
    cfg.prepare_output_dirs()
    target = CheckpointPaths(cfg.experiment.nets_dir).latest_checkpoint_path
    receipt = cfg.experiment.output_dir / "lineage.json"
    if receipt.exists() or target.exists():
        raise ValueError(f"{cfg.experiment.output_dir} already holds a replay")
    write_json(lineage, receipt)
    atomic_save(result, target)
    print(json.dumps(dict(event="replay_forked", output=str(target), changes=changes)))


if __name__ == "__main__":
    main()
