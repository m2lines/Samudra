# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Explicitly migrate an uncooled snapshot endpoint to a longer OBS cooldown run.

Only the configuration validator may differ in package code. Model, optimizer,
EMA, sampler and rank RNG state are preserved. Original checkpoints stay intact.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml

from samudra.observations.config import ObservationTrainConfig
from samudra.observations.snapshot_data import snapshot_manifest
from samudra.observations.state import atomic_save, write_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_extension(saved, current, old_config_path, *, qualification=False):
    parent = copy.deepcopy(saved["manifest"])
    child = copy.deepcopy(current)
    old_cfg, new_cfg = parent["config"], child["config"]
    if (
        saved["in_post_training"]
        or saved["scheduler"] is not None
        or saved["pending_plan"] is not None
        or saved.get("replan_pool_entries") is not None
        or saved["updates"] != old_cfg["main_updates"]
        or old_cfg["post_training_updates"] != 0
        or old_cfg["post_training_scheduler"] is not None
        or old_cfg["scheduler"] is not None
    ):
        raise ValueError("Parent must be a completed constant-LR main-only run")
    if (
        new_cfg["main_updates"] <= saved["updates"]
        or new_cfg["post_training_updates"] <= 0
        or new_cfg["post_training_scheduler"]
        != {"type": "linear_cooldown", "final_lr": 1e-6, "target_epochs": None}
    ):
        raise ValueError(
            "Extension requires a longer main phase and linear OBS cooldown"
        )
    changes: tuple[str, ...] = (
        "main_updates",
        "post_training_updates",
        "post_training_scheduler",
        "milestone_updates",
    )
    if qualification:
        changes += ("save_freq", "validation_freq", "validation_examples")
    changed = {key: {"parent": old_cfg[key], "child": new_cfg[key]} for key in changes}
    for key in changes:
        old_cfg.pop(key)
        new_cfg.pop(key)
    code_key = "observations/config.py"
    old_code = Path(old_config_path).read_text()
    if hashlib.sha256(old_code.encode()).hexdigest() != parent["code_sha256"][code_key]:
        raise ValueError("Parent configuration source differs from checkpoint")
    allowed_code = old_code.replace(
        "                or self.post_training_updates != 0\n", ""
    ).replace("no monthly loss/cooldown,", "no monthly loss,")
    if allowed_code == old_code:
        raise ValueError("Expected original snapshot cooldown guard is absent")
    if (
        hashlib.sha256(allowed_code.encode()).hexdigest()
        != child["code_sha256"][code_key]
    ):
        raise ValueError(
            "Runtime changes exceed the reviewed cooldown validator change"
        )
    parent["code_sha256"].pop(code_key)
    child["code_sha256"].pop(code_key)
    if parent != child:
        raise ValueError("Checkpoint data, runtime, code or scientific recipe differs")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--parent-config-source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qualification", action="store_true")
    args = parser.parse_args()
    if sha256(args.parent) != args.parent_sha256:
        raise ValueError("Parent checkpoint checksum differs")
    cfg = ObservationTrainConfig.model_validate(yaml.safe_load(args.config.read_text()))
    if cfg.resume_ckpt_path is not None or cfg.fork_ckpt_path is not None:
        raise ValueError(
            "Extension uses its own latest checkpoint for subsequent resumes"
        )
    saved = torch.load(args.parent, map_location="cpu", weights_only=False)
    current = snapshot_manifest(cfg)
    changes = validate_extension(
        saved, current, args.parent_config_source, qualification=args.qualification
    )
    if len(saved.get("rank_rng") or []) != 4:
        raise ValueError("Expected the original four-rank RNG state")
    if saved["updates"] != 2000:
        raise ValueError("This campaign extends the approved 2000-update parents")
    lineage = dict(
        parent=str(args.parent),
        parent_sha256=args.parent_sha256,
        parent_manifest=saved["manifest"],
        child_manifest=current,
        changes=changes,
        preserved="model, optimizer, EMA, sampler, source exposure, global and rank RNG",
        reset="best_score for new trajectory selection",
    )
    cfg.prepare_output_dirs()
    target = cfg.experiment.nets_dir / "ckpt.pt"
    receipt = cfg.experiment.output_dir / "lineage.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != lineage:
            raise ValueError("Existing continuation has different lineage")
        if target.exists():
            return
    elif target.exists():
        raise ValueError("Existing checkpoint has no approved continuation lineage")
    write_json(lineage, receipt)
    saved["manifest"] = current
    saved["best_score"] = float("inf")
    saved["continuation"] = dict(
        parent=str(args.parent), sha256=args.parent_sha256, changes=changes
    )
    atomic_save(saved, target)
    print(
        json.dumps(
            dict(
                event="checkpoint_extended",
                output=str(target),
                updates=saved["updates"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
