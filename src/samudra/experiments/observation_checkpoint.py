# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify explicitly declared fixed-budget checkpoints without relabeling selection."""

import json
import re
from pathlib import Path

import torch

from samudra.experiments.observation_pilot import digest


def load_quarter_endpoint(run, manifest, om4_updates, observation_updates):
    """Read a completed native-quarter endpoint without converting checkpoints."""
    complete_path = run / "TRAIN_COMPLETE.json"
    if not complete_path.is_file():
        raise ValueError("Quarter endpoint requires completed training")
    complete = json.loads(complete_path.read_text())
    path = run / "quarter-last.pt"
    checksum = digest(path)
    if checksum != complete.get("checkpoint_sha256"):
        raise ValueError("Quarter endpoint checkpoint hash differs")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    contract = saved.get("quarter_contract", {})
    schedules = {
        "u-global": [3000, 3000, "mixed"],
        "observation-only": [0, 6000, "scratch"],
        "u-multitask-early": [3000, 3000, "mixed"],
        "u-multitask-early-latent": [3000, 3000, "mixed"],
    }
    arm = contract.get("arm")
    expected = {"om4": om4_updates, "observation": observation_updates}
    if arm in ("u-multitask-early", "u-multitask-early-latent"):
        expected.update(om4_recent=1500, om4_early=1500)
    if (
        contract.get("version") != 1
        or arm not in schedules
        or contract.get("schedule") != schedules[arm]
        or [om4_updates, observation_updates] != schedules[arm][:2]
        or saved.get("completed") != 6000
        or saved.get("last_validated") != 6000
        or saved.get("diagnostic_only") is not False
        or saved.get("task_counts") != expected
        or complete.get("global_step") != 6000
        or complete.get("updates") != 6000
        or complete.get("task_counts") != expected
        or complete.get("quarter_contract") != contract
    ):
        raise ValueError("Quarter endpoint completion/science/counts differ")
    producer = contract.get("code_commit", "")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", producer)
        or manifest.get("code_commit") != producer
        or manifest.get("runtime_extension_sha256")
        != contract.get("runtime_extension_sha256")
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", contract.get(key, ""))
            for key in (
                "runtime_extension_sha256",
                "runner_sha256",
                "pilot_sha256",
                "distributed_helper_sha256",
                "selection_reference_sha256",
            )
        )
    ):
        raise ValueError("Quarter endpoint producer/runtime differs from manifest")
    options = {
        "initializer_architecture": "unet",
        "surface_policy": "observed-only",
        "task_conditioning": "input-adapters",
        "latent_channels": 10 if arm == "u-multitask-early-latent" else 0,
    }
    science = {
        "normalization": "instance",
        "evolution_architecture": "extent-unet",
        "surface_fill": "zero",
        "completion_weight": 0.1,
        "reconstruction_weight": 0.1,
        "global_observations": True,
        "seed": 1729,
    }
    arguments = manifest.get("arguments", {})
    expected_arguments = {
        **science,
        **options,
        "from_scratch": True,
        "observation_normalization": True,
        "strict_velocity_support": True,
        "selection_metric": "rmse-only",
        "core_lr": 1e-4,
        "accumulate": 8,
        "fit_probe": False,
    }
    if (
        manifest.get("normalization_mode") != "observation-only"
        or contract.get("model_options") != options
        or any(contract.get(k) != v for k, v in science.items())
        or contract.get("learning_rate") != 1e-4
        or contract.get("global_batch") != 8
        or contract.get("world_size") not in (4, 8)
        or any(arguments.get(k) != v for k, v in expected_arguments.items())
        or not arguments.get("iap_grid")
    ):
        raise ValueError("Quarter endpoint manifest/model science differs")
    data_hashes = contract.get("data", {})
    for name, field in (
        ("SHA256SUMS", "data_manifest_sha256"),
        ("grid.npz", "grid_sha256"),
        ("statistics.npz", "statistics_sha256"),
        ("iap-grid.npz", "iap_grid_sha256"),
    ):
        if (
            not re.fullmatch(r"[0-9a-f]{64}", data_hashes.get(name, ""))
            or manifest.get(field) != data_hashes[name]
        ):
            raise ValueError("Quarter endpoint manifest data differs: " + name)
    return saved["model"], {
        "checkpoint_policy": "fixed-budget, not validation-selected",
        "checkpoint": path.name,
        "checkpoint_sha256": checksum,
        "global_step": 6000,
        "task_counts": expected,
        "training_manifest_sha256": digest(run / "manifest.json"),
        "training_complete_sha256": digest(complete_path),
        "training_producer": producer,
        "quarter_contract": contract,
    }


def load_fixed_checkpoint(run, name, split, om4_updates, observation_updates):
    run = Path(run)
    if om4_updates < 0 or observation_updates < 0:
        raise ValueError("Fixed-budget counts must be nonnegative")
    if name == "quarter-last.pt":
        manifest = json.loads((run / "manifest.json").read_text())
        return load_quarter_endpoint(run, manifest, om4_updates, observation_updates)
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
