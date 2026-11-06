# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Explicitly derive a smaller-world checkpoint from a trusted deterministic run.

This utility runs outside the frozen training package. It never changes the
checkpoint manifest, bypasses restore validation, or overwrites a source/output.
The supported runtime is the audited 7bef723 observation pipeline. Global plans
and per-example corruption seeds are independent of rank layout. Floating-point
reduction order changes across layouts, so future trajectories are NOT claimed
bitwise identical to continuation on the old world size.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

# SHA-256 of the canonical code_sha256 mapping, not a credential.
AUDITED_CODE_DIGEST = "322327bb942513ff11d1433064fb128a2897649878430b3fad2705b2718eb9b8"  # pragma: allowlist secret


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def equal_tree(left: Any, right: Any) -> None:
    """Reject any loaded-payload difference, including tensor dtype and shape."""
    if type(left) is not type(right):
        raise ValueError("Payload types differ")
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError("Tensor payload differs")
    elif isinstance(left, np.ndarray):
        if left.dtype != right.dtype or not np.array_equal(left, right):
            raise ValueError("Array payload differs")
    elif dataclasses.is_dataclass(left):
        for field in dataclasses.fields(left):
            equal_tree(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("Payload keys differ")
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError("Payload lengths differ")
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    elif left != right:
        raise ValueError("Payload values differ")


def migrated_payload(saved: dict[str, Any], rank_map: list[int]) -> dict[str, Any]:
    """Return a shallow copy with only the rank_rng list replaced."""
    required = {
        "manifest",
        "model",
        "optimizer",
        "ema",
        "updates",
        "cooled",
        "best_score",
        "exposure",
        "sampler_rng",
        "pools",
        "pending_plan",
        "rank_rng",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
        "cuda_rng_scope",
    }
    if set(saved) != required:
        raise ValueError("Unsupported checkpoint fields")
    manifest = saved["manifest"]
    code_digest = hashlib.sha256(
        json.dumps(
            manifest["code_sha256"], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if code_digest != AUDITED_CODE_DIGEST:
        raise ValueError("Checkpoint runtime has not been audited for rank migration")
    config = manifest["config"]
    if not config["update_prefetch"] or config["batch_size"] != 1:
        raise ValueError("Requires the audited global-plan update pipeline")
    model = config["model"]
    if model["forcing_mode"] != "shared_era5":
        raise ValueError("Only deterministic shared-ERA5 models are supported")
    for name in ("backbone", "initializer_backbone", "evolution_backbone"):
        backbone = model[name]
        if backbone is None or backbone["drop_path_rate"] != 0:
            raise ValueError(
                "Requires explicit backbones with zero stochastic drop path"
            )
        if backbone["core_block"]["block_type"] != "conv_next_block":
            raise ValueError("Unsupported backbone block")
    old = saved["rank_rng"]
    if not isinstance(old, list) or not old:
        raise ValueError("Missing per-rank RNG state")
    if len(old) != 4 or rank_map != [0, 1]:
        raise ValueError("Only the qualified four-to-two mapping [0, 1] is supported")
    if not rank_map or len(rank_map) >= len(old) or len(set(rank_map)) != len(rank_map):
        raise ValueError("Rank mapping must be a smaller nonempty unique subset")
    if rank_map[0] != 0 or any(i < 0 or i >= len(old) for i in rank_map):
        raise ValueError("Rank mapping must preserve rank zero and use existing ranks")
    batch = config["gradient_accumulation_steps"]
    if batch % len(old) or batch % len(rank_map):
        raise ValueError("Global batch must be divisible by both world sizes")
    for rng in old:
        if set(rng) != {
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
            "cuda_rng_scope",
        }:
            raise ValueError("Unsupported rank RNG fields")
        if rng["cuda_rng_scope"] != "current" or len(rng["cuda_rng"]) != 1:
            raise ValueError("Requires one current-device CUDA RNG state per rank")
    plan = saved["pending_plan"]
    if not isinstance(plan, dict) or set(plan) != {"update", "source", "examples"}:
        raise ValueError("Requires an unfinished global pending plan")
    if plan["update"] != saved["updates"] or len(plan["examples"]) != batch:
        raise ValueError(
            "Pending plan does not match the checkpoint update/global batch"
        )
    for example in plan["examples"]:
        if set(example) != {"origin", "month", "coverage", "seed", "entry"}:
            raise ValueError("Unsupported pending-plan example fields")
        if not isinstance(example["seed"], int) or not 0 <= example["seed"] < 2**31:
            raise ValueError("Missing explicit per-example corruption seed")
    return {**saved, "rank_rng": [old[i] for i in rank_map]}


def migrate(
    source: str | Path, output: str | Path, rank_map: list[int]
) -> dict[str, Any]:
    source, output = Path(source), Path(output)
    sidecar = Path(str(output) + ".migration.json")
    if source.resolve() == output.resolve() or output.exists() or sidecar.exists():
        raise ValueError("Migration requires new output and sidecar paths")
    parent_sha = digest(source)
    saved = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    derived = migrated_payload(saved, rank_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save(derived, stream)
        stream.flush()
        os.fsync(stream.fileno())
    actual = torch.load(output, map_location="cpu", weights_only=False, mmap=True)
    equal_tree(derived, actual)
    unchanged = sorted(set(saved) - {"rank_rng"})
    for key in unchanged:
        equal_tree(saved[key], actual[key])
    if digest(source) != parent_sha:
        raise ValueError("Source checkpoint changed during migration")
    receipt = {
        "source": str(source.resolve()),
        "source_sha256": parent_sha,
        "output": str(output.resolve()),
        "output_sha256": digest(output),
        "old_world_size": len(saved["rank_rng"]),
        "new_world_size": len(rank_map),
        "new_rank_to_old_rank": rank_map,
        "changed_fields": ["rank_rng"],
        "unchanged_fields_verified": unchanged,
        "manifest_unchanged": True,
        "source_unchanged": True,
        "updates": saved["updates"],
        "cross_layout_bitwise_trajectory_claim": False,
        "limitation": "Reduction order changes; this preserves global scientific state, not cross-layout bitwise continuation.",
    }
    with sidecar.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Trusted locally produced checkpoint")
    parser.add_argument(
        "output", type=Path, help="New derived checkpoint; never overwritten"
    )
    parser.add_argument("--rank-map", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.output, args.rank_map), indent=2))
