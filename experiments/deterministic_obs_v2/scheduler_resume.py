# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Trigger one real Slurm restart, then compare production-training endpoints."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def checkpoint_update(records):
    """A checkpoint timing record is emitted only after the atomic save finishes."""
    return max(
        (r["updates"] for r in records if "timing/checkpoint_seconds" in r),
        default=0,
    )


def request_requeue(root, *, timeout=1800):
    job = os.environ["SLURM_JOB_ID"]
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    marker = root / "requeue-request.json"
    if marker.exists():
        raise RuntimeError("Refusing a second intentional restart")
    run = root / "resumed"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        metrics = run / "metrics.jsonl"
        if metrics.exists():
            # Ignore a final partially written line while the trainer is appending.
            lines = metrics.read_text().splitlines(keepends=True)
            records = [json.loads(line) for line in lines if line.endswith("\n")]
            update = checkpoint_update(records)
            if update >= 16:
                raise RuntimeError("Missed the intended main-training interruption")
            if update >= 6:
                receipt = {
                    "job_id": job,
                    "restart_count_before": restart,
                    "completed_checkpoint_update": update,
                    "requested_at": time.time(),
                    "wandb_id": (run / "wandb-id.txt").read_text().strip(),
                    "signal": "USR1 to Slurm batch shell, then scontrol requeue",
                }
                temporary = marker.with_suffix(".tmp")
                temporary.write_text(json.dumps(receipt, indent=2) + "\n")
                temporary.replace(marker)
                subprocess.run(["scancel", "--signal=USR1", "--batch", job], check=True)
                # Requeue must terminate this allocation, including this watcher.
                time.sleep(60)
                raise RuntimeError("Allocation survived the requested Slurm requeue")
        time.sleep(1)
    raise TimeoutError("No completed checkpoint available for scheduler interruption")


def audit(root):
    import torch

    from samudra.observations.qualify import equal_tree
    from samudra.observations.qualify_scaling import state_dict_digest
    from samudra.observations.shared_era5 import digest

    torch.set_num_threads(1)
    request = json.loads((root / "requeue-request.json").read_text())
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
    if (
        request["job_id"] != os.environ["SLURM_JOB_ID"]
        or restart <= request["restart_count_before"]
    ):
        raise ValueError("The requested job did not restart through Slurm")
    if (root / "resumed/wandb-id.txt").read_text().strip() != request["wandb_id"]:
        raise ValueError("W&B run identity changed during scheduler resume")
    paths = [root / name / "saved_nets/endpoint.pt" for name in ("control", "resumed")]
    expected, actual = [
        torch.load(p, map_location="cpu", weights_only=False, mmap=True) for p in paths
    ]
    equal_tree(actual, expected)
    if actual["updates"] != 18 or not actual["cooled"]:
        raise ValueError("Incomplete main/cooldown trajectory")
    if int(actual["ema"]["num_updates"]) != 2 or actual["pending_plan"] is not None:
        raise ValueError("Invalid cooldown EMA or pending plan")
    if len(actual["rank_rng"]) != int(os.environ["GPUS_PER_NODE"]):
        raise ValueError("Saved rank count differs from the allocation")
    for state in (expected, actual):
        if not all(torch.isfinite(value).all() for value in state["model"].values()):
            raise ValueError("Nonfinite final model")
    receipt = {
        "purpose": "Exact production-training resume across a real Slurm requeue",
        "job_id": os.environ["SLURM_JOB_ID"],
        "restart_count_after": restart,
        "request": request,
        "world_size": len(actual["rank_rng"]),
        "main_updates": 16,
        "cooldown_updates": 2,
        "exact_full_checkpoint_payload": True,
        "model_sha256": state_dict_digest(actual["model"]),
        "checkpoint_sha256": {str(path): digest(path) for path in paths},
        "exposure": actual["exposure"],
    }
    (root / "SCHEDULER_QUALIFIED.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("request", "audit"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    (request_requeue if args.mode == "request" else audit)(args.root)
