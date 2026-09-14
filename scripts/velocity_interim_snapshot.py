#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Capture checkpoint metadata and validation histories without evaluating test data."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path


def snapshot(root, load_checkpoints=True):
    if load_checkpoints:
        import torch

        torch.set_num_threads(1)
    campaign = json.loads((root / "campaign.json").read_text())
    result = {
        "captured_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "campaign": campaign,
        "runs": [],
    }
    for stage in ("screen", "confirm"):
        for key, job in campaign["jobs"].get(stage, {}).items():
            variant, seed = key.split(":")
            path = root / f"{stage}-{variant}-s{seed}"
            run = {
                "name": path.name,
                "stage": stage,
                "variant": variant,
                "seed": int(seed),
                "job_id": job,
                "started": path.exists(),
            }
            result["runs"].append(run)
            if not path.exists():
                continue
            run["config"] = json.loads((path / "config.json").read_text())
            run["provenance"] = json.loads((path / "run-provenance.json").read_text())
            best_path = path / "best.pt"
            if load_checkpoints and best_path.exists():
                with best_path.open("rb") as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest()
                    f.seek(0)
                    state = torch.load(f, map_location="cpu", weights_only=False)
                run["best_checkpoint"] = {
                    "path": str(best_path),
                    "sha256": digest,
                    "update": state["update"],
                    "training_gpu_hours": state["elapsed"]
                    * state["config"]["world_size"]
                    / 3600,
                    "exposure": state["exposure"],
                    "validation_rmse_10d": state["best"],
                }
                del state
            raw = (path / "history.jsonl").read_bytes()
            # A concurrently appended final line may not yet be complete.
            rows = [
                json.loads(line)
                for line in raw.splitlines(keepends=True)
                if line.endswith(b"\n")
            ]
            run["complete"] = bool(rows and rows[-1].get("complete"))
            run["validation_history"] = []
            previous = {}
            for row in rows:
                if "train/loss" in row:
                    previous = row
                    run["latest_training"] = row
                if "validation/vector_rmse_10d" in row:
                    point = {
                        **row,
                        "preceding_training_update": previous.get("update"),
                        "approx_training_gpu_hours": previous.get("gpu_hours"),
                        "approx_duacs_examples": previous.get("exposure/duacs", 0),
                    }
                    run["validation_history"].append(point)
                    if (
                        run.get("best_checkpoint", {}).get("validation_rmse_10d")
                        == row["validation/vector_rmse_10d"]
                    ):
                        run["best_checkpoint"]["metrics"] = row
            run["best_observed_validation"] = min(
                run["validation_history"],
                key=lambda point: point["validation/vector_rmse_10d"],
            )
            if "best_checkpoint" in run:
                assert "metrics" in run["best_checkpoint"], run["name"]
            if run["complete"]:
                run["completion"] = rows[-1]
                assert (
                    run["best_observed_validation"]["validation/vector_rmse_10d"]
                    == rows[-1]["best_validation_rmse_10d"]
                )
                if "best_checkpoint" in run:
                    assert (
                        run["best_checkpoint"]["validation_rmse_10d"]
                        == rows[-1]["best_validation_rmse_10d"]
                    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Read small logs without loading model checkpoints",
    )
    args = parser.parse_args()
    data = snapshot(args.campaign_root, load_checkpoints=not args.metadata_only)
    data["snapshot_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    with args.output.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
    for run in data["runs"]:
        best = run.get("best_checkpoint", {})
        print(
            run["name"],
            "complete",
            run.get("complete", False),
            "best",
            best.get(
                "validation_rmse_10d",
                run.get("best_observed_validation", {}).get(
                    "validation/vector_rmse_10d"
                ),
            ),
            "update",
            best.get("update"),
            flush=True,
        )
