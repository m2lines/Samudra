#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect completed early/fine arms with checkpoint and exposure verification."""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

from run_early_fine_wave import ARMS


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(root):
    paths = read(root / "paths.json")
    cooldown = paths.get("cooldown")
    names = [*ARMS, "U-global"] if cooldown else ARMS
    result = dict(
        collected_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        root=str(root),
        paths=paths,
        collector_sha256=digest(Path(__file__)),
        selection_reference=read(root / "selection-reference.json"),
        selection_reference_sha256=digest(root / "selection-reference.json"),
        arms={},
    )
    for name in names:
        run = root / name
        training = read(run / "TRAIN_COMPLETE.json")
        complete = read(run / "test-selected/COMPLETE.json")
        fingerprint = read(run / "test-selected/evaluation-input.json")
        best = read(run / "best.json")
        manifest = read(run / "manifest.json")
        if manifest["code_commit"] != paths["producer"]:
            raise ValueError("Producer mismatch")
        if training["global_step"] != 4000 or training["task_counts"] != {
            "om4": 2000,
            "observation": 2000,
        }:
            raise ValueError("Incomplete training")
        if {
            training["best_checkpoint_sha256"],
            complete["selected_sha256"],
            fingerprint["checkpoint_sha256"],
            best["checkpoint_sha256"],
        } != {digest(run / "best.pt")}:
            raise ValueError("Checkpoint lineage mismatch")
        if (
            complete["origins"] != 96
            or fingerprint["split"] != "test"
            or not fingerprint["global_observations"]
        ):
            raise ValueError("Incorrect evaluation cohort")
        exposure = read(run / "EXPOSURE.json")
        expected = (
            dict(om4_recent=2000, om4_early=0, observation=2000)
            if name == "U-global"
            else dict(om4_recent=1000, om4_early=1000, observation=2000)
        )
        if exposure != expected:
            raise ValueError("Unexpected exposure")
        events = [
            json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()
        ]
        updates = [e for e in events if e["event"] == "joint_train"]
        if [e["global_step"] for e in updates] != list(range(1, 4001)):
            raise ValueError("Incomplete or duplicated update log")
        if not all(
            math.isfinite(e[k]) for e in updates for k in ("loss", "gradient_norm")
        ):
            raise ValueError("Nonfinite training event")
        migration = None
        if cooldown:
            migration = read(run / "MIGRATION.json")
            if not migration["all_other_checkpoint_state_exact"]:
                raise ValueError("Cooldown source state audit failed")
            for event in updates[migration["global_step"] :]:
                fraction = max(
                    0,
                    min(
                        1,
                        (event["global_step"] - cooldown["start"])
                        / (cooldown["end"] - cooldown["start"]),
                    ),
                )
                expected_lr = 1e-4 * (1 - (1 - cooldown["final_ratio"]) * fraction)
                if not math.isclose(event["lr"], expected_lr, rel_tol=1e-12):
                    raise ValueError("Executed cooldown learning rate differs")
            if best["global_step"] < migration["global_step"]:
                raise ValueError("Selected checkpoint precedes cooldown fork")
        result["arms"][name + ("-cooldown" if cooldown else "")] = dict(
            migration=migration,
            training_complete=training,
            evaluation_complete=complete,
            best=best,
            manifest=manifest,
            exposure=exposure,
            final_training_event=updates[-1],
            validation_events=[
                dict(e, optimizer_updates=e["global_step"])
                for e in events
                if e["event"] == "joint_validation"
            ],
            files={
                p.name: dict(sha256=digest(p), data=read(p))
                for p in sorted((run / "test-selected").glob("*.json"))
            },
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root), indent=2, allow_nan=False))
