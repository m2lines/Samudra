#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect completed extent arms, retaining raw metrics and verified lineage."""

import argparse
import datetime
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(root, arms):
    result = dict(
        collected_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        root=str(root),
        paths=read(root / "paths.json"),
        collector_sha256=digest(Path(__file__)),
        selection_reference=read(root / "selection-reference.json"),
        selection_reference_sha256=digest(root / "selection-reference.json"),
        arms={},
    )
    for name in arms:
        run = root / name
        training = read(run / "TRAIN_COMPLETE.json")
        complete = read(run / "test-selected" / "COMPLETE.json")
        best = read(run / "best.json")
        fingerprint = read(run / "test-selected" / "evaluation-input.json")
        manifest = read(run / "manifest.json")
        if manifest["code_commit"] != result["paths"]["producer"]:
            raise ValueError("Training producer differs from frozen root")
        if training["global_step"] != 4000 or training["task_counts"] != dict(
            om4=2000, observation=2000
        ):
            raise ValueError("Incomplete or unexpected training budget")
        hashes = {
            training["best_checkpoint_sha256"],
            complete["selected_sha256"],
            best["checkpoint_sha256"],
            fingerprint["checkpoint_sha256"],
        }
        if hashes != {digest(run / "best.pt")}:
            raise ValueError("Selected checkpoint lineage differs")
        if (
            complete["origins"] != 96
            or fingerprint["split"] != "test"
            or not fingerprint["global_observations"]
        ):
            raise ValueError("Incomplete or incompatible global evaluation")
        events = [
            json.loads(line)
            for line in (run / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        updates = [e for e in events if e.get("event") == "joint_train"]
        actual_counts = {0: 0}
        actual_counts.update(
            {
                e["global_step"]: e["optimizer_updates"]
                for e in events
                if e.get("event") in {"joint_train", "joint_skip"}
            }
        )
        exposure = read(run / "EXPOSURE.json")
        final = updates[-1]
        if (
            exposure["optimizer_updates"]
            != exposure["om4_global"] + exposure["om4_patch"] + exposure["observation"]
        ):
            raise ValueError("Inconsistent actual update accounting")
        if final["optimizer_updates"] != exposure["optimizer_updates"]:
            raise ValueError("Final training event differs from completion exposure")
        targets = None
        if (run / "ACCESSORY_TARGETS.json").exists():
            targets = read(run / "ACCESSORY_TARGETS.json")
            qualified = read(root / ("probe-" + name) / "ACCESSORY_TARGETS.json")
            if targets != qualified:
                raise ValueError(
                    "Production accessory targets differ from qualification"
                )
            if targets["mode"] != manifest["arguments"]["auxiliary_target_mode"]:
                raise ValueError("Target mode differs from training arguments")
        accessory = [event for event in updates if "accessory_mse" in event]
        accessory_summary = {}
        if accessory:
            if len(accessory) != 2000 or any(
                event["task"] != "om4" or event["source_extent"] != "global"
                for event in accessory
            ):
                raise ValueError("Unexpected accessory-loss update accounting")
            for phase, lower in (("all", 0), ("last_1000_schedule_slots", 3000)):
                selected = [
                    event for event in accessory if event["global_step"] > lower
                ]
                accessory_summary[phase] = dict(
                    om4_updates=len(selected),
                    mean_training_accessory_mse=statistics.mean(
                        event["accessory_mse"] for event in selected
                    ),
                    mean_training_physical_objective=statistics.mean(
                        event["physical_objective"] for event in selected
                    ),
                )
        result["arms"][name] = dict(
            accessory_targets=targets,
            accessory_training_summary=accessory_summary,
            training_complete=training,
            evaluation_complete=complete,
            best=best,
            selected_optimizer_updates=actual_counts[best["global_step"]],
            exposure=exposure,
            manifest=manifest,
            final_training_event=final,
            validation_events=[
                dict(e, optimizer_updates=actual_counts[e["global_step"]])
                for e in events
                if e.get("event") == "joint_validation"
            ],
            files={
                p.name: dict(sha256=digest(p), data=read(p))
                for p in sorted((run / "test-selected").glob("*.json"))
            },
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.arms), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
