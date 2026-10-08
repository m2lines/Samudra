#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Fork retained optimizer checkpoints after fresh cooldown qualifications pass."""

import argparse
import copy
import json
from pathlib import Path

import torch

from samudra.experiments.observation_joint import assert_exact_state
from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from samudra.experiments.task_schedule import TaskSchedule


def audit_arguments(old, new):
    operational = {
        "output",
        "name",
        "qualification",
        "joint_qualification",
        "selection_reference",
        "data",
        "om4_data",
        "joint_hours",
        "deadline",
    }
    additions = {
        "early_data",
        "recent_only",
        "cooldown_start",
        "cooldown_end",
        "cooldown_final_ratio",
    }
    defaults = dict(
        auxiliary_cache=None,
        auxiliary_weight=0.0,
        auxiliary_target_mode="aligned",
        patch_mode="shared",
        patch_leads=6,
        patch_lr_scale=1.0,
        patch_loss_scale=1.0,
    )
    for key in old.keys() | new.keys():
        if key in operational | additions:
            continue
        if old.get(key, defaults.get(key)) != new.get(key, defaults.get(key)):
            raise ValueError("Unexpected scientific argument change: " + key)


def prepare(root):
    paths = json.loads((root / "paths.json").read_text())
    sources = json.loads((root / "cooldown-sources.json").read_text())
    cooldown = paths["cooldown"]
    if cooldown != dict(start=3000, end=4000, final_ratio=0.01):
        raise ValueError("Unexpected cooldown protocol")
    receipts = []
    for arm in sources:
        probe = root / ("probe-" + arm)
        q = json.loads((probe / "JOINT_QUALIFIED.json").read_text())
        if (
            not q["resume_verified"]
            or not q["resume_evidence"]["deterministic_reference_bitwise_exact"]
        ):
            raise ValueError("Replay qualification incomplete")
        if q["contract"]["code_commit"] != paths["producer"] or q["contract"][
            "cooldown"
        ] != dict(cooldown, axis="one-based global optimizer update"):
            raise ValueError("Qualification producer or schedule mismatch")
        io = json.loads((probe / "IO_EQUIVALENT.json").read_text())
        if (
            not io["cpu_arrays_exact"]
            or not io["observation_prepared_and_coverage_exact"]
        ):
            raise ValueError("I/O qualification incomplete")
        if (
            json.loads((probe / "THROUGHPUT.json").read_text())[
                "training_hours_for_4000_updates"
            ]
            > 20
        ):
            raise ValueError("Throughput gate failed")
    for arm, source in sources.items():
        path = Path(source["checkpoint"])
        if digest(path) != source["sha256"]:
            raise ValueError("Source checkpoint changed")
        destination = root / arm
        if destination.exists():
            raise FileExistsError(
                "Refuse to overwrite cooldown fork: " + str(destination)
            )
        original = torch.load(path, map_location="cpu", weights_only=False)
        if original["global_step"] != 2667 or original["task_counts"] != TaskSchedule(
            2000, 2000, "mixed"
        ).counts(2667):
            raise ValueError("Wrong fork point or task counts")
        manifest = json.loads((root / ("probe-" + arm) / "manifest.json").read_text())
        for key in original["manifest"]:
            if key not in {"arguments", "code_commit", "runtime_extension_sha256"}:
                assert_exact_state(original["manifest"][key], manifest[key])
        arguments = manifest["arguments"]
        arguments.update(
            output=str(destination),
            name="extent-" + arm,
            joint_qualification=str(root / ("probe-" + arm) / "JOINT_QUALIFIED.json"),
            om4_updates=2000,
            joint_steps=2000,
            joint_probe=False,
            validate_every=100,
            joint_hours=paths.get("joint_hours", 24),
            milestone_steps=[500, 1000, 1500, 2000],
        )
        audit_arguments(original["manifest"]["arguments"], arguments)
        if arguments["recent_only"] != (arm == "U-global"):
            raise ValueError("Incorrect control routing")
        destination.mkdir()
        migrated = dict(original, manifest=copy.deepcopy(manifest))
        atomic_torch(migrated, destination / "joint-last.pt")
        restored = torch.load(
            destination / "joint-last.pt", map_location="cpu", weights_only=False
        )
        assert_exact_state(restored, migrated)
        for key in original:
            if key != "manifest":
                assert_exact_state(original[key], restored[key])
        if digest(path) != source["sha256"]:
            raise ValueError("Source checkpoint mutated during migration")
        atomic_json(manifest, destination / "manifest.json")
        prefix = [
            json.loads(line) for line in Path(source["events"]).read_text().splitlines()
        ]
        prefix = [
            e
            for e in prefix
            if e.get("event") in {"joint_train", "joint_validation"}
            and e["global_step"] <= 2667
        ]
        if [e["global_step"] for e in prefix if e["event"] == "joint_train"] != list(
            range(1, 2668)
        ):
            raise ValueError("Incomplete source training history")
        (destination / "events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in prefix)
        )
        record = dict(
            arm=arm,
            source=source,
            source_producer=original["manifest"]["code_commit"],
            producer=paths["producer"],
            global_step=2667,
            task_counts=original["task_counts"],
            destination_sha256=digest(destination / "joint-last.pt"),
            all_other_checkpoint_state_exact=True,
            scientific_change="Linear LR cooldown at global updates 3001-4000, ending at 1% of each task rate",
            selection="Reevaluate fork checkpoint; select only checkpoints on the fork, without importing later original-run best weights",
            old_arguments=original["manifest"]["arguments"],
            new_arguments=arguments,
        )
        atomic_json(record, destination / "MIGRATION.json")
        receipts.append(record)
        del original, migrated, restored
    atomic_json(dict(migrations=receipts), root / "MIGRATION_COMPLETE.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    prepare(parser.parse_args().root)
