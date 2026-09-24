#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Clone a completed scratch 8k run for an audited, optimizer-preserving 16k continuation."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import torch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to modify existing continuation: {output}")
    manifest = read(source / "manifest.json")
    args = manifest["arguments"]
    if not (
        args["from_scratch"]
        and args["fixed_updates"]
        and args["joint_steps"] == 8000
        and args["seed"] == 1729
        and args["accumulate"] == 8
    ):
        raise ValueError("Not the authorized one-seed scratch 8k run")
    if read(source / "joint-complete.json")["step"] != 8000:
        raise ValueError("The 8k phase must finish first")
    best = read(source / "best.json")
    if digest(source / "best.pt") != best["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash mismatch")
    if (
        read(source / "TRAIN_COMPLETE.json")["best_checkpoint_sha256"]
        != best["checkpoint_sha256"]
    ):
        raise ValueError("Training completion disagrees with selected checkpoint")
    evaluation = read(source.parent / (source.name + "-evaluation") / "COMPLETE.json")
    if (
        evaluation["origins"] != 96
        or evaluation["selected_sha256"] != best["checkpoint_sha256"]
    ):
        raise ValueError("Preserve the completed 8k held-out evaluation first")
    milestones = {}
    for step in (4000, 6000, 8000):
        prefix = source / f"joint-{step:05d}"
        metadata = read(str(prefix) + "-best.json")
        selected_hash = digest(str(prefix) + "-best.pt")
        if metadata["step"] > step or metadata["checkpoint_sha256"] != selected_hash:
            raise ValueError(f"Invalid frozen milestone {step}")
        milestones[str(step)] = {
            "terminal_sha256": digest(prefix.with_suffix(".pt")),
            "selected_sha256": selected_hash,
        }
    last = torch.load(source / "joint-last.pt", map_location="cpu", weights_only=False)
    terminal = torch.load(
        source / "joint-08000.pt", map_location="cpu", weights_only=False
    )
    if last["state"]["step"] != 8000 or terminal["step"] != 8000:
        raise ValueError("Resume must start from terminal update 8000")
    if not all(key in last for key in ("model", "optimizer", "torch_rng", "cuda_rng")):
        raise ValueError("Incomplete optimizer/RNG resume state")
    if last["model"].keys() != terminal["model"].keys() or not all(
        torch.equal(value, terminal["model"][key])
        for key, value in last["model"].items()
    ):
        raise ValueError(
            "Resume weights differ from the immutable terminal 8k snapshot"
        )
    if not last["optimizer"]["state"]:
        raise ValueError("Optimizer moments are missing")
    del terminal, last
    continued = copy.deepcopy(manifest)
    changes = {
        "output": str(output),
        "name": args["name"] + "-continue-16k",
        "joint_steps": 16000,
        "milestone_steps": [4000, 6000, 8000, 12000, 16000],
    }
    continued["arguments"].update(changes)
    files = [
        "best.pt",
        "best.json",
        "joint-last.pt",
        "joint-best.pt",
        "reconstruction-best.pt",
        "reconstruction-complete.json",
        "selection-reference.json",
        "baseline-validation.json",
        "events.jsonl",
    ]
    staging = output.with_name(output.name + ".staging")
    staging.mkdir(exist_ok=False)
    hashes = {}
    for name in files:
        hashes[name] = digest(source / name)
        shutil.copyfile(source / name, staging / name)
        if digest(staging / name) != hashes[name]:
            raise ValueError(f"Copy verification failed: {name}")
    (staging / "manifest.json").write_text(json.dumps(continued, indent=2) + "\n")
    record = {
        "source": str(source),
        "source_manifest_sha256": digest(source / "manifest.json"),
        "producer": manifest["code_commit"],
        "resume_step": 8000,
        "target_step": 16000,
        "argument_changes": {k: {"from": args[k], "to": v} for k, v in changes.items()},
        "copied_files_sha256": hashes,
        "preserved_milestones": milestones,
        "script_sha256": digest(Path(__file__)),
        "resume_contract": "Exact terminal model, AdamW state and CPU/CUDA RNG; unchanged producer, data, normalization, LR, seed, batch, warmup, validation cadence and qualification. Completed reconstruction reused. Original 8k directory remains untouched.",
    }
    (staging / "CONTINUATION_READY.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    staging.rename(output)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output)), flush=True)


if __name__ == "__main__":
    main()
