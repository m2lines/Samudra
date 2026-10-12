# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Evaluate one immutable scratch checkpoint with the established three probes.

The existing evaluators retain their input checks and strict model loading. This
wrapper adds explicit exposure/architecture checks and restart-safe orchestration.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from samudra.experiments.diffusion_direct import checkpoint_processor_prediction
from samudra.experiments.diffusion_global_annual import ORIGINS
from samudra.experiments.observation_pilot import atomic_json, digest

VALIDATION_ORIGINS = ["2013-11", "2013-12", *[f"2014-{m:02d}" for m in range(1, 8)]]


def checkpoint_signature(saved, checkpoint, prediction, om4, observation):
    counts = {"om4": om4, "observation": observation}
    if min(counts.values()) < 1:
        raise ValueError("Both expected task counts must be positive")
    if saved["counts"] != counts or saved["step"] != sum(counts.values()):
        raise ValueError("Checkpoint does not match requested task exposure")
    if checkpoint_processor_prediction(saved) != prediction:
        raise ValueError("Checkpoint processor prediction does not match requested arm")
    return dict(
        checkpoint_sha256=digest(checkpoint),
        counts=counts,
        step=saved["step"],
        processor_prediction=prediction,
        training_contract=saved["contract"],
        members=8,
        sampling_steps=32,
    )


def commands(root, checkpoint, output, signature):
    common = [
        "--data",
        str(root / "data/observations"),
        "--checkpoint",
        str(checkpoint),
        "--reference",
        str(root / "checkpoints/selection-reference.json"),
        "--steps",
        "32",
    ]
    return {
        "validation": [
            "-m",
            "samudra.experiments.diffusion_global_evaluate",
            *common,
            "--split",
            "validation",
            "--output",
            str(output / "validation"),
        ],
        "native": [
            "-m",
            "samudra.experiments.diffusion_intervention_native",
            "--root",
            str(root),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output / "native"),
        ],
        "annual": [
            "-m",
            "samudra.experiments.diffusion_global_annual",
            *common,
            "--annual-data",
            str(root / "data/annual_observations"),
            "--expected-observation-updates",
            str(signature["counts"]["observation"]),
            "--expected-om4-updates",
            str(signature["counts"]["om4"]),
            "--output",
            str(output / "annual"),
        ],
    }


def verify_output(output, section, signature):
    """Reject stale/corrupt outputs before either skipping or marking completion."""
    complete = output / section / "COMPLETE.json"
    if not complete.exists():
        return False
    receipt = json.loads(complete.read_text())
    protocol = receipt["inputs"] if section == "annual" else receipt
    for key in ("checkpoint_sha256", "counts", "sampling_steps"):
        if protocol[key] != signature[key]:
            raise ValueError(f"Different {key} in {section} output")
    if section == "validation":
        if (
            receipt["split"] != "validation"
            or receipt["metrics"]["origins"] != VALIDATION_ORIGINS
        ):
            raise ValueError("Changed validation cohort")
    elif protocol["members"] != 8:
        raise ValueError("Changed ensemble size")
    if section == "native" and protocol["leads_days"] != [0, 30, 90, 180, 365]:
        raise ValueError("Changed native diagnostic leads")
    if section == "annual":
        if sorted(receipt["origins"]) != ORIGINS:
            raise ValueError("Changed annual cohort")
        member_receipt = output / section / "MEMBERS_COMPLETE.json"
        if not member_receipt.exists():
            return False
        exported = json.loads(member_receipt.read_text())
        if exported["inputs"] != protocol:
            raise ValueError("Annual member protocol differs")
    else:
        exported = receipt
    if not exported["files"]:
        raise ValueError("Missing member payloads")
    for filename, checksum in exported["files"].items():
        if (
            Path(filename).name != filename
            or digest(output / section / filename) != checksum
        ):
            raise ValueError("Evaluation payload hash/path mismatch")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--processor-prediction", choices=("direct", "residual"), required=True
    )
    parser.add_argument("--expected-om4-updates", type=int, required=True)
    parser.add_argument("--expected-observation-updates", type=int, required=True)
    parser.add_argument(
        "--gpus", default="0,1,2", help="Three distinct visible GPU identifiers"
    )
    args = parser.parse_args()
    gpu_ids = args.gpus.split(",")
    if len(gpu_ids) != 3 or len(set(gpu_ids)) != 3 or not all(gpu_ids):
        raise ValueError("Supply three distinct GPUs for the three simultaneous probes")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    signature = checkpoint_signature(
        saved,
        args.checkpoint,
        args.processor_prediction,
        args.expected_om4_updates,
        args.expected_observation_updates,
    )
    del saved
    signature["evaluator"] = os.environ["SAMUDRA_CODE_COMMIT"]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "input.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise ValueError(
                "Existing output belongs to a different checkpoint/protocol"
            )
    elif any(args.output.iterdir()):
        raise ValueError("Nonempty evaluation directory without wrapper provenance")
    atomic_json(signature, manifest)
    planned = commands(args.root, args.checkpoint, args.output, signature)

    def run(item):
        gpu, (section, command) = item
        if not verify_output(args.output, section, signature):
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpu,
                RANK="0",
                LOCAL_RANK="0",
                WORLD_SIZE="1",
            )
            with (args.output / f"{section}.log").open("a") as log:
                subprocess.run(
                    [sys.executable, *command],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            if not verify_output(args.output, section, signature):
                raise RuntimeError(f"Missing {section} completion receipt")
        return section

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        sections = list(pool.map(run, zip(gpu_ids, planned.items(), strict=True)))
    if digest(args.checkpoint) != signature["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    atomic_json(
        dict(
            inputs=signature,
            receipts={
                section: digest(args.output / section / "COMPLETE.json")
                for section in sections
            },
        ),
        args.output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
