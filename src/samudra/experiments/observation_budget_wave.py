# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pinned one-seed update-allocation wave; each stage owns one GPU allocation."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_pilot import atomic_json, digest

SOURCE = Path("/scratch/jr7309/runs/2026-09-22-initializer-wave3-primary/D/best.pt")
CONTRACT = Path(
    "/scratch/jr7309/runs/2026-09-22-observation-D/qualification-h200/QUALIFIED.json"
)
REFERENCE = Path(
    "/scratch/jr7309/runs/2026-09-22-observation-D/fitting/selection-reference.json"
)
DATA = "/scratch/jr7309/data/obs-d-pilot"


def child(module, arguments, output, marker):
    """Resume only the exact same command; a failed child stops dependent work."""
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", module, *map(str, arguments)]
    signature = {"command": command, "producer": os.environ.get("SAMUDRA_CODE_COMMIT")}
    signature_path = output / "stage-command.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError(f"Changed command for {output}")
    atomic_json(signature, signature_path)
    if (output / marker).exists():
        return
    subprocess.run(command, check=True)
    if not (output / marker).exists():
        raise RuntimeError(f"Child did not complete: {output}")


def observation(
    root,
    name,
    checkpoint,
    rate,
    steps,
    *,
    scratch=False,
    probe=False,
    qualification=None,
    interval=50,
    milestones=(),
):
    output = root / name
    contract = {
        "checkpoint_sha256": digest(checkpoint),
        "parent_checkpoint_sha256": json.loads(CONTRACT.read_text())[
            "checkpoint_sha256"
        ],
        "scope": "Strict-loaded source; this run separately verifies fitting and gradients",
    }
    contract_path = root / (name + "-source-contract.json")
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Source changed")
    atomic_json(contract, contract_path)
    args = [
        "--data",
        DATA,
        "--checkpoint",
        checkpoint,
        "--source-contract",
        contract_path,
        "--selection-reference",
        REFERENCE,
        "--output",
        output,
        "--name",
        root.name + "-" + name,
        "--adapter-steps",
        0,
        "--reconstruction-steps",
        1000,
        "--reconstruction-hours",
        6,
        "--joint-steps",
        steps,
        "--joint-hours",
        40,
        "--core-lr",
        rate,
        "--warmup-steps",
        50,
        "--fixed-updates",
        "--validate-every",
        interval,
    ]
    if scratch:
        args += ["--from-scratch"]
    if probe:
        args += ["--fit-probe", "--wandb-mode", "disabled"]
    else:
        args += ["--qualification", qualification]
    if milestones:
        args += ["--milestone-steps", *milestones]
    child(
        "samudra.experiments.observation_pilot",
        args,
        output,
        "QUALIFIED.json" if probe else "TRAIN_COMPLETE.json",
    )
    return output


def om4(root, name, rate, steps, milestones=()):
    output = root / name
    args = [
        "--arm",
        "D",
        "--phase",
        "joint",
        "--initial-checkpoint",
        SOURCE,
        "--output",
        output,
        "--name",
        root.name + "-" + name,
        "--max-steps",
        steps,
        "--hours",
        3,
        "--deadline",
        "2026-10-02T00:00:00Z",
        "--batch-size",
        2,
        "--accumulate",
        4,
        "--learning-rate",
        rate,
        "--warmup-steps",
        50,
        "--patience",
        100000,
        "--validation-seconds",
        300,
        "--train-only",
    ]
    if milestones:
        args += ["--milestone-steps", *milestones]
    child("samudra.experiments.initializer_wave", args, output, "TRAIN_COMPLETE.json")
    complete = json.loads((output / "TRAIN_COMPLETE.json").read_text())
    if complete["state"]["step"] != steps:
        raise ValueError("OM4 did not complete its fixed budget")
    return output


def calibrate(root):
    selected = {}
    scores = []
    for index, rate in enumerate((3e-5, 1e-4)):
        output = om4(root, f"om4-lr{index}", rate, 200)
        result = json.loads((output / "TRAIN_COMPLETE.json").read_text())
        scores.append(
            {
                "rate": rate,
                "score": result["selected_validation"]["ts_mse"],
                "run": str(output),
            }
        )
    selected["om4"] = {
        "rate": min(scores, key=lambda x: x["score"])["rate"],
        "trials": scores,
    }
    for scratch, label, rates in [
        (False, "transfer", (1e-5, 3e-5)),
        (True, "scratch", (3e-5, 1e-4)),
    ]:
        fitting = observation(
            root, label + "-fit", SOURCE, rates[0], 200, scratch=scratch, probe=True
        )
        scores = []
        for index, rate in enumerate(rates):
            output = observation(
                root,
                f"{label}-lr{index}",
                SOURCE,
                rate,
                200,
                scratch=scratch,
                qualification=fitting / "QUALIFIED.json",
            )
            reference = json.loads(REFERENCE.read_text())
            values = [
                selection_score(
                    json.loads(p.read_text()),
                    reference["control"],
                    reference["spectral_keys"],
                )
                for p in output.glob("joint-validation-*.json")
            ]
            result = {"score": min(values)}
            scores.append({"rate": rate, "score": result["score"], "run": str(output)})
        selected[label] = {
            "rate": min(scores, key=lambda x: x["score"])["rate"],
            "trials": scores,
        }
    atomic_json(selected, root / "calibration.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--stage",
        choices=["calibration", "om4", "obs0", "obs25", "obs50", "obs75", "random"],
        required=True,
    )
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if digest(SOURCE) != json.loads(CONTRACT.read_text())["checkpoint_sha256"]:
        raise ValueError("Original D checkpoint no longer matches its contract")
    if args.stage == "calibration":
        calibrate(root)
        return
    selected = json.loads((root / "calibration.json").read_text())
    if args.stage == "om4":
        om4(root, "om4-prefix", selected["om4"]["rate"], 3000, [1000, 2000, 3000])
        return
    om4_steps, obs_steps = {
        "obs0": (0, 4000),
        "obs25": (1000, 3000),
        "obs50": (2000, 2000),
        "obs75": (3000, 1000),
        "random": (0, 8000),
    }[args.stage]
    scratch = args.stage == "random"
    checkpoint = (
        root / "om4-prefix" / f"step-{om4_steps:05d}.pt" if om4_steps else SOURCE
    )
    rate = selected["scratch" if scratch else "transfer"]["rate"]
    fit = observation(
        root, args.stage + "-fit", checkpoint, rate, 200, scratch=scratch, probe=True
    )
    interval = 1000 if scratch else obs_steps // 4
    milestones = [4000, 6000, 8000] if scratch else [obs_steps]
    output = observation(
        root,
        args.stage,
        checkpoint,
        rate,
        obs_steps,
        scratch=scratch,
        qualification=fit / "QUALIFIED.json",
        interval=interval,
        milestones=milestones,
    )
    atomic_json(
        {
            "run": str(output),
            "om4_updates": om4_steps,
            "observation_updates": obs_steps,
            "seed": 1729,
            "core_lr": rate,
            "producer": os.environ.get("SAMUDRA_CODE_COMMIT"),
        },
        root / (args.stage + "-complete.json"),
    )

    # Freeze scratch's matched-budget best before evaluating its later extension.
    evaluation_runs = [output]
    if scratch:
        prefix = root / "random-4000"
        prefix.mkdir(exist_ok=True)
        for source, dest in [
            ("manifest.json", "manifest.json"),
            ("joint-04000-best.pt", "best.pt"),
            ("joint-04000-best.json", "best.json"),
        ]:
            target = prefix / dest
            if not target.exists():
                shutil.copyfile(output / source, target)
        atomic_json(
            {"scope": "Selection frozen at joint update 4000", "step": 4000},
            prefix / "TRAIN_COMPLETE.json",
        )
        evaluation_runs.insert(0, prefix)
    for run in evaluation_runs:
        evaluation = root / (run.name + "-evaluation")
        child(
            "samudra.experiments.observation_evaluate",
            ["--run", run, "--output", evaluation],
            evaluation,
            "COMPLETE.json",
        )


if __name__ == "__main__":
    main()
