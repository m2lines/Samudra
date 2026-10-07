#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired early-data and fine encoder/decoder screen; fit, resume qualification, then train and held-out eval."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ARMS = {
    "U-multitask-early": ("unet", 0),
    "U-multitask-early-latent": ("unet", 10),
    "U-multitask-early-fine": ("fine", 0),
    "U-multitask-early-fine-latent": ("fine", 10),
}


def command(module, arguments):
    cmd = [sys.executable, "-m", module, *map(str, arguments)]
    print(json.dumps(dict(event="launch", command=cmd)), flush=True)
    child = subprocess.Popen(cmd)
    previous = {}
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: child.send_signal(signal.SIGUSR1))
    try:
        status = child.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if status:
        raise SystemExit(status if status > 0 else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--stage", choices=["fit", "probe", "train"], required=True)
    args = parser.parse_args()
    root = Path(args.root)
    config = json.loads((root / "paths.json").read_text())
    if os.environ["SAMUDRA_CODE_COMMIT"] != config["producer"]:
        raise ValueError("Launch producer differs from frozen paths")
    architecture, latent = ARMS[args.arm]
    fit = root / ("fit-" + architecture + "-" + str(latent))
    probe = root / ("probe-" + args.arm)
    out = (
        fit
        if args.stage == "fit"
        else probe
        if args.stage == "probe"
        else root / args.arm
    )
    common = [
        "--data",
        config["observations"],
        "--checkpoint",
        "unused-fresh",
        "--source-contract",
        "unused-fresh",
        "--output",
        out,
        "--name",
        "extent-" + out.name,
        "--normalization",
        "instance",
        "--initializer-architecture",
        "unet",
        "--evolution-architecture",
        "extent-" + architecture,
        "--latent-channels",
        str(latent),
        "--from-scratch",
        "--observation-normalization",
        "--strict-velocity-support",
        "--global-observations",
        "--surface-policy",
        "observed-only",
        "--surface-fill",
        "zero",
        "--completion-weight",
        "0.1",
        "--task-conditioning",
        "input-adapters",
        "--reconstruction-weight",
        "0.1",
        "--core-lr",
        "0.0001",
        "--accumulate",
        "8",
        "--warmup-steps",
        "0",
        "--seed",
        "1729",
        "--wandb-mode",
        "offline",
        "--selection-reference",
        root / "selection-reference.json",
    ]
    if args.stage == "fit":
        command("samudra.experiments.observation_pilot", common + ["--fit-probe"])
        if not (fit / "QUALIFIED.json").exists():
            raise RuntimeError("Fit exited without qualification")
        return
    joint = [
        "--qualification",
        fit / "QUALIFIED.json",
        "--ordering",
        "mixed",
        "--om4-lr",
        "0.0001",
        "--om4-data",
        config["global_om4"],
        "--om4-validation-origins",
        "12",
        "--deadline",
        "2026-12-31T00:00:00+00:00",
    ]
    joint += ["--early-data", config["early_data"]]
    if args.stage == "probe":
        command(
            "samudra.experiments.observation_joint",
            common
            + joint
            + [
                "--om4-updates",
                "6",
                "--joint-steps",
                "4",
                "--joint-probe",
                "--validate-every",
                "1",
                "--joint-hours",
                "1.5",
            ],
        )
        if not (probe / "JOINT_QUALIFIED.json").exists():
            raise RuntimeError("Resume probe exited without qualification")
        return
    throughput = json.loads((probe / "THROUGHPUT.json").read_text())
    if throughput["training_hours_for_4000_updates"] > config.get(
        "training_hours_gate", 20
    ):
        raise RuntimeError(
            "Measured update throughput cannot meet the one-day screen; revise budget before training"
        )
    if not (out / "TRAIN_COMPLETE.json").exists():
        command(
            "samudra.experiments.observation_joint",
            common
            + joint
            + [
                "--joint-qualification",
                probe / "JOINT_QUALIFIED.json",
                "--om4-updates",
                "2000",
                "--joint-steps",
                "2000",
                "--validate-every",
                "100",
                "--joint-hours",
                str(config.get("joint_hours", 24)),
                "--milestone-steps",
                "500",
                "1000",
                "1500",
                "2000",
            ],
        )
    if not (out / "TRAIN_COMPLETE.json").exists():
        raise RuntimeError(
            "Training stopped at its wall-time cap; checkpoint preserved, budget incomplete"
        )
    if not (out / "test-selected" / "COMPLETE.json").exists():
        command(
            "samudra.experiments.observation_evaluate",
            [
                "--run",
                out,
                "--output",
                out / "test-selected",
                "--selected-only",
                "--global-observations",
            ],
        )


if __name__ == "__main__":
    main()
