#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage a bounded missingness wave; production requires real qualification markers."""

import argparse
import json
from pathlib import Path

import submit_observation_pilot as pilot_submit  # type: ignore[import-not-found]

VARIANTS = {
    "legacy": ["--surface-policy", "legacy-copy", "--surface-fill", "climatology"],
    "masked": [
        "--surface-policy",
        "observed-only",
        "--surface-fill",
        "zero",
        "--completion-weight",
        "0.1",
    ],
    "conditioned": [
        "--surface-policy",
        "observed-only",
        "--surface-fill",
        "zero",
        "--completion-weight",
        "0.1",
        "--task-conditioning",
        "input-adapters",
    ],
}
ARMS = {
    "legacy-scratch": ("legacy", "scratch", 0, 16000, 0),
    "masked-scratch": ("masked", "scratch", 0, 16000, 0),
    "masked-sequential": ("masked", "sequential", 8000, 8000, 0),
    "masked-mixed-finish": ("masked", "mixed-finish", 8000, 8000, 2000),
    "conditioned-mixed-finish": ("conditioned", "mixed-finish", 8000, 8000, 2000),
    "conditioned-mixed": ("conditioned", "mixed", 8000, 8000, 0),
}


def common(args, name, variant):
    return [
        "--",
        "--data",
        "/scratch/jr7309/data/obs-d-pilot",
        "--checkpoint",
        "unused-fresh",
        "--source-contract",
        "unused-fresh",
        "--output",
        str(Path(args.root) / name),
        "--name",
        "missingness-" + name,
        "--normalization",
        "instance",
        "--initializer-architecture",
        "unet",
        "--evolution-architecture",
        "d",
        "--from-scratch",
        "--observation-normalization",
        "--strict-velocity-support",
        "--core-lr",
        "0.0001",
        "--accumulate",
        "8",
        "--seed",
        "1729",
        "--selection-reference",
        args.reference,
        "--reconstruction-weight",
        "0.1",
        "--warmup-steps",
        "0",
    ] + VARIANTS[variant]


def qualification(args):
    jobs = {}
    for variant in VARIANTS:
        fit_name = "fit-" + variant
        fit = pilot_submit.submit(
            args,
            fit_name,
            "samudra.experiments.observation_pilot",
            common(args, fit_name, variant) + ["--fit-probe"],
            1,
        )
        probe_name = "probe-" + variant
        probe = pilot_submit.submit(
            args,
            probe_name,
            "samudra.experiments.observation_joint",
            common(args, probe_name, variant)
            + [
                "--qualification",
                str(Path(args.root) / fit_name / "QUALIFIED.json"),
                "--ordering",
                "mixed",
                "--om4-updates",
                "6",
                "--joint-steps",
                "4",
                "--om4-lr",
                "0.0001",
                "--om4-validation-origins",
                "12",
                "--deadline",
                args.deadline,
                "--joint-probe",
                "--validate-every",
                "1",
                "--joint-hours",
                "0.9",
            ],
            1,
            [fit],
        )
        jobs[variant] = {"fit": fit, "probe": probe}
    return jobs


def production(args):
    jobs = {}
    for variant in VARIANTS:
        fit = json.loads(
            (Path(args.root) / ("fit-" + variant) / "QUALIFIED.json").read_text()
        )
        probe = json.loads(
            (
                Path(args.root) / ("probe-" + variant) / "JOINT_QUALIFIED.json"
            ).read_text()
        )
        if (
            fit["code_commit"] != args.code_commit
            or probe["contract"]["code_commit"] != args.code_commit
            or not probe["resume_verified"]
        ):
            raise ValueError(
                "Production producer must match completed fitting and resume qualifications"
            )
    for name, (variant, ordering, om4, observation, finish) in ARMS.items():
        command = common(args, name, variant) + [
            "--qualification",
            str(Path(args.root) / ("fit-" + variant) / "QUALIFIED.json"),
            "--joint-qualification",
            str(Path(args.root) / ("probe-" + variant) / "JOINT_QUALIFIED.json"),
            "--ordering",
            ordering,
            "--om4-updates",
            str(om4),
            "--joint-steps",
            str(observation),
            "--observation-finish",
            str(finish),
            "--om4-lr",
            "0.0001",
            "--om4-validation-origins",
            "12",
            "--deadline",
            args.deadline,
            "--validate-every",
            "100",
            "--joint-hours",
            "11.5",
            "--milestone-steps",
            "10",
            "25",
            "50",
            "100",
            "250",
            "500",
            "1000",
            "2000",
            "4000",
            "6000",
            "6100",
            "6500",
            "7000",
            "8000",
            "12000",
            "16000",
        ]
        jobs[name] = pilot_submit.submit(
            args, name, "samudra.experiments.observation_joint", command, 12
        )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["qualification", "production"])
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--root", default="/scratch/jr7309/runs/2026-09-28-observation-missingness"
    )
    parser.add_argument(
        "--reference",
        default="/scratch/jr7309/runs/2026-09-25-observation-instance/observation-qualification-v3/selection-reference.json",
    )
    parser.add_argument("--deadline", default="2026-10-01T18:00:00+00:00")
    parser.add_argument("--gpu-type", choices=["rtx6000", "h200"], default="rtx6000")
    parser.add_argument("--memory-gb", type=int, default=96)
    args = parser.parse_args()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    jobs = qualification(args) if args.stage == "qualification" else production(args)
    path = Path(args.root) / (args.stage.upper() + "_DAG.json")
    record = {
        "producer": args.code_commit,
        "jobs": jobs,
        "operational_gpu_hour_cap": 100,
    }
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("DAG differs from existing record")
    path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
