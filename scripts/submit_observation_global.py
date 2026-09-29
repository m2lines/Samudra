#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched global-observation rerun of Small conditioned mixed and control rescoring."""

import argparse
import json
from pathlib import Path

import submit_observation_missingness as wave  # type: ignore[import-not-found]
import submit_observation_pilot as pilot  # type: ignore[import-not-found]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--root", default="/scratch/jr7309/runs/2026-09-29-observation-global"
    )
    parser.add_argument("--gpu-type", default="rtx6000", choices=["rtx6000", "h200"])
    parser.add_argument("--memory-gb", type=int, default=96)
    parser.add_argument("--deadline", default="2026-10-02T20:00:00+00:00")
    parser.add_argument("--latent-channels", type=int, default=0)
    parser.add_argument("--skip-control-evaluation", action="store_true")
    parser.add_argument("--shared-reference")
    parser.add_argument("--reference-job")
    args = parser.parse_args()
    if args.latent_channels < 0:
        parser.error("Latent channel count must be nonnegative")
    if bool(args.shared_reference) != bool(args.reference_job):
        parser.error("Shared reference requires its producing job dependency")
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    args.reference = str(root / "fit-conditioned" / "selection-reference.json")

    def common(name, fit=False):
        result = wave.common(args, name, "conditioned") + ["--global-observations"]
        if args.latent_channels:
            result += ["--latent-channels", str(args.latent_channels)]
        if fit:
            i = result.index("--selection-reference")
            if args.shared_reference:
                result[i + 1] = args.shared_reference
            else:
                del result[i : i + 2]
        return result

    jobs = {}
    jobs["fit"] = pilot.submit(
        args,
        "fit-conditioned",
        "samudra.experiments.observation_pilot",
        common("fit-conditioned", True) + ["--fit-probe"],
        1,
        [args.reference_job] if args.reference_job else [],
    )
    joint = [
        "--qualification",
        str(root / "fit-conditioned" / "QUALIFIED.json"),
        "--ordering",
        "mixed",
        "--om4-lr",
        "0.0001",
        "--om4-validation-origins",
        "12",
        "--deadline",
        args.deadline,
    ]
    jobs["probe"] = pilot.submit(
        args,
        "probe-conditioned",
        "samudra.experiments.observation_joint",
        common("probe-conditioned")
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
            "0.9",
        ],
        1,
        [jobs["fit"]],
    )
    name = "conditioned-mixed-global" + (
        f"-latent{args.latent_channels}" if args.latent_channels else ""
    )
    jobs["train"] = pilot.submit(
        args,
        name,
        "samudra.experiments.observation_joint",
        common(name)
        + joint
        + [
            "--joint-qualification",
            str(root / "probe-conditioned" / "JOINT_QUALIFIED.json"),
            "--om4-updates",
            "8000",
            "--joint-steps",
            "8000",
            "--observation-finish",
            "0",
            "--validate-every",
            "100",
            "--joint-hours",
            "15.5",
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
        ],
        16,
        [jobs["probe"]],
    )
    control = Path(
        "/scratch/jr7309/runs/2026-09-28-observation-missingness/conditioned-mixed"
    )
    for label, run, deps in [
        ("control", control, []),
        ("global", root / name, [jobs["train"]]),
    ]:
        if label == "control" and args.skip_control_evaluation:
            continue
        # Fixed endpoints are the primary causal comparison; selected checkpoints
        # retain their original selection provenance and are secondary diagnostics.
        for fixed in [True, False]:
            suffix = "endpoint" if fixed else "selected"
            checkpoint = (
                [
                    "--checkpoint",
                    "joint-08000.pt",
                    "--fixed-om4-updates",
                    "8000",
                    "--fixed-observation-updates",
                    "8000",
                ]
                if fixed
                else []
            )
            for kind, module, extra in [
                ("monthly", "observation_evaluate", ["--selected-only"]),
                (
                    "annual",
                    "observation_annual",
                    ["--annual-data", "/scratch/jr7309/data/obs-d-annual-instance-v1"],
                ),
            ]:
                key = f"{label}-{suffix}-{kind}"
                jobs[key] = pilot.submit(
                    args,
                    key,
                    "samudra.experiments." + module,
                    [
                        "--",
                        "--run",
                        str(run),
                        "--output",
                        str(root / key),
                        "--split",
                        "test",
                        "--global-observations",
                        *checkpoint,
                        *extra,
                    ],
                    2 if kind == "monthly" else 1,
                    deps,
                )
    (root / "DAG.json").write_text(
        json.dumps(
            {
                "producer": args.code_commit,
                "jobs": jobs,
                "control": str(control),
                "primary": "matched 8000/8000 endpoint",
                "latent_channels": args.latent_channels,
                "shared_reference": args.shared_reference,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
