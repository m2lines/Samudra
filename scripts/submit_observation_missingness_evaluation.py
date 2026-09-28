#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Queue selected/fixed and initialization audits after qualified production jobs."""

import argparse
import json
from pathlib import Path

import submit_observation_missingness as wave  # type: ignore[import-not-found]
import submit_observation_pilot as pilot  # type: ignore[import-not-found]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="/scratch/jr7309/runs/2026-09-28-observation-missingness"
    )
    parser.add_argument(
        "--code-commit",
        required=True,
        help="Evaluation producer; training remains separately pinned",
    )
    parser.add_argument("--gpu-type", default="h200", choices=["h200", "rtx6000"])
    parser.add_argument("--memory-gb", type=int, default=96)
    args = parser.parse_args()
    root = Path(args.root)
    production = json.loads((root / "PRODUCTION_DAG.json").read_text())
    if set(production["jobs"]) != set(wave.ARMS):
        raise ValueError("Production arm set differs")
    jobs = {}
    for arm, (_, ordering, om4, observation, finish) in wave.ARMS.items():
        run = root / arm
        dependency = [production["jobs"][arm]]
        specifications = [
            (
                "selected-monthly",
                "observation_evaluate",
                ["--selected-only", "--split", "test"],
                2,
            ),
            (
                "selected-annual",
                "observation_annual",
                [
                    "--annual-data",
                    "/scratch/jr7309/data/obs-d-annual-instance-v1",
                    "--split",
                    "test",
                ],
                1,
            ),
            (
                "fixed-endpoint-monthly",
                "observation_evaluate",
                [
                    "--selected-only",
                    "--split",
                    "test",
                    "--checkpoint",
                    f"joint-{observation:05d}.pt",
                    "--fixed-om4-updates",
                    str(om4),
                    "--fixed-observation-updates",
                    str(observation),
                ],
                1,
            ),
        ]
        if ordering == "scratch":
            specifications.append(
                (
                    "fixed8k-monthly",
                    "observation_evaluate",
                    [
                        "--selected-only",
                        "--split",
                        "test",
                        "--checkpoint",
                        "joint-08000.pt",
                        "--fixed-om4-updates",
                        "0",
                        "--fixed-observation-updates",
                        "8000",
                    ],
                    1,
                )
            )
        checkpoints = [
            "joint-00000.pt",
            "joint-00010.pt",
            "joint-00100.pt",
            "joint-01000.pt",
            "joint-04000.pt",
            "joint-08000.pt",
            "best.pt",
        ]
        if ordering == "scratch":
            checkpoints.append("joint-16000.pt")
        if finish:
            checkpoints.extend(
                ["before-observation-finish.pt", "joint-06100.pt", "joint-06500.pt"]
            )
        specifications.extend(
            [
                (
                    "completion",
                    "observation_completion",
                    ["--checkpoints", *checkpoints],
                    1,
                ),
                (
                    "state-diagnostics",
                    "observation_state_diagnostics",
                    ["--checkpoints", *checkpoints],
                    1,
                ),
            ]
        )
        for suffix, module, extra, hours in specifications:
            name = arm + "-" + suffix
            command = ["--", "--run", str(run), "--output", str(root / name), *extra]
            jobs[name] = pilot.submit(
                args, name, "samudra.experiments." + module, command, hours, dependency
            )
    record = {
        "training_producer": production["producer"],
        "evaluation_producer": args.code_commit,
        "jobs": jobs,
        "scope": "Test reports require TRAIN_COMPLETE and selected/fixed lineage. Completion/state audits use the nine fixed validation origins only. Annual outputs are diagnostic, not selection.",
    }
    path = root / "EVALUATION_DAG.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("Existing evaluation DAG differs")
    path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
