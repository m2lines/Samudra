#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""One matched global observation-only arm with immutable 8k/16k evaluations."""

import argparse
import hashlib
import json
from pathlib import Path

import submit_observation_missingness as wave  # type: ignore[import-not-found]
import submit_observation_pilot as pilot  # type: ignore[import-not-found]

PRODUCER = "79025a163817a577ab81af95b401f5cb0563cd12"  # pragma: allowlist secret
MILESTONES = [
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2000,
    4000,
    6000,
    8000,
    10000,
    12000,
    14000,
    16000,
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", default=PRODUCER, choices=[PRODUCER])
    parser.add_argument(
        "--root", default="/scratch/jr7309/runs/2026-09-30-observation-global-scratch"
    )
    parser.add_argument(
        "--qualification-root",
        default="/scratch/jr7309/runs/2026-09-29-observation-global",
    )
    parser.add_argument("--gpu-type", choices=["rtx6000"], default="rtx6000")
    parser.add_argument("--memory-gb", type=int, default=96)
    parser.add_argument("--deadline", default="2026-10-03T02:00:00+00:00")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root, qualified_root = Path(args.root), Path(args.qualification_root)
    root.mkdir(parents=True, exist_ok=True)
    args.reference = str(qualified_root / "fit-conditioned/selection-reference.json")
    fit_path = qualified_root / "fit-conditioned/QUALIFIED.json"
    probe_path = qualified_root / "probe-conditioned/JOINT_QUALIFIED.json"
    fit = json.loads(fit_path.read_text())
    probe = json.loads(probe_path.read_text())
    if (
        fit["code_commit"] != args.code_commit
        or probe["contract"]["code_commit"] != args.code_commit
        or not probe["resume_verified"]
        or not fit["global_observations"]
        or not probe["contract"]["global_observations"]
        or fit["model_options"]
        != {
            "initializer_architecture": "unet",
            "surface_policy": "observed-only",
            "task_conditioning": "input-adapters",
        }
        or fit["selection_reference_sha256"] != digest(args.reference)
        or probe["contract"]["selection_reference_sha256"] != digest(args.reference)
        or fit["data_manifest_sha256"]
        != digest("/scratch/jr7309/data/obs-d-pilot/SHA256SUMS")
    ):
        raise ValueError(
            "Existing fitting/resume qualification does not cover this producer/config/data"
        )
    name = "conditioned-scratch-global"
    train_args = wave.common(args, name, "conditioned") + [
        "--global-observations",
        "--qualification",
        str(fit_path),
        "--joint-qualification",
        str(probe_path),
        "--ordering",
        "scratch",
        "--om4-updates",
        "0",
        "--joint-steps",
        "16000",
        "--observation-finish",
        "0",
        "--om4-lr",
        "0.0001",
        "--om4-validation-origins",
        "12",
        "--deadline",
        args.deadline,
        "--validate-every",
        "100",
        "--joint-hours",
        "23.5",
        "--milestone-steps",
        *map(str, MILESTONES),
    ]
    evaluations = {}
    for updates in [8000, 16000]:
        for kind, module, extra in [
            ("monthly", "observation_evaluate", ["--selected-only"]),
            (
                "annual",
                "observation_annual",
                ["--annual-data", "/scratch/jr7309/data/obs-d-annual-instance-v1"],
            ),
        ]:
            label = f"obs{updates:05d}-{kind}"
            evaluations[label] = {
                "module": "samudra.experiments." + module,
                "hours": 2 if kind == "monthly" else 1,
                "args": [
                    "--",
                    "--run",
                    str(root / name),
                    "--output",
                    str(root / label),
                    "--split",
                    "test",
                    "--global-observations",
                    "--checkpoint",
                    f"joint-{updates:05d}.pt",
                    "--fixed-om4-updates",
                    "0",
                    "--fixed-observation-updates",
                    str(updates),
                    *extra,
                ],
            }
    plan = {
        "producer": args.code_commit,
        "model_name": "Small conditioned scratch global",
        "run": str(root / name),
        "updates": {"om4": 0, "observation": 16000},
        "milestone_observation_updates": [0, *MILESTONES],
        "train_args": train_args,
        "evaluations": evaluations,
        "qualification": {
            "fitting": str(fit_path),
            "resume": str(probe_path),
            "fitting_sha256": digest(fit_path),
            "resume_sha256": digest(probe_path),
        },
        "selection_reference_sha256": digest(args.reference),
        "submission_source_sha256": digest(__file__),
        "helper_sources_sha256": {
            module.__name__: digest(module.__file__) for module in [wave, pilot]
        },
        "requested_gpu_hour_caps": {"training": 24, "evaluations": 6},
        "primary_comparisons": {
            "8000": "same observation exposure",
            "16000": "same total optimizer updates",
        },
        "evaluation_gate": "All held-out jobs follow successful full 16k training; runtime also requires TRAIN_COMPLETE.json.",
        "qualification_reuse": "Same immutable producer, global support, model and optimizer contract. Probe weights are never loaded into production.",
    }
    (root / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    jobs = {
        "train": pilot.submit(
            args,
            name,
            "samudra.experiments.observation_joint",
            train_args,
            24,
            ["18811858", "18811859"],
        )
    }
    for label, item in evaluations.items():
        jobs[label] = pilot.submit(
            args, label, item["module"], item["args"], item["hours"], [jobs["train"]]
        )
    (root / "DAG.json").write_text(
        json.dumps(
            {"producer": args.code_commit, "jobs": jobs, "plan": "PLAN.json"}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
