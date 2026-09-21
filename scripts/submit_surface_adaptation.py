#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Submit the authorized adaptation stage with at most two four-GPU jobs at once."""

import argparse
import datetime
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

ARMS = {
    "A": ("initializer", 1729),
    "B": ("evolution", 1729),
    "C": ("joint", 1729),
    "D": ("joint", 1730),
    "E": ("joint", 1729),
    "F": ("joint", 1730),
}


def submit(args):
    root = Path(args.root)
    if any(not job.isdigit() for job in args.after_jobs):
        raise ValueError("Dependency IDs must be numeric")
    if args.stage == "rate-control" and not args.after_jobs:
        raise ValueError(
            "Rate control requires dependencies to avoid overlapping the primary wave"
        )
    if (root / "jobs.json").exists():
        raise FileExistsError(
            "Existing submission: inspect and recover specific jobs, do not duplicate this stage"
        )
    deadline = datetime.datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("Deadline must have an explicit UTC offset")
    wall_hours = 0.25 if args.stage == "qualification" else 4.0
    if (
        datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=wall_hours)
        >= deadline
    ):
        raise ValueError("Insufficient time before the study deadline")
    for key in ("GHCR_USERNAME", "GHCR_TOKEN", "WANDB_API_KEY"):
        if not os.environ.get(key):
            raise ValueError(f"Missing remote {key}")
    for path in (
        args.code_layer,
        args.wrapper,
        str(Path(args.wave1_root) / "ar/pretrain-best.pt"),
        str(Path(args.wave1_root) / "ar/joint-best.pt"),
        str(Path(args.wave1_root) / "initializer/initializer-best.pt"),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if args.stage != "qualification":
        if not args.qualification_root:
            raise ValueError(
                "Production requires the completed qualification directory"
            )
        for name in ("A", "B", "C"):
            directory = Path(args.qualification_root) / name
            marker = json.loads((directory / "COMPLETE.json").read_text())
            manifest = json.loads((directory / "manifest.json").read_text())
            if (
                marker["arm"] != ARMS[name][0]
                or manifest["code_commit"] != args.code_commit
                or manifest["arguments"]["max_steps"] != 30
                or manifest["world_size"] != 4
            ):
                raise ValueError(f"Qualification mismatch for {name}")
    root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {
            "CONFIG": "src/samudra/configs/samudra_om4/train.yaml",
            "REPO_DIR": "/scratch/jr7309",
            "OUTPUT_BASE": str(root.parent),
            "DATA_ROOT": "/scratch/jr7309/data/om4_onedeg_v3",
            "SIF_DIR": "/scratch/jr7309/.apptainer-images",
            "APPTAINER_CACHEDIR": "/scratch/jr7309/apptainer-cache",
            "SINGULARITY_CACHEDIR": "/scratch/jr7309/singularity-cache",
            "CONTAINER_HASH": args.container_hash,
            "CODE_LAYER": args.code_layer,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NCCL_P2P_DISABLE": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "WANDB_MODE": "online",
            "SAMUDRA_MANAGE_RUN_DIR": "0",
            "SAMUDRA_MODULE": "samudra.experiments.surface_adaptation",
        }
    )
    jobs: dict[str, dict[str, Any]] = {}
    selected = {
        "qualification": ("A", "B", "C"),
        "production": ("A", "B", "C", "D"),
        "rate-control": ("E",) if args.control_seed == 1729 else ("F",),
    }[args.stage]
    for name in selected:
        arm, seed = ARMS[name]
        dependency_key = {"C": "A", "D": "B"}.get(name)
        dependencies = [jobs[dependency_key]["id"]] if dependency_key else []
        dependencies.extend(args.after_jobs)
        dependency = ":".join(dependencies) if dependencies else None
        module_args = [
            "--arm",
            arm,
            "--seed",
            str(seed),
            "--wave1-root",
            args.wave1_root,
            "--output",
            str(root / name),
            "--name",
            root.name + "-" + name,
            "--hours",
            "3.5",
            "--learning-rate",
            "1e-4" if name in ("E", "F") else "1e-5",
            "--readers",
            "8",
            "--batch-size",
            "2",
            "--device-cache",
        ]
        if args.stage == "qualification":
            module_args += ["--max-steps", "30", "--val-origins", "8"]
        env = dict(
            environment,
            NAME=root.name + "/" + name,
            DATA_CACHE_DIR="/scratch/jr7309/.data_cache/" + root.name + "-" + name,
            SAMUDRA_MODULE_ARGS=shlex.join(module_args),
            UTILIZATION_CSV=str(root / name / "utilization.csv"),
        )
        command = [
            "sbatch",
            "--parsable",
            "--account=torch_pr_347_lzanna",
            "--partition=rtx6000_lzanna",
            "--chdir=/scratch/jr7309",
            "--output=/scratch/jr7309/slurm-%j.out",
            "--error=/scratch/jr7309/slurm-%j.err",
            "--kill-on-invalid-dep=yes",
            "--nodes=1",
            "--ntasks-per-node=1",
            "--cpus-per-task=8",
            "--mem=64G",
            "--gres=gpu:rtx6000:4",
            "--gres-flags=disable-binding",
            "--time=" + ("00:15:00" if args.stage == "qualification" else "04:00:00"),
            "--job-name=surface-w2-" + args.stage + "-" + name,
        ]
        if dependency:
            command += ["--dependency=afterok:" + dependency]
        command += [args.wrapper]
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, check=True
        )
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise ValueError("Unexpected sbatch response")
        jobs[name] = dict(
            id=job,
            arm=arm,
            seed=seed,
            gpus=4,
            wall_hours=wall_hours,
            dependency=dependency,
            code_commit=args.code_commit,
            module_args=module_args,
            command=command,
            deadline=deadline.isoformat(),
            submitted_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        )
        temporary = root / "jobs.tmp"
        temporary.write_text(json.dumps(jobs, indent=2) + "\n")
        temporary.replace(root / "jobs.json")
        print(name, job, flush=True)
    print(
        "Maximum allocated GPU-hours for stage:",
        len(selected) * 4 * wall_hours,
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("qualification", "production", "rate-control"),
        required=True,
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--wave1-root", required=True)
    parser.add_argument("--qualification-root")
    parser.add_argument("--code-layer", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--container-hash", required=True)
    parser.add_argument("--wrapper", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--control-seed", type=int, choices=(1729, 1730), default=1729)
    parser.add_argument(
        "--after-jobs",
        nargs="*",
        default=[],
        help="Wait for these existing jobs to succeed",
    )
    submit(parser.parse_args())


if __name__ == "__main__":
    main()
