#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Submit the approved, bounded first wave on Torch. Does not submit later waves."""

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def submit(args):
    root = Path(args.root)
    if (root / "jobs.json").exists():
        raise FileExistsError(
            "Wave already submitted; inspect accounting before recovery"
        )
    for task in ("initializer", "ar", "direct"):
        if not (Path(args.smoke_root) / task / "COMPLETE.json").exists():
            raise ValueError(f"Missing successful end-to-end smoke: {task}")
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
            "CONTAINER_HASH": "9cd36b1bdcf4921fb027494afa4157e217818ccd",  # pragma: allowlist secret
            "CODE_LAYER": args.code_layer,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "WANDB_MODE": "online",
            "SAMUDRA_MANAGE_RUN_DIR": "0",
            "SAMUDRA_MODULE": "samudra.experiments.surface_wave",
        }
    )
    jobs = {}
    common = [
        "sbatch",
        "--parsable",
        "--account=torch_pr_347_lzanna",
        "--chdir=/scratch/jr7309",
        "--output=/scratch/jr7309/slurm-%j.out",
        "--error=/scratch/jr7309/slurm-%j.err",
        "--kill-on-invalid-dep=yes",
    ]

    def train(task, phase, gpus, wall_hours, train_hours, dependency=None):
        name = root.name + "-" + task
        env = dict(environment)
        env["NAME"] = name
        env["DATA_CACHE_DIR"] = "/scratch/jr7309/.data_cache/" + name
        module_args = [
            "--task",
            task,
            "--output",
            str(root / task),
            "--name",
            name,
            "--hours",
            str(train_hours),
            "--readers",
            "2",
        ]
        if task != "initializer":
            module_args += ["--initializer-dir", str(root / "initializer")]
        if phase == "pretrain":
            module_args += ["--stop-after-pretrain"]
        env["SAMUDRA_MODULE_ARGS"] = " ".join(module_args)
        command = common + [
            "--partition=rtx6000_lzanna",
            "--nodes=1",
            "--ntasks-per-node=1",
            f"--cpus-per-task={2 * gpus}",
            f"--mem={16 * gpus}G",
            f"--gres=gpu:rtx6000:{gpus}",
            "--gres-flags=disable-binding",
            f"--time={wall_hours}:00:00",
            f"--job-name=surface-w1-{task}-{phase}",
        ]
        if dependency:
            command += [f"--dependency=afterok:{dependency}"]
        command += [str(Path.home() / "slurm_apptainer_train.sbatch")]
        result = subprocess.run(
            command, env=env, check=True, capture_output=True, text=True
        )
        job = result.stdout.strip().split(";")[0]
        if not job.isdigit():
            raise ValueError(f"Unexpected sbatch response: {result.stdout}")
        jobs[task + "-" + phase] = {
            "id": job,
            "gpus": gpus,
            "wall_hours": wall_hours,
            "dependency": dependency,
            "command": command,
        }
        # Persist after every submission: a partial failure must never hide live jobs.
        (root / "jobs.json").write_text(json.dumps(jobs, indent=2))
        print(task, phase, job, flush=True)
        return job

    initializer = train("initializer", "train", 2, 8, 7)
    ar_pretrain = train("ar", "pretrain", 4, 40, 54, initializer)
    direct_pretrain = train("direct", "pretrain", 4, 40, 54, initializer)
    ar_joint = train("ar", "joint", 4, 20, 54, ar_pretrain)
    direct_joint = train("direct", "joint", 4, 20, 54, direct_pretrain)
    all_ids = args.smoke_jobs.split(",") + [job["id"] for job in jobs.values()]
    report_command = shlex.join(
        [
            "python3",
            args.report_script,
            "--root",
            str(root),
            "--jobs",
            ",".join(all_ids),
        ]
    )
    report = subprocess.run(
        common
        + [
            "--partition=cpu_short",
            "--cpus-per-task=1",
            "--mem=2G",
            "--time=00:10:00",
            "--job-name=surface-w1-report",
            f"--dependency=afterany:{ar_joint}:{direct_joint}",
            "--wrap",
            report_command,
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    jobs["report"] = {
        "id": report.stdout.strip().split(";")[0],
        "gpus": 0,
        "wall_hours": 1 / 6,
    }
    (root / "jobs.json").write_text(json.dumps(jobs, indent=2))
    print("report", jobs["report"]["id"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--code-layer", required=True)
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument(
        "--smoke-jobs",
        required=True,
        help="All smoke attempts, including failed/cancelled ones",
    )
    parser.add_argument("--report-script", required=True)
    submit(parser.parse_args())
