#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Submit one immutable, resumable wave-three run, persisting its full request."""

import argparse
import datetime
import json
import os
import shlex
import subprocess
from pathlib import Path


def submit(args):
    root = Path(args.root)
    out = root / args.name
    submission = root / (args.name + "-submission.json")
    if submission.exists():
        raise FileExistsError(submission)
    deadline = datetime.datetime.fromisoformat(args.deadline.replace("Z", "+00:00"))
    if (
        datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=args.wall_hours)
        >= deadline
    ):
        raise ValueError("Insufficient time before deadline")
    for key in ("GHCR_USERNAME", "GHCR_TOKEN", "WANDB_API_KEY"):
        if not os.environ.get(key):
            raise ValueError("Missing credential variable " + key)
    if args.stage != "qualification":
        proof = json.loads(Path(args.qualification).read_text())
        if (
            proof.get("code_commit") != args.code_commit
            or proof.get("arm") != args.arm
            or not proof.get("max_steps")
            or proof.get("phase") != args.phase
        ):
            raise ValueError("Qualification producer, arm, or phase differs")
    root.mkdir(parents=True, exist_ok=True)
    module_args = [
        "--arm",
        args.arm,
        "--phase",
        args.phase,
        "--output",
        str(out),
        "--name",
        root.name + "-" + args.name,
        "--hours",
        str(args.hours),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--deadline",
        args.deadline,
        "--batch-size",
        "2",
        "--accumulate",
        str(4 // args.gpus),
    ]
    if args.stage == "qualification":
        module_args += ["--max-steps", "3", "--val-origins", "4"]
    if args.stage == "pilot":
        module_args += ["--validation-seconds", "600"]
    if args.initial_checkpoint:
        module_args += ["--initial-checkpoint", args.initial_checkpoint]
    if args.evaluate_only:
        module_args += ["--evaluate-only"]
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
            "SAMUDRA_MODULE": "samudra.experiments.initializer_wave",
            "NAME": root.name + "/" + args.name,
            "DATA_CACHE_DIR": "/scratch/jr7309/.data_cache/"
            + root.name
            + "-"
            + args.name,
            "SAMUDRA_MODULE_ARGS": shlex.join(module_args),
            "UTILIZATION_CSV": str(out / "utilization.csv"),
        }
    )
    command = [
        "sbatch",
        "--parsable",
        "--chdir=/scratch/jr7309",
        "--output=/scratch/jr7309/slurm-%j.out",
        "--error=/scratch/jr7309/slurm-%j.err",
        "--nodes=1",
        "--ntasks-per-node=1",
        "--cpus-per-task=" + str(args.gpus * 2),
        "--mem=" + str(args.gpus * 16) + "G",
        "--gres-flags=disable-binding",
        "--time=" + str(int(args.wall_hours * 60)),
        "--job-name=surface-w3-" + args.name,
        "--kill-on-invalid-dep=yes",
    ]
    if args.gpu == "h200":
        command += [
            "--account=torch_pr_347_general",
            "--comment=preemption=yes;"
            + ("preemption_partitions_only=yes;" if args.preemption_only else "")
            + "requeue=true",
            "--constraint=h200",
            "--gres=gpu:" + str(args.gpus),
            "--requeue",
            "--signal=B:USR1@300",
        ]
    else:
        command += [
            "--account=torch_pr_347_lzanna",
            "--partition=rtx6000_lzanna",
            "--gres=gpu:rtx6000:" + str(args.gpus),
        ]
    if args.after:
        command += ["--dependency=afterok:" + args.after]
    command += [args.wrapper]
    result = subprocess.run(
        command, env=environment, check=True, text=True, capture_output=True
    )
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError(result.stdout)
    record = {
        "id": job,
        "arguments": vars(args),
        "command": command,
        "module_args": module_args,
        "submitted_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    submission.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--arm", choices=list("ABCDEF"), required=True)
    p.add_argument(
        "--stage",
        choices=[
            "qualification",
            "pilot",
            "production",
            "replication",
            "joint",
            "evaluation",
        ],
        required=True,
    )
    p.add_argument(
        "--phase", choices=["reconstruction", "joint"], default="reconstruction"
    )
    p.add_argument("--code-layer", required=True)
    p.add_argument("--code-commit", required=True)
    p.add_argument("--container-hash", required=True)
    p.add_argument("--wrapper", default="/scratch/jr7309/slurm_initializer_wave.sbatch")
    p.add_argument("--qualification")
    p.add_argument("--gpu", choices=["h200", "rtx6000"], default="h200")
    p.add_argument("--preemption-only", action="store_true")
    p.add_argument("--gpus", type=int, choices=[1, 2, 4], default=2)
    p.add_argument("--hours", type=float, default=8)
    p.add_argument("--wall-hours", type=float, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--deadline", default="2026-09-24T17:00:00Z")
    p.add_argument("--initial-checkpoint")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--after")
    submit(p.parse_args())


if __name__ == "__main__":
    main()
