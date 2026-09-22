#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Submit one single-GPU initializer diagnostic with an immutable producer."""

import argparse
import datetime
import json
import os
import shlex
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--code-commit", required=True)
    p.add_argument("--code-layer", required=True)
    p.add_argument("--container-hash", required=True)
    p.add_argument("--task", choices=["precision", "fit"], required=True)
    p.add_argument("--objective", choices=["full", "field"], default="field")
    p.add_argument("--subset", type=int, choices=[0, 1, 16], default=0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--wall-hours", type=float, default=3)
    p.add_argument("--qualification")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--heldout", action="store_true")
    args = p.parse_args()
    if args.smoke and (args.steps > 3 or args.heldout):
        p.error("Smoke requires <=3 steps and no heldout evaluation")
    if not args.smoke:
        proof = json.loads(Path(args.qualification).read_text())
        if proof["signature"]["producer"] != args.code_commit:
            raise ValueError("Qualification producer differs")
    for key in ["GHCR_USERNAME", "GHCR_TOKEN", "WANDB_API_KEY"]:
        if not os.environ.get(key):
            raise ValueError("Missing " + key)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / (args.name + "-submission.json")
    if record_path.exists():
        raise FileExistsError(record_path)
    out = root / args.name
    module_args = [
        "--task",
        args.task,
        "--objective",
        args.objective,
        "--subset",
        str(args.subset),
        "--max-steps",
        str(args.steps),
        "--eval-every",
        str(args.eval_every),
        "--precision",
        args.precision,
        "--output",
        str(out),
        "--name",
        root.name + "-" + args.name,
        "--initial-checkpoint",
        "/scratch/jr7309/runs/2026-09-22-initializer-wave3-primary/D/best.pt",
    ]
    if args.heldout:
        module_args.append("--heldout")
    env = dict(os.environ)
    env.update(
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
            "WANDB_MODE": "online",
            "SAMUDRA_MANAGE_RUN_DIR": "0",
            "SAMUDRA_MODULE": "samudra.experiments.initializer_diagnostic_wave",
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
        "--cpus-per-task=2",
        "--mem=24G",
        "--gres-flags=disable-binding",
        "--time=" + str(int(args.wall_hours * 60)),
        "--job-name=surface-diag-" + args.name,
        "--account=torch_pr_347_lzanna",
        "--partition=rtx6000_lzanna",
        "--gres=gpu:rtx6000:1",
        "/scratch/jr7309/slurm_initializer_wave.sbatch",
    ]
    result = subprocess.run(
        command, env=env, check=True, capture_output=True, text=True
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
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
