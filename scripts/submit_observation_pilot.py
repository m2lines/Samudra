#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Submit a pinned observational pilot DAG after verified data arrive on Torch."""

import argparse
import copy
import datetime
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


def pending_dependencies(jobs):
    """Retain live dependencies; use accounting to prove expired jobs succeeded."""
    pending, checks = [], []
    for job in jobs:
        queue = subprocess.run(
            ["squeue", "-h", "-j", job, "-o", "%i|%T"],
            capture_output=True,
            text=True,
        )
        rows = [line.split("|") for line in queue.stdout.splitlines()]
        if any(row[0] == job for row in rows):
            pending.append(job)
            checks.append({"job": job, "proof": "live scheduler dependency"})
            continue
        accounting = subprocess.check_output(
            ["sacct", "-nXP", "-j", job, "--format=JobIDRaw,State,ExitCode"],
            text=True,
        )
        rows = [line.strip().split("|") for line in accounting.splitlines()]
        matching = [row for row in rows if row[0] == job]
        if len(matching) != 1 or matching[0][1:3] != ["COMPLETED", "0:0"]:
            raise ValueError(
                f"Dependency {job} lacks successful completion proof: {accounting}"
            )
        checks.append(
            {"job": job, "proof": "accounting COMPLETED 0:0; dependency satisfied"}
        )
    return pending, checks


def submit(args, name, module, module_args, hours, dependencies=()):
    root = Path(args.root)
    record_path = root / (name + "-submission.json")
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if (
            record["code_commit"] != args.code_commit
            or record["module_args"] != module_args
        ):
            raise ValueError("Existing submission has different producer or arguments")
        return record["job"]
    for key in ("GHCR_USERNAME", "GHCR_TOKEN", "WANDB_API_KEY"):
        if not os.environ.get(key):
            raise ValueError("Missing " + key)
    layer = Path("/scratch/jr7309/.apptainer-code-layers") / (
        "samudra-code-" + args.code_commit + ".img"
    )
    if not layer.is_file():
        raise FileNotFoundError(layer)
    environment = dict(os.environ)
    environment.update(
        CONFIG="src/samudra/configs/samudra_om4/train.yaml",
        REPO_DIR="/scratch/jr7309",
        OUTPUT_BASE="/scratch/jr7309/runs",
        DATA_ROOT="/scratch/jr7309/data",
        SIF_DIR="/scratch/jr7309/.apptainer-images",
        APPTAINER_CACHEDIR="/scratch/jr7309/apptainer-cache",
        SINGULARITY_CACHEDIR="/scratch/jr7309/singularity-cache",
        CONTAINER_HASH="9cd36b1bdcf4921fb027494afa4157e217818ccd",  # pragma: allowlist secret
        CODE_LAYER=str(layer),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NCCL_P2P_DISABLE="1",
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
        WANDB_MODE="online",
        SAMUDRA_MANAGE_RUN_DIR="0",
        SAMUDRA_MODULE=module,
        SAMUDRA_MODULE_ARGS=shlex.join(module_args),
        NAME=root.name + "/" + name,
        DATA_CACHE_DIR="/scratch/jr7309/.data_cache/obs-D-" + name,
        REQUEUE_ON_USR1="1",
        UTILIZATION_CSV=str(root / ("telemetry-" + name + ".csv")),
    )
    command = [
        "sbatch",
        "--parsable",
        "--account=torch_pr_347_general",
        "--comment=preemption=yes;preemption_partitions_only=yes;requeue=true",
        "--constraint=h200",
        "--gres=gpu:1",
        "--nodes=1",
        "--ntasks-per-node=1",
        "--cpus-per-task=4",
        "--mem=32G",
        "--requeue",
        "--signal=B:USR1@300",
        "--chdir=/scratch/jr7309",
        "--output=/scratch/jr7309/slurm-%j.out",
        "--error=/scratch/jr7309/slurm-%j.err",
        "--time=" + str(int(hours * 60)),
        "--job-name=obs-D-" + name,
        "--kill-on-invalid-dep=yes",
    ]
    dependencies, dependency_checks = pending_dependencies(dependencies)
    if dependencies:
        command += ["--dependency=afterok:" + ":".join(dependencies)]
    command += ["/scratch/jr7309/slurm_initializer_wave.sbatch"]
    result = subprocess.run(
        command, env=environment, check=False, capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(f"sbatch rejected {name}: {result.stderr.strip()}")
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError(result.stdout)
    record = dict(
        job=job,
        code_commit=args.code_commit,
        command=command,
        module=module,
        module_args=module_args,
        dependency_checks=dependency_checks,
        submitted_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--eval-code-commit",
        help="Optional evaluation-only producer; training stays pinned",
    )
    parser.add_argument(
        "--root", default="/scratch/jr7309/runs/2026-09-22-observation-D"
    )
    parser.add_argument("--data", default="/scratch/jr7309/data/obs-d-pilot")
    parser.add_argument("--wait-hours", type=float, default=8)
    parser.add_argument(
        "--parallel-scratch",
        action="store_true",
        help="Allow a third Torch GPU for the scratch control",
    )
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_hours * 3600
    while not (Path(args.data) / "DATA_READY.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Verified observational data did not arrive within staging window"
            )
        time.sleep(60)
    contract = json.loads((root / "qualification-h200/QUALIFIED.json").read_text())
    if (
        not contract["strict_load"]
        or contract["zero_adapter_source_equivalence"] != "bitwise exact"
    ):
        raise ValueError("Source checkpoint contract qualification is incomplete")
    checkpoint = "/scratch/jr7309/runs/2026-09-22-initializer-wave3-primary/D/best.pt"
    module = "samudra.experiments.observation_pilot"
    base = [
        "--data",
        args.data,
        "--checkpoint",
        checkpoint,
        "--source-contract",
        str(root / "qualification-h200/QUALIFIED.json"),
    ]

    def train(name, extra=(), dependencies=(), hours=9):
        return submit(
            args,
            name,
            module,
            base
            + ["--output", str(root / name), "--name", root.name + "-" + name]
            + list(extra),
            hours,
            dependencies,
        )

    fit = train("fitting", ["--fit-probe", "--wandb-mode", "disabled"], hours=1)
    reference = [
        "--selection-reference",
        str(root / "fitting/selection-reference.json"),
    ]
    qualified = ["--qualification", str(root / "fitting/QUALIFIED.json")] + reference
    primary = train("primary", qualified, [fit])
    adapter = train("adapter-only", qualified + ["--adapter-only"], [fit])
    # Primary and adapter share budgets and each use one GPU. The optional
    # third Torch GPU starts scratch early without changing any per-arm budget.
    scratch_fit = train(
        "scratch-fitting",
        ["--fit-probe", "--from-scratch", "--wandb-mode", "disabled"] + reference,
        [fit] if args.parallel_scratch else [primary, adapter],
        hours=1,
    )
    scratch = train(
        "scratch",
        [
            "--from-scratch",
            "--qualification",
            str(root / "scratch-fitting/QUALIFIED.json"),
            "--reconstruction-hours",
            "4",
            "--joint-hours",
            "4",
        ]
        + reference,
        [scratch_fit],
        hours=10,
    )
    eval_module = "samudra.experiments.observation_evaluate"
    eval_args = copy.copy(args)
    eval_args.code_commit = args.eval_code_commit or args.code_commit
    primary_eval = submit(
        eval_args,
        "primary-evaluation",
        eval_module,
        [
            "--",
            "--run",
            str(root / "primary"),
            "--output",
            str(root / "primary-evaluation"),
        ],
        3,
        [primary] if args.parallel_scratch else [primary, adapter],
    )
    adapter_eval = submit(
        eval_args,
        "adapter-evaluation",
        eval_module,
        [
            "--",
            "--run",
            str(root / "adapter-only"),
            "--output",
            str(root / "adapter-evaluation"),
        ],
        3,
        [adapter, primary_eval],
    )
    scratch_eval = submit(
        eval_args,
        "scratch-evaluation",
        eval_module,
        [
            "--",
            "--run",
            str(root / "scratch"),
            "--output",
            str(root / "scratch-evaluation"),
        ],
        3,
        [scratch, adapter_eval],
    )
    (root / "DAG_SUBMITTED.json").write_text(
        json.dumps(
            dict(
                fitting=fit,
                primary=primary,
                adapter=adapter,
                scratch_fitting=scratch_fit,
                scratch=scratch,
                primary_evaluation=primary_eval,
                adapter_evaluation=adapter_eval,
                scratch_evaluation=scratch_eval,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
