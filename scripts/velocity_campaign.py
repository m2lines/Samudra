#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Advance the authorized Torch campaign only after successful, inspectable stages.

Runs on the Torch host using stdlib Python and the existing Apptainer harness.
Training/evaluation source remains pinned to the manifest's immutable code layer.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
from pathlib import Path


def write_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def records(path):
    with (path / "history.jsonl").open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def completed_run(path):
    rows = records(path)
    if not rows or not rows[-1].get("complete"):
        raise RuntimeError(f"Training did not complete: {path}")
    scores = [
        row["validation/vector_rmse_10d"]
        for row in rows
        if "validation/vector_rmse_10d" in row
    ]
    if not scores or not all(math.isfinite(s) and s > 0 for s in scores):
        raise RuntimeError(f"Missing or invalid validation scores: {path}")
    if not (path / "best.pt").is_file():
        raise RuntimeError(f"Missing selected checkpoint: {path}")
    return min(scores), rows


def allocation_usage(job_ids):
    """Use Slurm allocation time, including duplicate/requeued accounting records."""
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "--duplicates",
            "-nP",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,State,ElapsedRaw,AllocTRES,Partition",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    usage: dict[str, float] = {}
    seen = set()
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 4 or fields[0] not in job_ids:
            continue
        seen.add(fields[0])
        tres = dict(item.split("=", 1) for item in fields[3].split(",") if "=" in item)
        gpus = int(tres.get("gres/gpu", 0))
        if not gpus:
            if int(fields[2]) == 0 and fields[1].startswith("CANCELLED"):
                continue
            raise RuntimeError(f"Missing GPU accounting for {fields[0]}")
        partition = fields[4] if len(fields) > 4 else ""
        family = next(
            (
                gpu
                for gpu in ("rtx6000", "a100", "h100", "h200", "l40s")
                if partition.startswith(gpu)
            ),
            "unspecified",
        )
        usage[family] = usage.get(family, 0.0) + int(fields[2]) * gpus / 3600
    if seen != set(job_ids):
        raise RuntimeError(
            "Slurm accounting is incomplete; refusing to allocate more compute"
        )
    return usage


def allocation_hours(job_ids):
    return sum(allocation_usage(job_ids).values())


class Campaign:
    def __init__(self, path):
        self.path = path
        self.manifest = json.loads(path.read_text())
        self.root = path.parent
        self.scratch = self.root.parents[1]
        self.env = os.environ.copy()
        self.env.update(
            SCRATCH_DIR=str(self.scratch),
            REPO_DIR=str(self.scratch),
            SIF_DIR=str(self.scratch / ".apptainer-images"),
            SIF_PATH=self.manifest["sif_path"],
            CONTAINER_HASH=self.manifest["runtime_commit"],
            CODE_LAYER=self.manifest["code_layer"],
            APPTAINER_CACHEDIR=str(self.scratch / "apptainer-cache"),
            SINGULARITY_CACHEDIR=str(self.scratch / "singularity-cache"),
            DATA_ROOT=str(self.scratch / "data/velocity-transfer-v1"),
            OUTPUT_BASE=str(self.root),
            CONFIG="src/samudra/experiments/velocity_transfer/train.py",
            SAMUDRA_MANAGE_RUN_DIR="0",
            WANDB_MODE="online",
            REQUEUE_ON_USR1="1",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="2",
        )

    def save(self):
        write_manifest(self.path, self.manifest)

    def submit(self, name, args, env=None, account="torch_pr_347_general"):
        job_env = self.env.copy()
        job_env.update(env or {})
        command = [
            "sbatch",
            "--parsable",
            "--account=" + account,
            "--chdir=" + str(self.scratch),
            "--job-name=velocity-" + name,
            "--output=" + str(self.scratch / ("velocity-" + name + "-%j.out")),
            "--error=" + str(self.scratch / ("velocity-" + name + "-%j.err")),
        ] + args
        result = subprocess.run(
            command, env=job_env, check=True, capture_output=True, text=True
        )
        job = result.stdout.strip().split(";")[0]
        print(name, job, flush=True)
        return job

    def launch_run(self, stage, variant, seed, gpu_hours, predecessor=None):
        name = f"{stage}-{variant}-s{seed}"
        key = f"{variant}:{seed}"
        jobs = self.manifest["jobs"].setdefault(stage, {})
        if key in jobs:
            return jobs[key]
        args = [
            "--constraint=" + self.manifest.get("gpu_constraint", "a100"),
            "--gres=gpu:4",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=32",
            "--mem=64G",
            f"--time={int(gpu_hours / 4) + 1}:00:00",
            "--comment=preemption=yes;requeue=true",
            "--requeue",
            "--signal=B:USR1@300",
        ]
        if predecessor:
            args.append("--dependency=afterany:" + predecessor)
        args.append("/home/jr7309/slurm_apptainer_train.sbatch")
        module_args = [
            "--data-root",
            self.env["DATA_ROOT"],
            "--output",
            str(self.root / name),
            "--variant",
            variant,
            "--seed",
            str(seed),
            "--gpu-hours",
            str(gpu_hours),
            "--checkpoint-every",
            "64",
            "--wandb-mode",
            "online",
        ]
        job = self.submit(
            name,
            args,
            {
                "NAME": name,
                "SAMUDRA_MODULE": "samudra.experiments.velocity_transfer.train",
                "SAMUDRA_MODULE_ARGS": " ".join(shlex.quote(v) for v in module_args),
                "GPUS_PER_NODE": "4",
                "DATA_CACHE_DIR": str(self.scratch / ".data_cache" / name),
            },
            account=self.manifest.get("gpu_account", "torch_pr_347_general"),
        )
        jobs[key] = job
        self.save()
        return job

    def next_stage(self, stage, dependencies):
        key = "controller_" + stage
        if key in self.manifest["jobs"]:
            return
        command = " ".join(
            shlex.quote(s)
            for s in [
                "python3",
                self.manifest["controller_path"],
                "--manifest",
                str(self.path),
                "--stage",
                stage,
            ]
        )
        job = self.submit(
            "advance-" + stage,
            [
                "--partition=cs",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=1",
                "--mem=2G",
                "--time=00:15:00",
                "--dependency=afterany:" + ":".join(dependencies),
                "--wrap",
                command,
            ],
        )
        self.manifest["jobs"][key] = job
        self.save()

    def spent(self):
        jobs = self.manifest["jobs"]
        ids = list(jobs["pilots"].values())
        ids.extend(jobs.get("hardware_pilots", {}).values())
        for stage in ("screen", "confirm", "evaluate"):
            ids.extend(jobs.get(stage, {}).values())
        for retired in self.manifest.get("retired_gpu_jobs", []):
            ids.extend(retired["job_ids"])
        usage = allocation_usage(list(dict.fromkeys(ids)))
        hours = sum(usage.values())
        self.manifest["allocated_gpu_hours_at_last_gate"] = hours
        self.manifest["allocated_gpu_hours_by_type_at_last_gate"] = usage
        self.save()
        return hours

    def screen(self):
        pilots = [f"pilot-{variant}-s15" for variant in ("D0", "D3")]
        pilots.extend(self.manifest.get("hardware_pilot_runs", []))
        for pilot in pilots:
            _, rows = completed_run(self.root / pilot)
            config_path = self.root / pilot / "config.json"
            world = (
                json.loads(config_path.read_text())["world_size"]
                if config_path.exists()
                else 2
            )
            measurements = [r for r in rows if "peak_rss_gib_rank0" in r]
            if (
                not measurements
                or max(r["peak_rss_gib_rank0"] for r in measurements) * world > 24
                or max(r["peak_cuda_gib_rank0"] for r in measurements) > 24
            ):
                raise RuntimeError(
                    "Pilot memory gate failed; inspect before larger allocations"
                )
        if self.spent() + 288 + 768 + 96 > self.manifest["budget_gpu_hours"]:
            raise RuntimeError("Insufficient remaining campaign budget")
        lanes = [None, None]
        submitted = []
        for index in range(6):
            job = self.launch_run("screen", f"D{index}", 15, 48, lanes[index % 2])
            lanes[index % 2] = job
            submitted.append(job)
        self.manifest["stage"] = "screen"
        self.manifest["screen_not_submitted"] = False
        self.save()
        self.next_stage("confirm", submitted)

    def confirm(self):
        scores = {
            f"D{i}": completed_run(self.root / (f"screen-D{i}-s15"))[0]
            for i in range(6)
        }
        winner = min((v for v in scores if v != "D0"), key=lambda v: scores[v])
        self.manifest.update(
            screen_validation_rmse=scores, selected_transfer_arm=winner
        )
        self.save()
        if self.spent() + 768 + 96 > self.manifest["budget_gpu_hours"]:
            raise RuntimeError("Insufficient remaining budget for seed confirmation")
        lanes = [None, None]
        submitted = []
        for seed in (15, 16, 17):
            for lane, variant in enumerate(("D0", winner)):
                job = self.launch_run("confirm", variant, seed, 128, lanes[lane])
                lanes[lane] = job
                submitted.append(job)
        self.manifest["stage"] = "confirm"
        self.save()
        self.next_stage("evaluate", submitted)

    def evaluate(self):
        winner = self.manifest["selected_transfer_arm"]
        for seed in (15, 16, 17):
            for variant in ("D0", winner):
                completed_run(self.root / f"confirm-{variant}-s{seed}")
        # Twelve single-GPU jobs, each capped at 90 minutes: at most 18 GPU-hours.
        if self.spent() + 18 > self.manifest["budget_gpu_hours"]:
            raise RuntimeError("Insufficient remaining evaluation budget")
        jobs = self.manifest["jobs"].setdefault("evaluate", {})
        for split in ("validation", "test"):
            for seed in (15, 16, 17):
                for variant in ("D0", winner):
                    key = f"{split}:{variant}:{seed}"
                    if key in jobs:
                        continue
                    run = f"confirm-{variant}-s{seed}"
                    name = f"{split}-{run}"
                    args = [
                        "--checkpoint",
                        str(self.root / run / "best.pt"),
                        "--data",
                        self.env["DATA_ROOT"] + "/duacs",
                        "--output",
                        str(self.root / name),
                        "--split",
                        split,
                    ]
                    prior = list(jobs.values())[-8] if len(jobs) >= 8 else None
                    sbatch = [
                        "--constraint=" + self.manifest.get("gpu_constraint", "a100"),
                        "--gres=gpu:1",
                        "--nodes=1",
                        "--ntasks=1",
                        "--cpus-per-task=8",
                        "--mem=16G",
                        "--time=01:30:00",
                        "--comment=preemption=yes;requeue=false",
                    ]
                    if prior:
                        sbatch.append("--dependency=afterany:" + prior)
                    sbatch.append("/home/jr7309/slurm_apptainer_train.sbatch")
                    jobs[key] = self.submit(
                        name,
                        sbatch,
                        {
                            "NAME": name,
                            "SAMUDRA_MODULE": "samudra.experiments.velocity_transfer.evaluate",
                            "SAMUDRA_MODULE_ARGS": " ".join(
                                shlex.quote(v) for v in args
                            ),
                            "GPUS_PER_NODE": "1",
                            "REQUEUE_ON_USR1": "0",
                            "DATA_CACHE_DIR": str(self.scratch / ".data_cache" / name),
                        },
                        account=self.manifest.get(
                            "gpu_account", "torch_pr_347_general"
                        ),
                    )
                    self.save()
        self.manifest["stage"] = "evaluate"
        self.save()
        self.next_stage("finish", list(jobs.values()))

    def finish(self):
        for split in ("validation", "test"):
            for seed in (15, 16, 17):
                for variant in ("D0", self.manifest["selected_transfer_arm"]):
                    path = (
                        self.root / f"{split}-confirm-{variant}-s{seed}/manifest.json"
                    )
                    if not json.loads(path.read_text())["complete"]:
                        raise RuntimeError(f"Incomplete evaluation: {path}")
        self.spent()
        layer = Path(self.manifest["code_layer"])
        subprocess.run(
            ["sha256sum", "--check", "--status", layer.name + ".sha256"],
            cwd=layer.parent,
            check=True,
        )
        comparisons = {}
        for split in ("validation", "test"):
            output = self.root / (split + "-comparison.json")
            if not output.exists():
                args = [
                    "apptainer",
                    "exec",
                    "--overlay",
                    str(layer) + ":ro",
                    "--bind",
                    str(self.root) + ":" + str(self.root),
                    "--pwd",
                    "/opt/samudra-code",
                    self.manifest["sif_path"],
                    "env",
                    "PYTHONPATH=/opt/samudra-code/src",
                    "/workspace/.venv/bin/python",
                    "-m",
                    "samudra.experiments.velocity_transfer.compare",
                    "--control",
                ]
                args += [
                    str(self.root / f"{split}-confirm-D0-s{seed}")
                    for seed in (15, 16, 17)
                ]
                args += ["--candidate"] + [
                    str(
                        self.root
                        / "{}-confirm-{}-s{}".format(
                            split, self.manifest["selected_transfer_arm"], seed
                        )
                    )
                    for seed in (15, 16, 17)
                ]
                args += ["--output", str(output)]
                subprocess.run(args, env=self.env, check=True)
            comparisons[split] = str(output)
        self.manifest.update(stage="complete", comparisons=comparisons)
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("screen", "confirm", "evaluate", "finish"), required=True
    )
    args = parser.parse_args()
    with args.manifest.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        campaign = Campaign(args.manifest)
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if digest != campaign.manifest["controller_sha256"]:
            raise RuntimeError("Controller checksum mismatch")
        try:
            getattr(campaign, args.stage)()
        except Exception as error:
            campaign.manifest["stopped_at_stage"] = args.stage
            campaign.manifest["failure"] = str(error)
            campaign.save()
            raise


if __name__ == "__main__":
    main()
