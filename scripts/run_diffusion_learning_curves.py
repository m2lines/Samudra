#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Evaluate prespecified intermediate v2 checkpoints, without changing training."""

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

from samudra.experiments.diffusion_interventions import V2_ARMS
from samudra.experiments.observation_pilot import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    gpus: queue.Queue[int] = queue.Queue()
    for gpu in range(4):
        gpus.put(gpu)
    logroot = args.root / "logs" / os.environ.get("SLURM_JOB_ID", "curves-local")
    logroot.mkdir(parents=True, exist_ok=True)
    work = [(arm, update) for update in (32, 64, 128) for arm in V2_ARMS]

    def evaluate(task):
        arm, update = task
        gpu = gpus.get()
        start = time.monotonic()
        checkpoint = args.root / "runs" / arm / f"step-{update:04d}.pt"
        output = args.root / "learning-curves" / arm / f"step-{update:04d}"
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            LOCAL_RANK="0",
            WORLD_SIZE="1",
            RANK="0",
        )
        try:
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            done = output / "COMPLETE.json"
            if done.exists():
                saved = json.loads(done.read_text())["inputs"]
                if saved["checkpoint_sha256"] != digest(checkpoint) or saved[
                    "counts"
                ] != {"om4": 8000 + update // 2, "observation": 8000 + update // 2}:
                    raise ValueError(
                        "Existing evaluation belongs to different checkpoint"
                    )
                return dict(arm=arm, update=update, success=True, reused=True)
            command = [
                sys.executable,
                "-m",
                "samudra.experiments.diffusion_global_annual",
                "--data",
                str(args.root / "data/observations"),
                "--annual-data",
                str(args.root / "data/annual_observations"),
                "--reference",
                str(args.root / "checkpoints/selection-reference.json"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--steps",
                "32",
                "--expected-observation-updates",
                str(8000 + update // 2),
                "--expected-om4-updates",
                str(8000 + update // 2),
            ]
            with (logroot / f"{arm}-{update:04d}.log").open("a") as log:
                subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            return dict(
                arm=arm,
                update=update,
                success=True,
                reused=False,
                seconds=time.monotonic() - start,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as error:
            return dict(
                arm=arm,
                update=update,
                success=False,
                error=str(error),
                seconds=time.monotonic() - start,
            )
        finally:
            gpus.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(evaluate, work))
    (logroot / "RESULTS.json").write_text(json.dumps(results, indent=2) + "\n")
    if not all(row["success"] for row in results):
        raise RuntimeError(
            "Learning curve evaluations incomplete; inspect RESULTS.json"
        )


if __name__ == "__main__":
    main()
