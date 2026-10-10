#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Run six independent single-GPU arms within one four-GPU Beta allocation."""

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
from pathlib import Path

ARMS = (
    "control",
    "multiscale",
    "replay",
    "multiscale-replay",
    "unroll12",
    "latent-jitter",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", choices=("v1", "v2"), default="v1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    from samudra.experiments.diffusion_interventions import V2_ARMS

    arms = V2_ARMS if args.campaign == "v2" else ARMS
    updates = 256 if args.campaign == "v2" else 128
    gpus: queue.Queue[int] = queue.Queue()
    for gpu in range(4):
        gpus.put(gpu)
    logroot = args.root / "logs" / os.environ.get("SLURM_JOB_ID", "local")
    logroot.mkdir(parents=True, exist_ok=True)

    def arm_run(arm):
        gpu = gpus.get()
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            LOCAL_RANK="0",
            WORLD_SIZE="1",
            RANK="0",
        )
        env["TORCHINDUCTOR_CACHE_DIR"] = str(args.root / "cache" / f"gpu-{gpu}")
        env["TRITON_CACHE_DIR"] = str(args.root / "cache" / f"triton-{gpu}")
        try:
            with (logroot / f"{arm}.log").open("a") as log:
                command = [
                    sys.executable,
                    "-m",
                    "samudra.experiments.diffusion_interventions",
                    "--root",
                    str(args.root),
                    "--arm",
                    arm,
                    "--campaign",
                    args.campaign,
                    "--updates",
                    str(updates),
                    "--hours",
                    "18",
                ]
                if args.smoke:
                    command += ["--smoke", "--updates", "2", "--hours", "0.8"]
                subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
                if not args.smoke:
                    # Fixed final budget, never choose weights on annual test errors.
                    ckpt = str(args.root / "runs" / arm / f"step-{updates:04d}.pt")
                    common = [
                        "--data",
                        str(args.root / "data/observations"),
                        "--checkpoint",
                        ckpt,
                        "--reference",
                        str(args.root / "checkpoints/selection-reference.json"),
                        "--steps",
                        "32",
                    ]
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "samudra.experiments.diffusion_global_evaluate",
                            *common,
                            "--output",
                            str(args.root / "evaluation" / arm / "validation"),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                    native = [
                        sys.executable,
                        "-m",
                        "samudra.experiments.diffusion_intervention_native",
                        "--root",
                        str(args.root),
                    ]
                    subprocess.run(
                        [
                            *native,
                            "--checkpoint",
                            ckpt,
                            "--output",
                            str(args.root / "evaluation" / arm / "native"),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                    if arm == "control":
                        subprocess.run(
                            [
                                *native,
                                "--checkpoint",
                                str(args.root / "checkpoints/parent-16000.pt"),
                                "--output",
                                str(args.root / "evaluation/parent/native"),
                            ],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "samudra.experiments.diffusion_global_annual",
                            *common,
                            "--annual-data",
                            str(args.root / "data/annual_observations"),
                            "--expected-observation-updates",
                            str(8000 + updates // 2),
                            "--expected-om4-updates",
                            str(8000 + updates // 2),
                            "--output",
                            str(args.root / "evaluation" / arm / "annual"),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            return dict(arm=arm, success=True, gpu=gpu)
        except subprocess.CalledProcessError as error:
            return dict(arm=arm, success=False, exit_code=error.returncode, gpu=gpu)
        finally:
            gpus.put(gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(arm_run, arms))
    (logroot / "RESULTS.json").write_text(json.dumps(results, indent=2) + "\n")
    if not all(r["success"] for r in results):
        raise RuntimeError(f"Some arms failed; inspect {logroot}/RESULTS.json")


if __name__ == "__main__":
    main()
