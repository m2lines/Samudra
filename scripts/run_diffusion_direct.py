#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Run direct versus residual processor continuations and matched evaluations."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

ARMS = ("residual", "direct", "residual-warm", "direct-warm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--hours", type=float, default=7)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.updates != 128 and not args.smoke:
        raise ValueError("Production uses the fixed 128-update budget")
    updates = 2 if args.smoke else args.updates
    logroot = args.root / "logs" / os.environ.get("SLURM_JOB_ID", "local")
    logroot.mkdir(parents=True, exist_ok=True)

    def arm_run(item):
        gpu, arm = item
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            LOCAL_RANK="0",
            WORLD_SIZE="1",
            RANK="0",
            TORCHINDUCTOR_CACHE_DIR=str(args.root / "cache" / f"gpu-{gpu}"),
            TRITON_CACHE_DIR=str(args.root / "cache" / f"triton-{gpu}"),
        )
        checkpoint = args.root / ("smoke" if args.smoke else "runs") / arm
        checkpoint /= f"step-{updates:04d}.pt"
        try:
            with (logroot / f"{arm}.log").open("a") as log:

                def run(command):
                    subprocess.run(
                        [sys.executable, *command],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )

                command = [
                    "-m",
                    "samudra.experiments.diffusion_direct_train",
                    "--root",
                    str(args.root),
                    "--arm",
                    arm,
                    "--updates",
                    str(updates),
                    "--hours",
                    str(0.8 if args.smoke else args.hours),
                ]
                if args.smoke:
                    command += ["--smoke"]
                run(command)
                if args.smoke:
                    # Exercise the same architecture-aware loader used by every evaluation.
                    run(
                        [
                            "-c",
                            "import sys, numpy as np, torch; "
                            "from samudra.experiments.diffusion_global import make_model; "
                            "from samudra.experiments.diffusion_direct import load_latent_checkpoint; "
                            "saved = torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
                            "names = np.load(sys.argv[2])['names'].tolist(); "
                            "model = make_model(names, 'cuda', steps=32); "
                            "load_latent_checkpoint(model, saved); "
                            "assert saved['counts'] == {'om4': 8129, 'observation': 8129}; "
                            "print('SMOKE_ARCHITECTURE_STRICT_LOAD_OK', flush=True)",
                            str(checkpoint),
                            str(args.root / "data/observations/grid.npz"),
                        ]
                    )
                else:
                    common = [
                        "--data",
                        str(args.root / "data/observations"),
                        "--checkpoint",
                        str(checkpoint),
                        "--reference",
                        str(args.root / "checkpoints/selection-reference.json"),
                        "--steps",
                        "32",
                    ]
                    output = args.root / "evaluation" / arm
                    run(
                        [
                            "-m",
                            "samudra.experiments.diffusion_global_evaluate",
                            *common,
                            "--output",
                            str(output / "validation"),
                        ]
                    )
                    run(
                        [
                            "-m",
                            "samudra.experiments.diffusion_intervention_native",
                            "--root",
                            str(args.root),
                            "--checkpoint",
                            str(checkpoint),
                            "--output",
                            str(output / "native"),
                        ]
                    )
                    run(
                        [
                            "-m",
                            "samudra.experiments.diffusion_global_annual",
                            *common,
                            "--annual-data",
                            str(args.root / "data/annual_observations"),
                            "--expected-observation-updates",
                            "8192",
                            "--expected-om4-updates",
                            "8192",
                            "--output",
                            str(output / "annual"),
                        ]
                    )
            return dict(arm=arm, success=True, gpu=gpu)
        except subprocess.CalledProcessError as error:
            return dict(arm=arm, success=False, exit_code=error.returncode, gpu=gpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(arm_run, enumerate(ARMS)))
    (logroot / "RESULTS.json").write_text(json.dumps(results, indent=2) + "\n")
    if not all(result["success"] for result in results):
        raise RuntimeError(f"Some arms failed; inspect {logroot}/RESULTS.json")


if __name__ == "__main__":
    main()
