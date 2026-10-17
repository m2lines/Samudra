#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Pack the five approved single-GPU pilots onto one four-GPU Beta node."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--producer", required=True)
    p.add_argument("--qualification", required=True)
    a = p.parse_args()

    def worker(gpu, arms):
        results = []
        for arm in arms:
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_monthly_latent_pilot.py")),
                "--arm",
                arm,
                "--root",
                a.root,
                "--producer",
                a.producer,
                "--qualification",
                a.qualification,
                "--observations",
                "/projects/ny/lz1955/multiscale/jrusak/data/obs_full_range/d-observation-pilot/samples",
                "--om4",
                "/projects/ny/lz1955/multiscale/jrusak/data/diffusion-interior-beta/data/om4_onedeg_v3",  # pragma: allowlist secret
                "--cm4",
                "/projects/ny/lz1955/multiscale/jrusak/data/cm4_onedeg/CM4.zarr",
            ]
            with (
                Path(a.root) / "logs" / f"pilot-{arm}-{os.environ['SLURM_JOB_ID']}.out"
            ).open("w") as log:
                result = subprocess.run(
                    command,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            results.append((arm, result.returncode))
        return results

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(worker, gpu, arms)
            for gpu, arms in enumerate(["AE", "B", "C", "D"])
        ]
        results = [value for f in futures for value in f.result()]
    print(results, flush=True)
    if any(code for _, code in results):
        raise RuntimeError("At least one pilot failed; inspect per-arm logs")


if __name__ == "__main__":
    main()
