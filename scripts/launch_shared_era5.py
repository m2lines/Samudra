#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Keep the 8k trajectory independent from the separate 4k cooldown fork."""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--qualify", action="store_true")
    a = p.parse_args()
    runner = str(Path(__file__).with_name("run_monthly_replay.py"))
    logs = a.root / "logs"
    logs.mkdir(exist_ok=True)

    def start(name, gpu, qualification=False):
        stage = "qualification" if qualification else "production"
        command = [sys.executable, runner, "--root", str(a.root), "--name", name]
        if qualification:
            command.append("--qualify")
        with (logs / f"{stage}-{name}-{os.environ['SLURM_JOB_ID']}.out").open(
            "a"
        ) as stream:
            return subprocess.Popen(
                command,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=stream,
                stderr=subprocess.STDOUT,
            )

    if a.qualify:
        code = start("A-shared-era5-8k", 0, True).wait()
        if code:
            raise RuntimeError(f"Shared forcing qualification failed: {code}")
        return
    main_process = start("A-shared-era5-8k", 0)

    def fork():
        checkpoint = a.root / "production/A-shared-era5-8k/endpoint-04000.pt"
        while not checkpoint.exists():
            if main_process.poll() is not None:
                raise RuntimeError("Main training ended without the 4k fork")
            time.sleep(20)
        # Qualification uses a fresh copy of the saved state; never promotes its updates.
        code = start("A-shared-era5-cool4k", 1, True).wait()
        if code:
            raise RuntimeError(f"4k fork qualification failed: {code}")
        code = start("A-shared-era5-cool4k", 1).wait()
        if code:
            raise RuntimeError(f"4k cooldown failed: {code}")

    with ThreadPoolExecutor(max_workers=1) as pool:
        branch = pool.submit(fork)
        code = main_process.wait()
        branch.result()
        if code:
            raise RuntimeError(f"8k main/cooldown failed: {code}")


if __name__ == "__main__":
    main()
