#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Qualify sequentially; train matched A/E without processor forcing in parallel."""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--qualify", action="store_true")
    a = p.parse_args()

    def worker(index):
        name = ("A-no-processor-forcing", "E-no-processor-forcing")[index]
        stage = "qualification" if a.qualify else "production"
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_monthly_replay.py")),
            "--root",
            str(a.root),
            "--name",
            name,
        ]
        if a.qualify:
            command.append("--qualify")
        with (
            a.root / "logs" / f"{stage}-{name}-{os.environ['SLURM_JOB_ID']}.out"
        ).open("a") as stream:
            subprocess.run(
                command,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "0" if a.qualify else str(index),
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    if a.qualify:
        for i in range(2):
            worker(i)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, i) for i in range(2)]
            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
