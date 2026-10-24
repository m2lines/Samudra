#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Qualify then train the five additional independent OBS cooldowns."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.root / "paths.json").read_text())["runs"]
    if sorted(row["fork_step"] for row in rows) != [2000, 3000, 5000, 6000, 7000]:
        raise ValueError("Expected precisely the five authorized cooldowns")
    logs = args.root / "logs"
    logs.mkdir(exist_ok=True)
    runner = Path(__file__).with_name("run_monthly_replay.py")

    def worker(gpu):
        for row in rows[gpu::4]:
            command = [
                sys.executable,
                str(runner),
                "--root",
                str(args.root),
                "--name",
                row["name"],
            ]
            for qualify in (True, False):
                stage = "qualification" if qualify else "production"
                with (
                    logs / f"{row['name']}-{stage}-{os.environ['SLURM_JOB_ID']}.out"
                ).open("a") as stream:
                    subprocess.run(
                        command + (["--qualify"] if qualify else []),
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))


if __name__ == "__main__":
    main()
