#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Use all four GPUs; run the independent 4k fork after the 8k trajectory."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    runner = Path(__file__).with_name("run_monthly_replay.py")

    def run(name, qualify=False):
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            str(runner),
            "--root",
            str(args.root),
            "--name",
            name,
        ]
        if qualify:
            command.append("--qualify")
        subprocess.run(command, check=True)

    if args.qualify:
        run("A-shared-era5-8k", True)
        return
    run("A-shared-era5-8k")
    run("A-shared-era5-cool4k", True)
    run("A-shared-era5-cool4k")


if __name__ == "__main__":
    main()
