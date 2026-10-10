#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Durable scheduler-exit relay; callback agent verifies accounting and artifacts."""

import argparse
import shlex
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--credential", required=True)
    parser.add_argument("--sender", required=True)
    args = parser.parse_args()
    if not args.job.isdigit():
        raise ValueError("Expected numeric Beta Slurm job ID")
    failures = 0
    while True:
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "ConnectTimeout=15",
                    "beta",
                    "bash -lc " + shlex.quote(f"squeue -h -j {args.job} -o %T"),
                ],
                capture_output=True,
                text=True,
                timeout=45,
            )
            if result.returncode:
                failures += 1
            else:
                failures = 0
                if not result.stdout.strip():
                    break
        except subprocess.TimeoutExpired:
            failures += 1
        if failures >= 3:
            break
        time.sleep(60)
    subprocess.run(
        [sys.executable, args.sender, "--credential-file", args.credential], check=True
    )


if __name__ == "__main__":
    main()
