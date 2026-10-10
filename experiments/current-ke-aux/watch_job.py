# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Supervised accounting watcher for Beta's GPU-only scheduler."""

import argparse
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True, type=int)
parser.add_argument("--credential", required=True)
args = parser.parse_args()
terminal = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}
while True:
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "ConnectTimeout=15",
            "beta",
            f"bash -lc 'sacct -X -j {args.job} --noheader --parsable2 --format=JobIDRaw,State'",
        ],
        capture_output=True,
        text=True,
        timeout=50,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            columns = line.split("|")
            if (
                columns[0] == str(args.job)
                and columns[1].split()[0].rstrip("+") in terminal
            ):
                subprocess.run(
                    [
                        "/usr/bin/python3",
                        "/home/jder/.codex/skills/async-job-followup/scripts/send.py",
                        "--credential-file",
                        args.credential,
                    ],
                    check=True,
                )
                raise SystemExit(0)
    time.sleep(60)
