#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only hourly Slurm/progress monitor; writes local status, never submits jobs."""

import argparse
import datetime
import json
import shlex
import subprocess
import time
from pathlib import Path

REMOTE = r"""
import json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
ids = sys.argv[2].split(',')
raw = subprocess.check_output(['sacct', '-j', ','.join(ids), '-n', '-P', '-o', 'JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode'], text=True)
jobs = []
for line in raw.splitlines():
    parts = line.split('|')
    if parts[0] in ids:
        jobs.append(dict(zip(['id','state','elapsed_seconds','resources','exit_code'],parts[:5])))
runs = {}
for name in ['smoke','diffusion','deterministic']:
    path = root/name
    events = {}
    progress = path/'progress.jsonl'
    if progress.exists():
        for line in progress.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get('event') in ['train_step','validation_selection','ensemble_evaluation']:
                events[record['event']] = record
    runs[name] = dict(train_complete=(path/'TRAIN_COMPLETE.json').exists(), evaluation_complete=(path/'COMPLETE.json').exists(), exception=(path/'exception.log').exists(), latest=events)
print(json.dumps(dict(jobs=jobs,runs=runs)))
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--jobs", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interval", type=int, default=3600)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    if not all(j.isdigit() for j in args.jobs) or args.interval < 600:
        p.error("Numeric job ids and at least 600 seconds between checks required")
    args.output.mkdir(parents=True, exist_ok=True)
    command = shlex.join(["python3", "-c", REMOTE, args.root, ",".join(args.jobs)])
    failures = {
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
    }
    while True:
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=20",
                    "torch",
                    command,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            snapshot = json.loads(result.stdout)
            states = [j["state"].split()[0].rstrip("+") for j in snapshot["jobs"]]
            if any(s in failures for s in states):
                phase = "blocked_job_failure"
            elif len(states) == len(args.jobs) and all(
                s == "COMPLETED" for s in states
            ):
                phase = "complete_awaiting_collection"
            else:
                phase = "running"
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            snapshot = {"error": str(exc)}
            phase = "blocked_monitor_connection"
        snapshot.update(
            phase=phase, time_utc=datetime.datetime.now(datetime.UTC).isoformat()
        )
        payload = json.dumps(snapshot, indent=2) + "\n"
        temporary = args.output / "latest.tmp"
        temporary.write_text(payload)
        temporary.replace(args.output / "latest.json")
        with (args.output / "history.jsonl").open("a") as f:
            f.write(json.dumps(snapshot) + "\n")
        print(
            json.dumps({"phase": phase, "time_utc": snapshot["time_utc"]}), flush=True
        )
        if args.once or phase != "running":
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
