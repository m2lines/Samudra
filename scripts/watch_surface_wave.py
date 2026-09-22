#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded, read-only wave monitor. Records failures/stalls; never launches jobs."""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path
from typing import Any

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


def watch(root, hours=120, interval=600):
    root = Path(root)
    deadline = time.monotonic() + hours * 3600
    while time.monotonic() < deadline:
        jobs = json.loads((root / "jobs.json").read_text())
        gpu_jobs = {name: job for name, job in jobs.items() if job["gpus"]}
        ids = ",".join(job["id"] for job in gpu_jobs.values())
        response = subprocess.run(
            [
                "sacct",
                "-j",
                ids,
                "-nP",
                "--allocations",
                "--format=JobIDRaw,State,ElapsedRaw,ExitCode",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        states: dict[str, dict[str, Any]] = {}
        for line in response.stdout.splitlines():
            job_id, state, elapsed, exit_code = line.split("|")[:4]
            states[job_id] = {
                "state": state.split()[0],
                "elapsed_seconds": int(elapsed),
                "exit_code": exit_code,
            }
        alerts = []
        for name, job in gpu_jobs.items():
            status = states.get(job["id"], {"state": "UNKNOWN", "elapsed_seconds": 0})
            if status["state"] in TERMINAL - {"COMPLETED"}:
                alerts.append(
                    f"{name}: job {job['id']} ended {status['state']}; inspect /scratch/jr7309/slurm-{job['id']}.err before recovery."
                )
            progress = root / name.split("-")[0] / "progress.jsonl"
            if status["state"] == "RUNNING" and status["elapsed_seconds"] > 3600:
                age = (
                    time.time() - progress.stat().st_mtime
                    if progress.exists()
                    else float("inf")
                )
                if age > 3600:
                    alerts.append(
                        f"{name}: no progress record for over an hour; inspect job {job['id']}."
                    )
        active_gpus = sum(
            job["gpus"]
            for job in gpu_jobs.values()
            if states.get(job["id"], {}).get("state") in ("RUNNING", "COMPLETING")
        )
        if active_gpus > 8:
            alerts.append(
                f"Concurrency exceeded: {active_gpus} GPUs. Investigate immediately."
            )
        snapshot = {
            "time_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "active_gpus": active_gpus,
            "states": states,
            "alerts": alerts,
            "policy": "Read-only monitoring; failures require review, no automatic retries or next wave",
        }
        temp = root / "monitor.tmp"
        temp.write_text(json.dumps(snapshot, indent=2))
        temp.replace(root / "monitor.json")
        with (root / "monitor.jsonl").open("a") as file:
            file.write(json.dumps(snapshot) + "\n")
        print(json.dumps(snapshot), flush=True)
        if alerts:
            (root / "needs_attention.md").write_text(
                "# Wave 1 needs attention\n\n"
                + "\n".join("- " + alert for alert in alerts)
                + "\n"
            )
        if len(states) == len(gpu_jobs) and all(
            s["state"] in TERMINAL for s in states.values()
        ):
            return
        time.sleep(interval)
    raise TimeoutError("Monitor cap reached; inspect remaining jobs and report state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    watch(args.root)
