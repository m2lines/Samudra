#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""One read-only status/accounting snapshot; does not submit or cancel jobs."""

import argparse
import datetime
import json
import subprocess
from pathlib import Path
from typing import Any


def inspect(roots):
    submissions: list[dict[str, Any]] = []
    for root in roots:
        submissions.extend(
            json.loads(p.read_text()) for p in Path(root).glob("*-submission.json")
        )
    ids = ",".join(s["id"] for s in submissions)
    if not ids:
        return {"jobs": [], "allocated_gpu_hours": 0}
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "--duplicates",
            "-n",
            "-P",
            "-j",
            ids,
            "--format=JobIDRaw,State%40,ElapsedRaw,AllocTRES%100,NodeList%40,ExitCode,Start,End",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    states = {}
    hours = 0.0
    for line in result.stdout.splitlines():
        row = line.split("|")
        if len(row) < 8:
            continue
        gpu = next(
            (
                int(item.split("=")[1])
                for item in row[3].split(",")
                if item.startswith("gres/gpu=")
            ),
            0,
        )
        hours += gpu * int(row[2] or 0) / 3600
        states[row[0]] = dict(
            state=row[1],
            seconds=int(row[2] or 0),
            gpus=gpu,
            nodes=row[4],
            exit_code=row[5],
            start=row[6],
            end=row[7],
        )
    jobs = []
    for submission in submissions:
        args = submission["arguments"]
        out = Path(args["root"]) / args["name"]
        log = out / "progress.jsonl"
        metrics = []
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 100000))
                lines = stream.read().decode(errors="replace").splitlines()
            for line in lines:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        last = metrics[-1] if metrics else {}
        validations = [m for m in metrics if m.get("event") == "validation"]
        allowed = (
            "event",
            "phase",
            "step",
            "loss_rank0",
            "elapsed_seconds",
            "best_ts_mse",
            "ts_mse",
            "reconstruction_ts_mse",
            "full_mse",
            "bad_checks",
            "time_utc",
            "peak_gpu_gib",
            "host_peak_gib_rank0",
        )
        jobs.append(
            {
                "id": submission["id"],
                "name": args["name"],
                "arm": args["arm"],
                "stage": args["stage"],
                **states.get(submission["id"], {}),
                "latest": {k: v for k, v in last.items() if k in allowed},
                "validation": {
                    k: v
                    for k, v in (validations[-1] if validations else {}).items()
                    if k in allowed
                },
                "train_complete": (out / "TRAIN_COMPLETE.json").exists(),
                "evaluation_complete": (out / "EVAL_COMPLETE.json").exists(),
                "complete": (out / "COMPLETE.json").exists(),
                "exceptions": [p.name for p in out.glob("exception-rank*.log")],
            }
        )
    queue = subprocess.run(
        ["squeue", "-h", "-j", ids, "-o", "%i|%T|%R"],
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        "time_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "allocated_gpu_hours": hours,
        "jobs": jobs,
        "queue": queue.stdout.splitlines(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    print(json.dumps(inspect(args.roots), indent=2))


if __name__ == "__main__":
    main()
