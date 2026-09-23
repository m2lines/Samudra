#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only, timestamped evidence capture for the observational D pilot."""

import argparse
import datetime
import json
import subprocess
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=45)
    return {
        "command": arguments,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="/scratch/jr7309/runs/2026-09-22-observation-D"
    )
    parser.add_argument("--data", default="/scratch/jr7309/data/obs-d-pilot")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-contract")
    args = parser.parse_args()
    root, data = Path(args.root), Path(args.data)
    result = {
        "snapshot_started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "root": str(root),
        "data_ready": read_json(data / "DATA_READY.json"),
        "materialization": read_json(data / "COMPLETE.json"),
        "source_contract": read_json(
            Path(args.source_contract)
            if args.source_contract
            else root / "qualification-h200/QUALIFIED.json"
        ),
        "submissions": {
            p.stem: read_json(p) for p in sorted(root.glob("*-submission.json"))
        },
        "arms": {},
        "evaluations": {},
    }
    controller = root / "controller.pid"
    if controller.exists():
        result["controller"] = command(
            ["ps", "-p", controller.read_text().strip(), "-o", "pid,etime,stat,args"]
        )
    jobs = sorted({s["job"] for s in result["submissions"].values()})
    if jobs:
        result["scheduler"] = command(
            ["squeue", "-j", ",".join(jobs), "-o", "%.18i %.9T %.12M %.40R"]
        )
        result["accounting"] = command(
            [
                "sacct",
                "-j",
                ",".join(jobs),
                "--parsable2",
                "--duplicates",
                "--format=JobID,State,ExitCode,Elapsed,Start,End,AllocTRES,NodeList",
            ]
        )
    for name in ("fitting", "primary", "adapter-only", "scratch-fitting", "scratch"):
        directory = root / name
        records = {
            key: read_json(directory / (key + ".json"))
            for key in (
                "manifest",
                "QUALIFIED",
                "TRAIN_COMPLETE",
                "best",
                "baseline-validation",
                "selection-reference",
                "adapter-complete",
                "reconstruction-complete",
                "joint-complete",
            )
        }
        events = directory / "events.jsonl"
        if events.exists():
            # A writer may currently be appending the final line. Keep only
            # newline-terminated records; a malformed complete record is an error.
            lines = events.read_text().splitlines(keepends=True)
            complete = [json.loads(line) for line in lines if line.endswith("\n")]
            records["events_count"] = len(complete)
            records["recent_events"] = complete[-20:]
            records["validation_events"] = [
                event for event in complete if event["event"] == "validation"
            ]
        result["arms"][name] = records
    for directory in sorted(root.glob("*-evaluation")):
        result["evaluations"][directory.name] = {
            path.stem: read_json(path) for path in sorted(directory.glob("*.json"))
        }
    result["snapshot_finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Snapshot names are immutable so a later capture cannot change cited evidence.
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
