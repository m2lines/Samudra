#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Submit the approved one-seed wave; bound concurrency by explicit dependencies."""

import argparse
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "pilot_submission", Path(__file__).with_name("submit_observation_pilot.py")
)
assert spec is not None and spec.loader is not None
submission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(submission)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--root", default="/scratch/jr7309/runs/2026-09-23-observation-budget"
    )
    parser.add_argument("--gpu-type", choices=["rtx6000", "h200"], default="rtx6000")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, str] = {}
    # 81 allocation-hours requested; 19 of the approved 100 reserved for recovery.
    specs = [
        ("calibration", 10, ()),
        ("om4", 2, ("calibration",)),
        ("obs0", 15, ("calibration",)),
        ("random", 27, ("calibration",)),
        ("obs25", 12, ("om4",)),
        ("obs50", 9, ("om4",)),
        ("obs75", 6, ("om4", "obs0")),
    ]
    for name, hours, deps in specs:
        jobs[name] = submission.submit(
            args,
            name,
            "samudra.experiments.observation_budget_wave",
            ["--root", str(root), "--stage", name],
            hours,
            [jobs[d] for d in deps],
        )
    (root / "DAG_SUBMITTED.json").write_text(json.dumps(jobs, indent=2) + "\n")


if __name__ == "__main__":
    main()
