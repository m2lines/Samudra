#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize recorded task gradients without running or modifying a model."""

import argparse
import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(events):
    rows = []
    for phase, lower, upper in (
        ("all", 0, 4000),
        ("early", 0, 1000),
        ("late", 3000, 4000),
    ):
        for task in ("observation", "global", "patch"):
            selected = [
                event
                for event in events
                if lower < event["global_step"] <= upper
                and event.get("source_extent", event["task"]) == task
            ]
            if not selected:
                continue
            norms = [event["gradient_norm"] for event in selected]
            if any(not math.isfinite(norm) or norm < 0 for norm in norms):
                raise ValueError("Invalid recorded gradient norm")
            rows.append(
                dict(
                    phase=phase,
                    task=task,
                    updates=len(selected),
                    gradient_norm_median=statistics.median(norms),
                    gradient_norm_mean=statistics.mean(norms),
                    gradient_norm_max=max(norms),
                    clipping_fraction=sum(norm > 1 for norm in norms) / len(norms),
                    mean_clip_multiplier=statistics.mean(
                        min(1.0, 1.0 / (norm + 1e-6)) for norm in norms
                    ),
                    learning_rates=sorted({event["lr"] for event in selected}),
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        collected_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        script_sha256=digest(Path(__file__)),
        protocol="Recorded total-model gradient norms after accumulation and loss weighting, before clipping at 1.0. Adam moments and actual parameter displacements are not measured. Phases use schedule slots, not wall time or observation exposure.",
        arms={},
    )
    for run in args.runs:
        if run.name in result["arms"]:
            raise ValueError("Duplicate model name")
        complete = json.loads((run / "TRAIN_COMPLETE.json").read_text())
        if complete["global_step"] != 4000:
            raise ValueError("Incomplete training budget")
        path = run / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines() if line]
        updates = [event for event in events if event.get("event") == "joint_train"]
        if len({event["global_step"] for event in updates}) != len(updates):
            raise ValueError("Duplicate update records require explicit retry handling")
        result["arms"][run.name] = dict(
            root=str(run),
            events_sha256=digest(path),
            manifest_sha256=digest(run / "manifest.json"),
            training_complete=complete,
            actual_updates=len(updates),
            summaries=summarize(updates),
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
