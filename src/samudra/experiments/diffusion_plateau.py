# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Describe validation progress without treating a monitoring heuristic as convergence."""

import argparse
import json
import math
from pathlib import Path


def assess(parent, continuation, *, interval=500, checks=3, tolerance=0.01):
    """Compare the latest contiguous scheduled window with its preceding best."""
    records = sorted(parent + continuation)
    if len({step for step, _ in records}) != len(records):
        raise ValueError("Duplicate validation steps")
    if any(
        step < 0 or not math.isfinite(score) or score <= 0 for step, score in records
    ):
        raise ValueError("Invalid validation record")
    scheduled = sorted(
        (step, score) for step, score in continuation if step % interval == 0
    )
    result = dict(
        records=records,
        best=min((score for _, score in records), default=None),
        provisional_plateau=False,
        reason="Insufficient consecutive scheduled checks",
    )
    if len(scheduled) < checks:
        return result
    window = scheduled[-checks:]
    if any(b[0] - a[0] != interval for a, b in zip(window, window[1:])):
        return result
    before = [score for step, score in records if step < window[0][0]]
    if not before:
        return result
    reference = min(before)
    improvement = (reference - min(score for _, score in window)) / reference
    return dict(
        result,
        window_steps=[step for step, _ in window],
        preceding_best=reference,
        fractional_improvement=improvement,
        provisional_plateau=improvement < tolerance,
        reason="Less than 1% improvement over three scheduled checks; not proof of convergence"
        if improvement < tolerance
        else "Material improvement within the latest window",
    )


def read(root):
    result = []
    for path in root.glob("validation-*.json"):
        record = json.loads(path.read_text())
        if path.stem != f"validation-{record['step']}":
            raise ValueError("Validation filename and step differ")
        result.append((record["step"], record["score"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assess(read(args.parent), read(args.continuation)), indent=2))


if __name__ == "__main__":
    main()
