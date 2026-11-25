# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Gate: does the control replay reproduce the original run, update by update?

Compares the replay's ``metrics.jsonl`` with the original run's for every update
the replay covers: gradient norm, per-source loss parts and the validation
selection score. Training is deterministic (``use_deterministic_algorithms``), so
the expected difference is zero. A counterfactual is interpretable only if its
control passes; otherwise the replay is a new random trajectory, and the spike
may simply not recur. Exits non-zero on failure and prints the first divergence.
"""

import argparse
import json
import math
import sys
from pathlib import Path

KEYS = ("train/gradient_norm", "snapshot_val/obs/normalized_rmse/channel_mean")
PART_SUFFIXES = ("/loss", "/forecast", "/completion", "/reconstruction")


def by_update(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        # A resumed run can log an update again; the last record is the one kept.
        rows.setdefault(int(record["updates"]), {}).update(record)
    return rows


def compared(record: dict) -> dict[str, float]:
    return {
        key: value
        for key, value in record.items()
        if key in KEYS or (key.startswith("train/") and key.endswith(PART_SUFFIXES))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    replay, original = by_update(args.replay), by_update(args.original)
    worst, first, checked = 0.0, None, 0
    for update in sorted(replay):
        expected = compared(original.get(update, {}))
        for key, value in compared(replay[update]).items():
            if key not in expected:
                continue
            checked += 1
            reference = expected[key]
            error = abs(value - reference) / max(abs(reference), 1e-30)
            worst = max(worst, error)
            if error > args.rtol and first is None:
                first = (update, key, value, reference)
    if checked == 0:
        sys.exit("No overlapping metrics to compare")
    print(
        json.dumps(
            dict(checked=checked, worst_relative_error=worst, first_divergence=first)
        )
    )
    if first is not None or not math.isfinite(worst):
        sys.exit(1)


if __name__ == "__main__":
    main()
