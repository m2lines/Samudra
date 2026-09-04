# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Rank constant-volume time/space shard envelopes across workload classes."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results: list[dict[str, Any]] = [
        json.loads(Path(path).read_text()) for path in sorted(glob.glob(args.pattern))
    ]
    if len(results) != 4 or not all(item["exact_validation"] for item in results):
        raise AssertionError("expected four exact candidates")
    metrics = (
        "independent_median_seconds",
        "union_median_seconds",
        "point_series_median_seconds",
        "crop_series_median_seconds",
    )
    minima = {metric: min(item[metric] for item in results) for metric in metrics}

    def score(item: dict[str, Any]) -> float:
        ratios = [item[metric] / minima[metric] for metric in metrics]
        return math.exp(sum(math.log(value) for value in ratios) / len(ratios))

    ranked = sorted(results, key=lambda item: (score(item), item["physical_objects"]))
    decision = {
        "method": "equal-workload geometric-mean slowdown",
        "metrics": metrics,
        "ranked": [
            {
                "candidate": item["candidate"],
                "score": score(item),
                "physical_objects": item["physical_objects"],
                "physical_bytes": item["physical_bytes"],
            }
            for item in ranked
        ],
        "winner": ranked[0],
        "all_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
