# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Rank schema candidates across training and general scientific reads."""

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
    parser.add_argument("--storage-factor", type=float, default=1.15)
    args = parser.parse_args()

    results: list[dict[str, Any]] = [
        json.loads(Path(path).read_text()) for path in sorted(glob.glob(args.pattern))
    ]
    if not results:
        raise FileNotFoundError(f"no results matched {args.pattern}")
    if not all(result.get("exact_validation") for result in results):
        raise AssertionError("one or more candidates failed exact validation")

    metrics = (
        "independent_median_seconds",
        "union_median_seconds",
        "single_depth_512x384_seconds",
        "seventeen_depth_300x500_seconds",
        "all_depth_300x500_seconds",
    )
    minimum_bytes = min(result["physical_bytes"] for result in results)
    eligible = [
        result
        for result in results
        if result["physical_bytes"] <= minimum_bytes * args.storage_factor
    ]
    minima = {
        metric: min(
            result.get("general_read_seconds", {}).get(metric, result.get(metric))
            for result in eligible
        )
        for metric in metrics
    }

    def metric_value(result: dict[str, Any], metric: str) -> float:
        return result.get("general_read_seconds", {}).get(metric, result.get(metric))

    def balanced_score(result: dict[str, Any]) -> float:
        # Equal weight per workload class; geometric mean avoids domination by
        # the metrics with the largest absolute wall times.
        ratios = [metric_value(result, metric) / minima[metric] for metric in metrics]
        return math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))

    ranked = sorted(
        eligible, key=lambda result: (balanced_score(result), result["physical_bytes"])
    )
    decision = {
        "method": (
            "lowest equal-workload geometric-mean slowdown among candidates "
            f"within {args.storage_factor:.2f}x of minimum physical bytes"
        ),
        "metrics": metrics,
        "minimum_physical_bytes": minimum_bytes,
        "ranked": [
            {
                "candidate": result["candidate"],
                "depth_inner": result["depth_inner"],
                "inner": result["inner"],
                "balanced_score": balanced_score(result),
                "physical_bytes": result["physical_bytes"],
            }
            for result in ranked
        ],
        "winner": ranked[0],
        "all_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
