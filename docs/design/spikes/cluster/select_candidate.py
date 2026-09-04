# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Select a non-dominated candidate for the next tournament phase."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern")
    parser.add_argument("output", type=Path)
    parser.add_argument("--storage-factor", type=float, default=1.10)
    args = parser.parse_args()

    paths = sorted(Path(path) for path in glob.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"no results matched {args.pattern}")
    results = [json.loads(path.read_text()) for path in paths]
    if not all(result.get("exact_validation") for result in results):
        raise AssertionError("one or more candidates failed exact validation")
    minimum_bytes = min(result["physical_bytes"] for result in results)
    eligible = [
        result
        for result in results
        if result["physical_bytes"] <= minimum_bytes * args.storage_factor
    ]
    winner = min(
        eligible,
        key=lambda result: (
            result["independent_median_seconds"] + result["union_median_seconds"],
            sum(result["encode_seconds"].values()),
            result["physical_bytes"],
        ),
    )
    decision = {
        "method": (
            "minimum measured read time among candidates within "
            f"{args.storage_factor:.2f}x of minimum physical bytes"
        ),
        "minimum_physical_bytes": minimum_bytes,
        "eligible": [result["candidate"] for result in eligible],
        "winner": winner,
        "all_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
