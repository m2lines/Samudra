# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize completed A--E layout tournament results without selecting blindly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    candidates: dict[str, Any] = {}
    for name in "ABCDE":
        result_path = args.results / f"candidate-{name}" / "result.json"
        result = json.loads(result_path.read_text())
        trace_path = args.results / f"candidate-{name}" / "trace-v3.json"
        trace = json.loads(trace_path.read_text())
        result["workloads"] = trace["workloads"]
        result["trace_tracker"] = trace["tracker"]
        candidates[name] = result

    workload_names = next(iter(candidates.values()))["workloads"]
    summary = {
        "decision_prior": "A",
        "selection_policy": (
            "Keep A unless a control materially improves representative ready-tensor "
            "latency or transferred bytes without unacceptable general-access cost."
        ),
        "candidates": {
            name: {
                "logical_chunks": result["logical_chunks"],
                "physical_shards": result["physical_shards"],
                "stored_bytes": result["stored_bytes"],
                "physical_data_objects": result["physical_data_objects"],
                "compression_ratio": result["compression_ratio"],
                "encode_seconds": result["encode_seconds"],
                "validation_seconds": result["validation_seconds"],
                "exact_full_validation": result["exact_full_validation"],
                "trace_tracker": result["trace_tracker"],
                "workloads": {
                    workload: {
                        "median_seconds": result["workloads"][workload][
                            "median_seconds"
                        ],
                        "median_data_returned_bytes": sorted(
                            trial["data_returned_bytes"]
                            for trial in result["workloads"][workload]["trials"]
                        )[1],
                        "median_data_calls": sorted(
                            trial["data_calls"]
                            for trial in result["workloads"][workload]["trials"]
                        )[1],
                        "median_unique_data_objects": sorted(
                            trial["unique_data_objects"]
                            for trial in result["workloads"][workload]["trials"]
                        )[1],
                    }
                    for workload in workload_names
                },
            }
            for name, result in candidates.items()
        },
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
