# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Export A--E Python and native reader results to tidy CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    candidate_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for candidate in "ABCDE":
        directory = args.results / f"candidate-{candidate}"
        baseline = json.loads((directory / "result.json").read_text())
        candidate_rows.append(
            {
                "candidate": candidate,
                "logical_chunks": "x".join(map(str, baseline["logical_chunks"])),
                "physical_shards": "x".join(map(str, baseline["physical_shards"])),
                "codec": baseline["codec"],
                "fixture_decoded_bytes": baseline["fixture_decoded_bytes"],
                "stored_bytes": baseline["stored_bytes"],
                "physical_data_bytes": baseline["physical_data_bytes"],
                "stored_files": baseline["stored_files"],
                "physical_data_objects": baseline["physical_data_objects"],
                "compression_ratio": baseline["compression_ratio"],
                "encode_seconds": baseline["encode_seconds"],
            }
        )

        readers: list[tuple[str, dict[str, Any], bool]] = []
        trace = json.loads((directory / "trace-v3.json").read_text())
        readers.append(("zarr-python", trace, False))
        for reader in ("zarrs", "tensorstore"):
            native = json.loads((directory / f"native-{reader}.json").read_text())
            readers.append((reader, native, False))
            validation_rows.append(
                {
                    "candidate": candidate,
                    "reader": reader,
                    "exact_full_validation": native["exact_full_validation"],
                    "validation_seconds": native["validation_seconds"],
                    "versions_json": json.dumps(native["versions"], sort_keys=True),
                }
            )
        validation_rows.append(
            {
                "candidate": candidate,
                "reader": "zarr-python",
                "exact_full_validation": baseline["exact_full_validation"],
                "validation_seconds": baseline["validation_seconds"],
                "versions_json": json.dumps({"zarr": "3.3.0"}),
            }
        )

        for reader, result, byte_metrics_valid in readers:
            for workload, workload_result in result["workloads"].items():
                rows_for_summary = []
                for trial_index, trial in enumerate(workload_result["trials"]):
                    wall_seconds = trial.get("wall_seconds", trial.get("seconds"))
                    row = {
                        "candidate": candidate,
                        "reader": reader,
                        "workload": workload,
                        "trial": trial_index,
                        "wall_seconds": wall_seconds,
                        "process_cpu_seconds": trial.get("process_cpu_seconds", ""),
                        "byte_metrics_valid": byte_metrics_valid,
                        "observed_data_returned_bytes": trial.get(
                            "data_returned_bytes", ""
                        ),
                        "observed_data_calls": trial.get("data_calls", ""),
                        "observed_unique_data_objects": trial.get(
                            "unique_data_objects", ""
                        ),
                    }
                    trial_rows.append(row)
                    rows_for_summary.append(row)
                summary_rows.append(
                    {
                        "candidate": candidate,
                        "reader": reader,
                        "workload": workload,
                        "median_wall_seconds": statistics.median(
                            float(row["wall_seconds"]) for row in rows_for_summary
                        ),
                        "median_process_cpu_seconds": (
                            statistics.median(
                                float(row["process_cpu_seconds"])
                                for row in rows_for_summary
                            )
                            if reader != "zarr-python"
                            else ""
                        ),
                        "byte_metrics_valid": byte_metrics_valid,
                    }
                )

    write_csv(
        args.results / "candidate_summary.csv",
        list(candidate_rows[0]),
        candidate_rows,
    )
    write_csv(
        args.results / "reader_validation.csv",
        list(validation_rows[0]),
        validation_rows,
    )
    write_csv(
        args.results / "reader_workload_trials.csv",
        list(trial_rows[0]),
        trial_rows,
    )
    write_csv(
        args.results / "reader_workload_summary.csv",
        list(summary_rows[0]),
        summary_rows,
    )


if __name__ == "__main__":
    main()
