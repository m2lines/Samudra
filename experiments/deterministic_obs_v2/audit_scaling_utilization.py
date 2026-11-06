# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Audit saved scaling telemetry across the complete steady training window."""

import argparse
import csv
import datetime
import json
import math
import statistics
from pathlib import Path
from zoneinfo import ZoneInfo


def audit(receipt, telemetry):
    records = receipt["training_records"]
    steady = [
        row for row in records if row["stage"] == "steady" and row["updates"] >= 25
    ]
    first = next(row for row in steady if row["updates"] == 25)
    seconds_key = next(key for key in first if key.endswith("/seconds"))
    start = first["timestamp"] - first[seconds_key]
    end = max(row["timestamp"] for row in steady)
    duration = end - start
    by_gpu: dict[str, list[tuple[float, float]]] = {}
    for row in csv.reader(telemetry.splitlines()):
        timestamp = (
            datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            .replace(tzinfo=ZoneInfo("America/New_York"))
            .timestamp()
        )
        by_gpu.setdefault(row[1].strip(), []).append((timestamp, float(row[2])))
    expected = {row["uuid"] for row in receipt["memory"]}
    if set(by_gpu) != expected:
        raise ValueError("Telemetry GPU UUIDs differ from the qualification receipt")
    per_gpu = {}
    for uuid, rows in by_gpu.items():
        values = [value for timestamp, value in rows if start <= timestamp <= end]
        coverage = weighted = 0.0
        gaps = []
        for (timestamp, value), (following, _) in zip(rows, rows[1:]):
            if following <= timestamp:
                raise ValueError("Telemetry timestamps are not strictly increasing")
            overlap = max(0.0, min(following, end) - max(timestamp, start))
            coverage += overlap
            weighted += overlap * value
            if overlap:
                gaps.append(following - timestamp)
        if not values or not math.isclose(coverage, duration, abs_tol=0.01):
            raise ValueError("Telemetry does not cover the complete steady window")
        if any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
            raise ValueError("GPU utilization contains invalid values")
        if max(gaps) > 2.0:
            raise ValueError("One-second telemetry has a gap longer than two seconds")
        per_gpu[uuid] = {
            "samples": len(values),
            "mean_percent": statistics.mean(values),
            "median_percent": statistics.median(values),
            "time_weighted_mean_percent": weighted / coverage,
            "coverage_seconds": coverage,
            "max_sample_gap_seconds": max(gaps),
        }
    sources: dict[str, dict[str, object]] = {}
    for source in receipt["exposure"]:
        source_rows = [row for row in steady if f"train/{source}/seconds" in row]
        sources[source] = {"updates": len(source_rows)}
        for metric in ("seconds", "data_seconds", "background_read_seconds"):
            values = [row[f"train/{source}/{metric}"] for row in source_rows]
            sources[source][metric] = {
                "min": min(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "max": max(values),
            }
    return {
        "window": f"steady update 25 start through all validation/checkpoint records at update {max(row['updates'] for row in steady)}",
        "start_unix": start,
        "end_unix": end,
        "duration_seconds": duration,
        "per_gpu": per_gpu,
        "source_timing": sources,
        "passed": all(
            row["mean_percent"] > 50 and row["time_weighted_mean_percent"] > 50
            for row in per_gpu.values()
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("telemetry", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(json.loads(args.receipt.read_text()), args.telemetry.read_text()),
            indent=2,
        )
    )
