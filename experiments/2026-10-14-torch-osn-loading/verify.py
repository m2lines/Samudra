# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Verify the archived benchmark records without accessing OSN or Torch."""

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    """Check source fingerprints, raw records, and combined summary arithmetic."""
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    combined = json.loads((ROOT / "combined-summary.json").read_text())
    assert [row["concurrency"] for row in combined] == list(range(1, 11))
    seen_keys = set()
    request_count = 0
    for job, source, levels in [
        ("19803626", "benchmark_initial.py.txt", range(1, 8)),
        ("19803911", "benchmark_recovery.py.txt", range(8, 11)),
    ]:
        directory = ROOT / f"results-{job}"
        provenance = json.loads((directory / "provenance.json").read_text())
        assert (
            hashlib.sha256((ROOT / "as-run" / source).read_bytes()).hexdigest()
            == provenance["script_sha256"]
        )
        assert (
            hashlib.sha256((directory / "metadata.json").read_bytes()).hexdigest()
            == provenance["metadata_sha256"]
        )
        with gzip.open(directory / "requests.csv.gz", "rt", newline="") as file:
            requests = list(csv.DictReader(file))
        archived = json.loads((directory / "summary.json").read_text())
        for level in levels:
            summary = next(row for row in combined if row["concurrency"] == level)
            original = next(row for row in archived if row["concurrency"] == level)
            assert all(summary[key] == value for key, value in original.items())
            rows = [row for row in requests if int(row["concurrency"]) == level]
            assert len(rows) == summary["requests"]
            assert summary["errors"] == 0 and all(not row["error"] for row in rows)
            assert len({row["worker"] for row in rows}) == level
            assert sum(int(row["wire_bytes"]) for row in rows) == summary["wire_bytes"]
            assert (
                sum(int(row["decoded_bytes"]) for row in rows)
                == summary["decoded_bytes"]
            )
            assert math.isclose(
                max(float(row["finish_s"]) for row in rows), summary["elapsed_s"]
            )
            for kind in ("wire", "decoded"):
                assert math.isclose(
                    summary[f"{kind}_bytes"] / summary["elapsed_s"] / 1e6,
                    summary[f"{kind}_MB_s"],
                )
            for row in rows:
                assert row["key"] not in seen_keys
                seen_keys.add(row["key"])
                assert math.isclose(
                    float(row["load_s"]), float(row["get_s"]) + float(row["decode_s"])
                )
                spec = provenance["variables"][row["key"].split("/")[0]]
                assert int(row["decoded_bytes"]) == math.prod(spec["chunks"]) * 4
            request_count += len(rows)
    assert (
        json.loads((ROOT / "results-19803911" / "COMPLETE.json").read_text())["phases"]
        == 3
    )
    assert request_count == 2851
    print(
        f"Verified ten valid levels, {request_count:,} distinct reads, source hashes, and byte totals."
    )


if __name__ == "__main__":
    main()
