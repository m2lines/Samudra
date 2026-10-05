#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Fail before the observational copy if Torch scratch quota lacks headroom."""

import argparse
import datetime
import json
import re
import subprocess
from pathlib import Path


def check(report, incoming_bytes):
    report = re.sub(r"\x1b\[[0-9;]*m", "", report)
    match = re.search(
        r"^/scratch\s+\S+\s+\S+\s+([\d.]+)([KMGT]B)/\S+\s+([\d.]+)([KMGT]B)",
        report,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError("Cannot parse live scratch quota; refusing automatic copy")
    units = {
        unit: 1000**power for power, unit in enumerate(("B", "KB", "MB", "GB", "TB"))
    }
    allocation = float(match[1]) * units[match[2]]
    used = float(match[3]) * units[match[4]]
    # myquota prints rounded values; subtract two hundredths of its largest unit.
    rounding_guard = 0.02 * max(units[match[2]], units[match[4]])
    free = allocation - used - rounding_guard
    budget, headroom = 150 * 2**30, 128 * 2**30
    if not 0 < incoming_bytes <= 40 * 2**30:
        raise ValueError(
            f"Prepared dataset exceeds the estimated envelope: {incoming_bytes}"
        )
    if free < budget + headroom:
        raise ValueError(
            f"Insufficient scratch quota: conservative free {free}, require {budget + headroom}"
        )
    return {
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # noqa: UP017 - DTN Python 3.9
        "quota_allocation_bytes": allocation,
        "quota_usage_bytes": used,
        "conservative_free_bytes": free,
        "pilot_budget_bytes": budget,
        "additional_headroom_bytes": headroom,
        "incoming_npz_bytes": incoming_bytes,
        "quota_report": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = subprocess.run(
        ["myquota"], check=True, capture_output=True, text=True, timeout=45
    ).stdout
    result = check(report, args.incoming_bytes)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
