#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Reverify every prepared observation sample against its immutable manifest."""

import concurrent.futures
import datetime
import hashlib
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    entries = [line.split() for line in (root / "SHA256SUMS").read_text().splitlines()]
    if len(entries) != 350:
        raise ValueError("Expected 350 exact prepared NPZ files")

    def verify(entry):
        expected, name = entry
        with (root / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Prepared observation checksum differs: {name}")
        return name

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(verify, entries))
    for split, count in (("train", 243), ("validation", 9), ("test", 96)):
        if sum(name.startswith(split + "/") for name in checked) != count:
            raise ValueError(f"Wrong cohort count: {split}")
    record = dict(
        verification="all 350 NPZ files checked against source SHA256SUMS",
        source_sha256sums=hashlib.sha256(
            (root / "SHA256SUMS").read_bytes()
        ).hexdigest(),
        verified_at=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    temporary = root / "DATA_READY.extent.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(root / "DATA_READY.json")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
