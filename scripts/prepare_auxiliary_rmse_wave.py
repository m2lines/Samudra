#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Audit transferred auxiliary targets and the unchanged observation cohort."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from scripts.verify_extent_observations import main as verify_observations


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    paths = json.loads((root / "paths.json").read_text())
    obs = Path(paths["observations"])
    obs.mkdir(exist_ok=True)
    source = Path(paths["observation_source"])
    for name in [
        "grid.npz",
        "statistics.npz",
        "SHA256SUMS",
        "train",
        "validation",
        "test",
    ]:
        target = obs / name
        if not target.exists():
            target.symlink_to(source / name)
        if target.resolve() != (source / name).resolve():
            raise ValueError("Observation source changed")
    sys.argv = ["verify", str(obs)]
    verify_observations()
    for filename, expected in paths["observation_hashes"].items():
        if digest(obs / filename) != expected:
            raise ValueError("Observation contract changed: " + filename)
    if digest(root / "selection-reference.json") != paths["selection_reference_sha256"]:
        raise ValueError("RMSE reference changed")
    receipt = root / "target-source-hashes.json"
    if digest(receipt) != paths["target_source_hashes_sha256"]:
        raise ValueError("Target transfer receipt changed")
    entries = json.loads(receipt.read_text())
    cache = Path(paths["auxiliary_cache"])
    for name, expected in entries.items():
        path = cache / name
        if (
            path.stat().st_size != expected["bytes"]
            or digest(path) != expected["sha256"]
        ):
            raise ValueError("Target file differs from Beta source: " + name)
    actual_files = {str(p.relative_to(cache)) for p in cache.rglob("*") if p.is_file()}
    if actual_files != set(entries):
        raise ValueError("Unexpected target-cache file inventory")
    record = dict(
        source_files_verified=len(entries),
        target_bytes=sum(item["bytes"] for item in entries.values()),
        target_source_hashes_sha256=digest(receipt),
        observation_hashes=paths["observation_hashes"],
        selection_reference_sha256=paths["selection_reference_sha256"],
        producer=paths["producer"],
    )
    (root / "DATA_AUDITED.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
