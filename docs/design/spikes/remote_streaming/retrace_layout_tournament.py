# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Rerun A--E workload traces with synchronous and asynchronous interception."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr  # type: ignore[import-untyped]
from layout_tournament import TrackingStore, run_workloads
from zarr.storage import LocalStore  # type: ignore[import-untyped]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("candidate_store", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    fixture = np.load(args.fixture, mmap_mode="r")
    store = TrackingStore(LocalStore(args.candidate_store, read_only=True))
    array = zarr.open_group(store=store, mode="r")["Theta"]
    result = {
        "tracker": "sync-and-async-v3-shared-clone-recorder",
        "candidate_store": str(args.candidate_store.resolve()),
        "workloads": run_workloads(store, array, fixture),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
