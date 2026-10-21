#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare existing product stores for the config-free observation baseline.

Stages (no provider downloads):
  grid --om4 /data/OM4.zarr --output /prepared/grid.npz
  daily --source /products --grid /prepared/grid.npz --output /prepared/daily --product sst --year 1993
  months --root /prepared
Repeat daily for sst/adt/era5/iap and each required year (1993--2014 for training
and annual validation). Keep the daily shards: replay must recompute intervals
on the ORIGINAL forecast-origin lattice, not concatenate rebased monthly files.
Finally run prepare_observation_normalization.py /prepared.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from samudra.observations.archive import ObservationArchive
from samudra.observations.grid import export_grid
from samudra.observations.prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    grid = commands.add_parser("grid")
    grid.add_argument("--om4", type=Path, required=True)
    grid.add_argument("--output", type=Path, required=True)
    daily = commands.add_parser("daily")
    daily.add_argument("--source", type=Path, required=True)
    daily.add_argument("--grid", type=Path, required=True)
    daily.add_argument("--output", type=Path, required=True)
    daily.add_argument(
        "--product", choices=["sst", "adt", "era5", "iap"], required=True
    )
    daily.add_argument("--year", type=int, required=True)
    daily.set_defaults(limit=0)
    months = commands.add_parser("months")
    months.add_argument("--root", type=Path, required=True)
    months.add_argument(
        "--split", choices=["train", "validation", "test"], action="append"
    )
    args = parser.parse_args()
    if args.stage == "grid":
        export_grid(args.om4, args.output)
    elif args.stage == "daily":
        prepare(args)
    else:
        daily_root = args.root / "daily"
        daily_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.root / "grid.npz", daily_root / "grid.npz")
        archive = ObservationArchive(daily_root)
        splits = args.split or ["train", "validation"]
        for split in splits:
            archive.materialize(split, args.root / split)
        manifests = sorted(daily_root.glob("*/*/manifest.json"))
        (args.root / "COMPLETE.json").write_text(
            json.dumps(
                {
                    "splits": splits,
                    "daily_root": "daily",
                    "grid_sha256": hashlib.sha256(
                        (args.root / "grid.npz").read_bytes()
                    ).hexdigest(),
                    "source_manifests": {
                        str(p.relative_to(args.root)): json.loads(p.read_text())
                        for p in manifests
                    },
                    "normalization": "separate required prepare_observation_normalization.py stage",
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
