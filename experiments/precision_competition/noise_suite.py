# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed manifests for the auxiliary-noise intervention."""

import argparse
import csv
import json
from pathlib import Path

from .benchmark import atomic_json
from .noise import SETTINGS


def make(root):
    old = "/scratch/jr7309/precision-followup-20261108-ae60e6ee3/qualification"  # pragma: allowlist secret
    rows = list(
        csv.DictReader(
            Path(__file__)
            .with_name("followup_results")
            .joinpath("milestones.csv")
            .open()
        )
    )
    prepare = []
    for row in rows:
        if float(row["lr"]) != 3e-4 or row["schedule"] != "constant":
            continue
        ident = f"seed{row['seed']}-{row['tag']}"
        prepare.append(
            {
                "id": ident,
                "stage": "prepare",
                "parent": f"{old}/{row['id']}/{row['tag']}.pt",
                "sha256": row["checkpoint_sha256"],
            }
        )
    assert len(prepare) == 6
    manifests = {"prepare": prepare, "perturb": [], "joint": []}
    for parent in prepare:
        for index, (lr, batch) in enumerate(SETTINGS):
            base = {
                "id": parent["id"] + f"-setting{index}",
                "prepared": str(root / "prepare" / parent["id"] / "prepared.pt"),
                "lr": lr,
                "batch": batch,
            }
            manifests["perturb"].append({**base, "stage": "perturb"})
            if "bits-6" in parent["id"]:
                manifests["joint"].append({**base, "stage": "joint"})
    return manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, entries in make(args.root).items():
        atomic_json(args.output / f"{name}.json", entries)
        print(json.dumps({"stage": name, "bundles": len(entries)}))


if __name__ == "__main__":
    main()
