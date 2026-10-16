# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Remove only named raw archives after local and publication-evidence checks.

Requires explicit --execute. Prepared stores, inventories and reports persist.
Run within a Grace allocation holding the original preparation/publication locks.
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from ocean_preprocessing.obs_preprocessing import full_range as fr


def remote_bytes(path):
    return subprocess.run(
        ["rclone", "cat", path, "--contimeout=15s", "--timeout=60s", "--retries=2"],
        check=True,
        capture_output=True,
        timeout=150,
    ).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--dest", default="nyu-osn:emulators/jr7309/data/full_range")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(args.work_root).resolve(strict=True)
    evidence = json.loads(
        (root / "reports/publication-completion-20260918.json").read_text()
    )
    records = {r["product"]: r for r in evidence["products"]}
    report_path = root / "reports/source-cleanup-20260920.json"
    report = (
        json.loads(report_path.read_text())
        if report_path.exists()
        else dict(products={})
    )
    for product in ("duacs", "oisst", "argo-iap"):
        raw = root / "raw" / product
        store = root / "prepared" / f"{product}.zarr"
        if raw.is_symlink() or store.is_symlink():
            raise ValueError("Refusing symlinked raw/store paths")
        if not raw.exists():
            if report["products"].get(product, {}).get("removed"):
                continue
            raise FileNotFoundError(f"Raw path absent without cleanup receipt: {raw}")
        manifest = root / "manifests" / f"{product}.json"
        plan = json.loads(manifest.read_text())
        local_validation = fr.validate(str(manifest), str(store))
        success = json.loads(remote_bytes(f"{args.dest}/{product}.SUCCESS.json"))
        if success != records[product]["success"] or success[
            "inventory_sha256"
        ] != fr._digest(plan):
            raise ValueError(
                "Remote publication evidence changed; inspect before cleanup"
            )
        if (
            remote_bytes(f"{args.dest}/{product}.inventory.json")
            != manifest.read_bytes()
        ):
            raise ValueError("Remote inventory mismatch")
        if (
            remote_bytes(f"{args.dest}/{product}.zarr/.zmetadata")
            != (store / ".zmetadata").read_bytes()
        ):
            raise ValueError("Remote consolidated metadata mismatch")
        count = size = 0
        for parent, _, files in os.walk(raw):
            for name in files:
                count += 1
                size += (Path(parent) / name).lstat().st_size
        report["products"][product] = dict(
            raw_path=str(raw),
            prepared_path=str(store),
            bytes=size,
            files=count,
            verified_at=datetime.datetime.now(datetime.UTC).isoformat(),
            inventory_sha256=fr._digest(plan),
            validation=local_validation,
            publication_success=success,
            remote_metadata_sha256=hashlib.sha256(
                (store / ".zmetadata").read_bytes()
            ).hexdigest(),
            removed=False,
        )
        fr._atomic_json(report_path, report)
        print(
            f"{product}: verified prepared data and published evidence; raw bytes={size}",
            flush=True,
        )
        if args.execute:
            shutil.rmtree(raw)
            report["products"][product]["removed"] = True
            report["products"][product]["removed_at"] = datetime.datetime.now(
                datetime.UTC
            ).isoformat()
            fr._atomic_json(report_path, report)
            print(f"{product}: removed raw; retained {store}", flush=True)


if __name__ == "__main__":
    main()
