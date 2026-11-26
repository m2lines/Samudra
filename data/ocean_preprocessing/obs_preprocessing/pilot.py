# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Bounded real-data pilot using an existing frozen upstream inventory."""

import argparse
import copy
import resource
import time
from pathlib import Path

import pandas as pd
import xarray as xr

from ocean_preprocessing.obs_preprocessing import full_range as full


def pilot_inventory(parent: dict) -> dict:
    """Use 48 daily maps or two months, wholly inside the parent's coverage."""
    plan = copy.deepcopy(parent)
    count = 2 if parent["product"] == "argo-iap" else 48
    dates = pd.date_range("2023-01-01", periods=count, freq=parent["frequency"])
    if dates[0] < pd.Timestamp(parent["start_date"]) or dates[-1] > pd.Timestamp(
        parent["end_date"]
    ):
        raise ValueError("Pilot interval is outside the frozen upstream inventory")
    plan.update(
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        time_count=count,
        pilot_parent_inventory_sha256=full._digest(parent),
        purpose="bounded native-cadence conversion pilot; not the full archive",
    )
    if "records" in plan:
        plan["records"] = [
            r
            for r in plan["records"]
            if plan["start_date"] <= r["date"] <= plan["end_date"]
        ]
    return plan


def run(product: str, inventory_dir: Path, root: Path, threads: int) -> dict:
    parent = full._read_manifest(str(inventory_dir / f"{product}.json"))
    plan = pilot_inventory(parent)
    manifest = root / "manifests" / f"{product}.json"
    if manifest.exists():
        if full._read_manifest(str(manifest)) != plan:
            raise ValueError("Existing pilot inventory differs; use a new pilot root")
    else:
        full._atomic_json(manifest, plan)
    full._read_manifest(str(manifest))
    start = time.monotonic()
    full.download(str(manifest), str(root / "raw"), workers=4)
    downloaded = time.monotonic()
    full.prepare(str(manifest), str(root / "raw"), str(root / "prepared"), threads)
    prepared = time.monotonic()
    store = root / "prepared" / f"{product}.zarr"
    result = full.validate(str(manifest), str(store))
    dates = pd.date_range(plan["start_date"], plan["end_date"], freq=plan["frequency"])
    decoded_bytes = 0
    # Compare every pilot value and mask, one block at a time. This compares
    # decoded, coordinate-standardized raw inputs with the stored output.
    with xr.open_zarr(store, consolidated=True) as actual:
        for offset in range(0, len(dates), full.BLOCK):
            stop = min(offset + full.BLOCK, len(dates))
            with full._open_block(
                plan, root / "raw" / product, dates[offset:stop]
            ) as src:
                expected = src.load()
                observed = actual.isel(time=slice(offset, stop)).load()
                xr.testing.assert_equal(observed, expected)
                for name in expected.data_vars:
                    if observed[name].attrs.get("units") != expected[name].attrs.get(
                        "units"
                    ):
                        raise ValueError(f"Units changed for {name}")
                    decoded_bytes += expected[name].nbytes
    result.update(
        download_seconds=downloaded - start,
        prepare_seconds=prepared - downloaded,
        verification_seconds=time.monotonic() - prepared,
        decoded_field_bytes=decoded_bytes,
        store_bytes=sum(p.stat().st_size for p in store.rglob("*") if p.is_file()),
        max_rss_process_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        max_rss_child_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        comparison="all pilot values, NaN masks, coordinates and units checked",
        store=str(store),
    )
    full._atomic_json(root / "reports" / f"{product}.json", result)
    print(result, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--product", required=True, choices=full.PRODUCTS)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.product, args.inventory_dir, args.root, args.threads)


if __name__ == "__main__":
    main()
