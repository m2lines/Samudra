# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""Audit required observation dates, smoke five days, then prepare bounded tasks.

All conversions remain in quarter_observation_prepare. Audit reads only metadata
and coordinates. Receipts hash source metadata (not entire source payloads), code,
the model grid and every produced shard. Run independent product/year tasks with
bounded scheduler concurrency; this script never submits jobs.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from samudra.experiments import (
    observation_data,
    observation_prepare,
    quarter_observation_prepare,
)

PRODUCTS = {
    "sst": ("oisst", ["sst"]),
    "adt": ("duacs", ["adt", "ugos", "vgos"]),
    "era5": ("era5-surface", list(observation_prepare.FORCING)),
    "iap": ("argo-iap", ["temp", "salt"]),
}
ANNUAL_ORIGINS = ["2015-01-01", "2018-01-01", "2021-01-01"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def receipt(path, value):
    """Immutable, exclusive publication; interrupted work never appears ready."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing changed receipt: {path}")
        return
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def required_calendar():
    months = {
        split: [str(month) for month in pd.period_range(*bounds, freq="M")]
        for split, bounds in observation_data.SPLITS.items()
    }
    daily: set[pd.Timestamp] = set()
    bins = 0
    for split in months.values():
        for month in split:
            starts, ends, _ = observation_data.month_intervals(month)
            bins += len(starts)
            daily.update(pd.date_range(starts[0], ends[-1], inclusive="left", freq="D"))
    annual: set[pd.Timestamp] = set()
    for origin in pd.DatetimeIndex(ANNUAL_ORIGINS):
        annual.update(pd.date_range(origin - pd.Timedelta(days=95), periods=460))
    extra = annual - daily
    daily.update(annual)
    return {
        "months": months,
        "daily_dates": [str(day.date()) for day in sorted(daily)],
        "annual_origins": ANNUAL_ORIGINS,
        "annual_additional_days": len(extra),
        "monthly_bins": bins,
    }


def source_indices(times, product, dates):
    """Validate exact support, retaining OISST noon and ERA ending-hour semantics."""
    times = pd.DatetimeIndex(times)
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError(f"Invalid {product} source timestamps")
    requested = pd.DatetimeIndex(dates)
    if product == "era5":
        expected = pd.DatetimeIndex(
            sorted(
                {
                    hour
                    for day in requested
                    for hour in pd.date_range(day, periods=25, freq="h")
                }
            )
        )
        indices = times.get_indexer(expected)
        if (indices < 0).any():
            raise ValueError(
                f"Missing ERA5 state/ending-hour support: {expected[indices < 0][:3]}"
            )
        return indices
    labels = times.to_period("M") if product == "iap" else times.normalize()
    target = requested.to_period("M") if product == "iap" else requested
    if labels.has_duplicates:
        raise ValueError(
            f"Expected one {product} analysis per {'month' if product == 'iap' else 'day'}"
        )
    indices = labels.get_indexer(target)
    if (indices < 0).any():
        raise ValueError(f"Missing {product} support: {target[indices < 0][:3]}")
    return indices


def inputs(source, grid):
    paths = [Path(grid)]
    if (Path(source) / "COMPLETE.json").exists():
        paths.append(Path(source) / "COMPLETE.json")
    for store, _ in PRODUCTS.values():
        root = Path(source) / f"{store}.zarr"
        paths.append(root / ".zmetadata")
        paths.extend(path for path in [root / "COMPLETE.json"] if path.exists())
    for module in [observation_data, observation_prepare, quarter_observation_prepare]:
        assert module.__file__ is not None
        paths.append(Path(module.__file__))
    paths.append(Path(__file__))
    ready = Path(grid).with_name("GRID_READY.json")
    if ready.exists():
        paths.append(ready)
    return {str(path.resolve()): digest(path) for path in paths}


def storage_estimate(calendar):
    """Logical float32 bytes; NPZ compression and filesystem headroom are separate."""
    days = len(calendar["daily_dates"])
    months = sum(map(len, calendar["months"].values()))
    plane = 720 * 1440 * 4
    interior = 30 * 360 * 720 * 4
    return {
        "daily_values_and_coverage_bytes": days * 24 * plane,
        "native_iap_values_and_ohc_bytes": months * interior,
        "monthly_samples_bytes": calendar["monthly_bins"] * 12 * plane
        + months * interior,
        "annual_samples_bytes": 3 * 92 * 12 * plane,
        "era5_logical_source_bytes": days * 24 * 8 * 721 * 1440 * 4,
        "note": "Logical arrays only; reserve temporary-file/headroom space. Measure compression and seconds/day in smoke before full work.",
    }


def audit(source, grid, output):
    source, grid, output = (Path(value).resolve() for value in [source, grid, output])
    before = inputs(source, grid)
    with np.load(grid) as saved:
        if (len(saved["lat"]), len(saved["lon"])) != (720, 1440):
            raise ValueError("Expected a 720x1440 model grid")
    calendar = required_calendar()
    tasks, sources = [], {}
    monthly = sorted(
        {month for months in calendar["months"].values() for month in months}
    )
    for product, (store, variables) in PRODUCTS.items():
        with xr.open_zarr(source / f"{store}.zarr", chunks=None) as ds:
            times = pd.DatetimeIndex(ds.time.values)
            dates = (
                [f"{month}-01" for month in monthly]
                if product == "iap"
                else calendar["daily_dates"]
            )
            selected = source_indices(times, product, dates)
            for name in variables:
                if name not in ds or not {"time", "lat", "lon"}.issubset(ds[name].dims):
                    raise ValueError(f"Missing/invalid variable {product}/{name}")
            for name in ["lat", "lon"]:
                coord = ds[name].values
                if (
                    coord.ndim != 1
                    or not np.isfinite(coord).all()
                    or len(np.unique(coord)) != len(coord)
                ):
                    raise ValueError(f"Invalid coordinate {product}/{name}")
            sources[product] = {
                "path": str(source / f"{store}.zarr"),
                "first_timestamp": times[0].isoformat(),
                "last_timestamp": times[-1].isoformat(),
                "timestamp_count": len(times),
                "timestamp_sha256": hashlib.sha256(
                    times.to_numpy().tobytes()
                ).hexdigest(),
                "required_source_timestamps": len(selected),
                "shape": [ds.sizes["lat"], ds.sizes["lon"]],
                "variables": variables,
            }
            # IAP sources need not label monthly means on the first of the month.
            task_dates = (
                [str(day.date()) for day in times[selected].normalize()]
                if product == "iap"
                else dates
            )
            years = sorted({pd.Timestamp(day).year for day in task_dates})
            for year in years:
                tasks.append(
                    {
                        "product": product,
                        "year": year,
                        "dates": [
                            day for day in task_dates if pd.Timestamp(day).year == year
                        ],
                    }
                )
    if before != inputs(source, grid):
        raise ValueError("Input provenance changed during audit")
    plan = {
        "schema": 1,
        "source": str(source),
        "grid": str(grid),
        "output": str(output),
        "calendar": calendar,
        "sources": sources,
        "tasks": tasks,
        "input_hashes": before,
        "storage": storage_estimate(calendar),
    }
    receipt(output / "PLAN.json", plan)
    return plan


def check_inputs(plan):
    if inputs(plan["source"], plan["grid"]) != plan["input_hashes"]:
        raise ValueError("Plan inputs changed; audit into a new output directory")


def verify_shard(path, product, date):
    with np.load(path, allow_pickle=False) as saved:
        expected_time = (
            str(pd.Timestamp(date).to_period("M"))
            if product == "iap"
            else pd.Timestamp(date).isoformat()
        )
        if str(saved["time"]) != expected_time:
            raise ValueError(f"Unexpected timestamp: {path}")
        shape = (
            (2, 14, 360, 720)
            if product == "iap"
            else (len(PRODUCTS[product][1]), 720, 1440)
        )
        values = saved["values"]
        if (
            values.shape != shape
            or values.dtype != np.float32
            or np.isinf(values).any()
            or not np.isfinite(values).any()
        ):
            raise ValueError(f"Invalid fields: {path}")
        if product == "era5" and not np.isfinite(values).all():
            raise ValueError(f"Nonfinite forcing would invalidate model inputs: {path}")
        if product == "iap":
            np.testing.assert_array_equal(saved["depth"], observation_prepare.DEPTHS)
            if (
                saved["ohc"].shape != (2, 360, 720)
                or np.isinf(saved["ohc"]).any()
                or not np.isfinite(saved["ohc"]).any()
            ):
                raise ValueError(f"Invalid OHC: {path}")
        else:
            coverage = saved["coverage"]
            if (
                coverage.shape != shape
                or not np.isfinite(coverage).all()
                or (coverage < 0).any()
                or (coverage > 1 + 1e-5).any()
            ):
                raise ValueError(f"Invalid coverage: {path}")
    return {"file": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def run_task(plan, task, root):
    args = argparse.Namespace(
        source=plan["source"],
        grid=plan["grid"],
        output=str(root),
        product=task["product"],
        year=task["year"],
        limit=0,
        selected_dates=task["dates"],
    )
    quarter_observation_prepare.prepare_quarter(args)
    records = []
    for date in task["dates"]:
        label = (
            str(pd.Timestamp(date).to_period("M")) if task["product"] == "iap" else date
        )
        path = root / task["product"] / str(task["year"]) / f"{label}.npz"
        records.append(verify_shard(path, task["product"], date))
    return records


def publish_file(source, root):
    root.mkdir(parents=True, exist_ok=True)
    target = root / Path(source).name
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=root, suffix=".npz") as temporary:
            shutil.copyfile(source, temporary.name)
            try:
                os.link(temporary.name, target)
            except FileExistsError:
                pass  # Another product task may publish this identical grid.
    if digest(source) != digest(target):
        raise ValueError(f"Prepared archive artifact differs: {target}")


def check_receipt(path, plan):
    value = json.loads(Path(path).read_text())
    if value["plan_sha256"] != digest(Path(plan["output"]) / "PLAN.json"):
        raise ValueError("Receipt belongs to a different plan")
    for record in value["records"]:
        if digest(record["file"]) != record["sha256"]:
            raise ValueError(f"Prepared shard changed: {record['file']}")
    return value


def execute(plan, task_id=None):
    check_inputs(plan)
    output = Path(plan["output"])
    target = (
        output / "SMOKE_READY.json"
        if task_id is None
        else output / "receipts" / f"{task_id:03d}.json"
    )
    if task_id is not None:
        check_receipt(output / "SMOKE_READY.json", plan)
    if target.exists():
        return check_receipt(target, plan)
    if task_id is None:
        days = set(plan["calendar"]["daily_dates"][:5])
        tasks = []
        for product in PRODUCTS:
            task = dict(
                next(task for task in plan["tasks"] if task["product"] == product)
            )
            task["dates"] = (
                task["dates"][:1]
                if product == "iap"
                else [day for day in task["dates"] if day in days]
            )
            tasks.append(task)
        root = output / "smoke"
    else:
        if not 0 <= task_id < len(plan["tasks"]):
            raise ValueError("Task id outside plan")
        tasks, root = [plan["tasks"][task_id]], output / "daily"
    publish_file(plan["grid"], root)
    if task_id is not None:
        # Publish the smoke-qualified geometry once, before concurrent IAP years.
        for name in ["iap-grid.npz", "iap-operator.json"]:
            publish_file(output / "smoke" / name, root)
    started = time.monotonic()
    records = [record for task in tasks for record in run_task(plan, task, root)]
    for name in ["grid.npz", "iap-grid.npz", "iap-operator.json"]:
        path = root / name
        records.append(
            {"file": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
        )
    check_inputs(plan)
    result = {
        "plan_sha256": digest(output / "PLAN.json"),
        "records": records,
        "seconds": time.monotonic() - started,
    }
    receipt(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    for name in ["source", "grid", "output"]:
        audit_parser.add_argument(f"--{name}", required=True)
    for name in ["smoke", "task", "verify"]:
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True)
        if name == "task":
            command.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        plan = audit(args.source, args.grid, args.output)
        print(
            json.dumps(
                {
                    "tasks": len(plan["tasks"]),
                    "days": len(plan["calendar"]["daily_dates"]),
                    "months": {
                        key: len(value)
                        for key, value in plan["calendar"]["months"].items()
                    },
                    "storage": plan["storage"],
                },
                indent=2,
            )
        )
        return
    plan = json.loads(Path(args.plan).read_text())
    if args.command == "verify":
        check_inputs(plan)
        check_receipt(Path(plan["output"]) / "SMOKE_READY.json", plan)
        for index in range(len(plan["tasks"])):
            check_receipt(Path(plan["output"]) / "receipts" / f"{index:03d}.json", plan)
        receipt(
            Path(plan["output"]) / "PREPARED_READY.json",
            {
                "plan_sha256": digest(args.plan),
                "task_receipts": {
                    f"{index:03d}": digest(
                        Path(plan["output"]) / "receipts" / f"{index:03d}.json"
                    )
                    for index in range(len(plan["tasks"]))
                },
            },
        )
    else:
        execute(plan, args.task_id if args.command == "task" else None)


if __name__ == "__main__":
    main()
