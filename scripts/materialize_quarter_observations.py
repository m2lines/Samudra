#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize fixed native-quarter cohorts without submitting jobs.

Order: plan; monthly --worker 0/1/2/3 (87 disjoint months each); finalize.
Then annual --origin 2015-01-01/2018-01-01/2021-01-01; verify-annual.
Every command takes --plan except plan itself. Run from one immutable source
archive/runtime; prepared daily data must remain read-only after PREPARED_READY.
Invoke as ``python -m scripts.materialize_quarter_observations`` from its archive.

Monthly arrays and training statistics use the existing ObservationArchive.sample
and fit_statistics unchanged. Only the archive's day cache is bounded to 15
entries (about 0.25 GiB for five days of each product). A monthly sample still
occupies about 1.3 GiB, and fit_statistics retains its existing several-GiB
working set. Annual construction/audit never stacks the 92-bin sequence; it
writes one bin at a time. The existing evaluation read_origin still stacks it.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from samudra.experiments import observation_data, observation_prepare
from samudra.experiments.observation_data import (
    SPLITS,
    ObservationArchive,
    fit_statistics,
    month_intervals,
)
from scripts.prepare_quarter_observations import ANNUAL_ORIGINS, required_calendar

STATIC = ("grid.npz", "iap-grid.npz", "iap-operator.json")
WORKERS = 4


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def immutable_bytes(path, data):
    """Publish atomically without replacing any existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp") as temporary:
        temporary.write(data)
        temporary.flush()
        os.fsync(temporary.fileno())
        try:
            os.link(temporary.name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Concurrent artifact differs: {path}") from None


def receipt(path, record):
    immutable_bytes(
        path, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    )


@contextmanager
def exclusive(path):
    """A crashed worker releases this lock; its incomplete files lack receipts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def code_hashes():
    return {
        "helper": digest(__file__),
        "observation_data": digest(observation_data.__file__),
        "observation_prepare": digest(observation_prepare.__file__),
        "preparation_orchestrator": digest(
            Path(__file__).with_name("prepare_quarter_observations.py")
        ),
    }


def runtime_identity():
    return dict(
        machine=platform.machine(),
        python=platform.python_version(),
        packages={
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scipy", "xarray", "zarr")
        },
    )


def month_jobs():
    return [
        {"split": split, "month": str(month)}
        for split, bounds in SPLITS.items()
        for month in pd.period_range(*bounds, freq="M")
    ]


def worker_jobs(jobs, worker):
    if worker not in range(WORKERS):
        raise ValueError("Worker must be 0, 1, 2, or 3")
    return jobs[worker::WORKERS]


def prepared_contract(prepared):
    """Verify the upstream receipt chain; payloads are checked as consumed."""
    prepared = Path(prepared).resolve()
    ready_path, plan_path = prepared / "PREPARED_READY.json", prepared / "PLAN.json"
    ready, source_plan = (
        json.loads(ready_path.read_text()),
        json.loads(plan_path.read_text()),
    )
    if ready["plan_sha256"] != digest(plan_path):
        raise ValueError("PREPARED_READY plan hash differs")
    if Path(source_plan["output"]).resolve() != prepared:
        raise ValueError("Prepared plan belongs to another directory")
    if source_plan["calendar"] != required_calendar():
        raise ValueError("Prepared calendar differs from fixed monthly/annual cohort")
    for path, expected in source_plan["input_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Prepared input provenance changed: {path}")
    required_receipts = {f"{index:03d}" for index in range(len(source_plan["tasks"]))}
    if set(ready["task_receipts"]) != required_receipts:
        raise ValueError("Prepared task receipt inventory differs")
    daily = prepared / "daily"
    records: dict[str, str] = {}
    for task_id, expected in sorted(ready["task_receipts"].items()):
        path = prepared / "receipts" / f"{task_id}.json"
        if digest(path) != expected:
            raise ValueError(f"Prepared task receipt changed: {task_id}")
        task = json.loads(path.read_text())
        if task["plan_sha256"] != ready["plan_sha256"]:
            raise ValueError("Prepared task belongs to a different plan")
        for entry in task["records"]:
            name = str(Path(entry["file"]).resolve().relative_to(daily))
            if name in records and records[name] != entry["sha256"]:
                raise ValueError(f"Conflicting prepared shard hashes: {name}")
            records[name] = entry["sha256"]
    expected_paths = set(STATIC)
    for date in source_plan["calendar"]["daily_dates"]:
        expected_paths.update(
            f"{product}/{date[:4]}/{date}.npz" for product in ("sst", "adt", "era5")
        )
    for job in month_jobs():
        month = job["month"]
        expected_paths.add(f"iap/{month[:4]}/{month}.npz")
    if set(records) != expected_paths:
        raise ValueError("Prepared shard receipt inventory differs")
    for name in STATIC:
        if digest(daily / name) != records[name]:
            raise ValueError(f"Prepared geometry changed: {name}")
    operator = json.loads((daily / "iap-operator.json").read_text())
    if (
        operator["model_grid_sha256"] != records["grid.npz"]
        or operator["iap_grid_sha256"] != records["iap-grid.npz"]
    ):
        raise ValueError("Prepared operator/grid hashes differ")
    return dict(
        root=str(prepared),
        plan_sha256=digest(plan_path),
        ready_sha256=digest(ready_path),
        records=records,
    )


def copy_verified(source, target, expected):
    if digest(source) != expected:
        raise ValueError(f"Source file changed: {source}")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp") as temporary:
            shutil.copyfile(source, temporary.name)
            if digest(temporary.name) != expected:
                raise ValueError("Copy readback differs")
            try:
                os.link(temporary.name, target)
            except FileExistsError:
                pass
    if digest(target) != expected:
        raise ValueError(f"Existing target differs: {target}")


def make_plan(prepared, output, annual, runtime_contract, source_commit):
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise ValueError("Require a pinned 40-character source commit")
    output, annual = Path(output).resolve(), Path(annual).resolve()
    if (
        output == annual
        or output.is_relative_to(annual)
        or annual.is_relative_to(output)
    ):
        raise ValueError("Monthly and annual output roots must be separate")
    upstream = prepared_contract(prepared)
    if output.is_relative_to(Path(upstream["root"])) or annual.is_relative_to(
        Path(upstream["root"])
    ):
        raise ValueError("Materialized outputs must not modify the prepared archive")
    daily = Path(upstream["root"]) / "daily"
    with np.load(daily / "grid.npz") as grid:
        if grid["mask"].shape != (77, 720, 1440):
            raise ValueError("Expected native OM4 77-channel quarter grid")
    with np.load(daily / "iap-grid.npz") as grid:
        if (len(grid["lat"]), len(grid["lon"])) != (360, 720):
            raise ValueError("Expected native half-degree IAP grid")
    for name in STATIC:
        copy_verified(daily / name, output / name, upstream["records"][name])
    plan = dict(
        schema=1,
        output=str(output),
        annual=str(annual),
        upstream=upstream,
        jobs=month_jobs(),
        workers=WORKERS,
        annual_origins=ANNUAL_ORIGINS,
        code=code_hashes(),
        source_commit=source_commit,
        runtime_contract=str(Path(runtime_contract).resolve()),
        runtime_sha256=digest(runtime_contract),
        runtime_identity=runtime_identity(),
    )
    receipt(output / "MATERIALIZATION_PLAN.json", plan)
    return plan


def check_plan(plan):
    if (
        plan["schema"] != 1
        or plan["workers"] != WORKERS
        or plan["jobs"] != month_jobs()
        or plan["annual_origins"] != ANNUAL_ORIGINS
    ):
        raise ValueError("Materialization cohort differs")
    if (
        plan["code"] != code_hashes()
        or plan["runtime_identity"] != runtime_identity()
        or plan["runtime_sha256"] != digest(plan["runtime_contract"])
    ):
        raise ValueError("Materialization source/runtime changed")
    if prepared_contract(plan["upstream"]["root"]) != plan["upstream"]:
        raise ValueError("Prepared source receipts changed")
    for name in STATIC:
        if digest(Path(plan["output"]) / name) != plan["upstream"]["records"][name]:
            raise ValueError("Materialization geometry changed")


def plan_digest(plan):
    return digest(Path(plan["output"]) / "MATERIALIZATION_PLAN.json")


class BoundedArchive(ObservationArchive):
    """Use the established sample/interval arithmetic with a 15-entry day cache."""

    def __init__(self, root, records):
        super().__init__(root)
        self.records = records
        # Per-instance bounded cache, avoiding the inherited global 128 entries.
        self.day = lru_cache(maxsize=15)(self._verified_day)

    def verify(self, name):
        if name not in self.records or digest(self.root / name) != self.records[name]:
            raise ValueError(f"Prepared payload changed or unreceipted: {name}")

    def _verified_day(self, product, date):
        stamp = pd.Timestamp(date).strftime("%Y-%m-%d")
        self.verify(f"{product}/{stamp[:4]}/{stamp}.npz")
        return ObservationArchive.day.__wrapped__(self, product, date)

    def sample(self, month):
        self.verify(f"iap/{month[:4]}/{month}.npz")
        return super().sample(month)


def validate_sample(sample, month, model_shape, iap_shape):
    starts, _, weights = month_intervals(month)
    shapes = {
        "surface": (len(starts), 2, *model_shape),
        "velocity": (len(starts), 2, *model_shape),
        "atmosphere": (len(starts), 8, *model_shape),
        "interior": (2, 14, *iap_shape),
        "ohc": (2, *iap_shape),
    }
    if set(sample) != {*shapes, "month_weights", "midpoints", "origin"}:
        raise ValueError("Monthly array inventory differs")
    for name, shape in shapes.items():
        values = sample[name]
        if (
            values.shape != shape
            or values.dtype != np.float32
            or np.isinf(values).any()
            or not np.isfinite(values).any()
        ):
            raise ValueError(f"Invalid monthly {name}")
    if not np.isfinite(sample["atmosphere"]).all():
        raise ValueError("Nonfinite monthly forcing")
    np.testing.assert_array_equal(sample["month_weights"], weights)
    np.testing.assert_array_equal(
        sample["midpoints"], [(s + pd.Timedelta(days=2.5)).isoformat() for s in starts]
    )
    if str(sample["origin"]) != pd.Timestamp(month).isoformat():
        raise ValueError("Monthly origin differs")


def save_npz(path, sample):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as saved:
            if set(saved.files) != set(sample):
                raise ValueError(f"Existing sample keys differ: {path}")
            for name, values in sample.items():
                np.testing.assert_array_equal(
                    saved[name], values, err_msg=f"{path}/{name}", strict=True
                )
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp") as temporary:
        saved = Path(temporary.name)
        observation_prepare.save_atomic(saved, **sample)
        # Readback catches array/schema errors before any receipt is published.
        with np.load(saved) as loaded:
            for name, values in sample.items():
                np.testing.assert_array_equal(loaded[name], values, strict=True)
        os.link(saved, path)


def month_receipt(plan, job):
    return Path(plan["output"]) / "receipts" / job["split"] / f"{job['month']}.json"


def check_output(record, root):
    name = Path(record["file"])
    if (
        name.is_absolute()
        or ".." in name.parts
        or digest(Path(root) / name) != record["sha256"]
    ):
        raise ValueError(f"Materialized payload changed: {name}")


def materialize_month(plan, job, archive, model_shape, iap_shape):
    root = Path(plan["output"])
    name = f"{job['split']}/{job['month']}.npz"
    target = month_receipt(plan, job)
    if target.exists():
        record = json.loads(target.read_text())
        if record["plan_sha256"] != plan_digest(plan) or record["file"] != name:
            raise ValueError("Monthly receipt belongs to another plan/month")
        check_output(record, root)
        return record
    sample = archive.sample(job["month"])
    validate_sample(sample, job["month"], model_shape, iap_shape)
    save_npz(root / name, sample)
    record = dict(file=name, sha256=digest(root / name), plan_sha256=plan_digest(plan))
    receipt(target, record)
    return record


def monthly_worker(plan, worker):
    check_plan(plan)
    root = Path(plan["output"])
    with exclusive(root / "receipts" / f"worker-{worker}.lock"):
        archive = BoundedArchive(
            Path(plan["upstream"]["root"]) / "daily", plan["upstream"]["records"]
        )
        with np.load(root / "iap-grid.npz") as grid:
            iap_shape = (len(grid["lat"]), len(grid["lon"]))
        records = []
        for job in worker_jobs(plan["jobs"], worker):
            records.append(
                materialize_month(
                    plan, job, archive, archive.grid["mask"].shape[-2:], iap_shape
                )
            )
            print(
                json.dumps(dict(event="month_complete", worker=worker, **job)),
                flush=True,
            )
        check_plan(plan)
        receipt(
            root / "receipts" / f"worker-{worker}.json",
            dict(plan_sha256=plan_digest(plan), records=records),
        )


def exact_inventory(root, expected):
    actual = {str(path.relative_to(root)) for path in Path(root).rglob("*.npz")}
    if actual != set(expected):
        raise ValueError(
            f"NPZ inventory differs: missing={sorted(set(expected) - actual)[:3]}, extra={sorted(actual - set(expected))[:3]}"
        )


def finalize(plan):
    check_plan(plan)
    root = Path(plan["output"])
    with exclusive(root / "receipts" / "finalize.lock"):
        monthly = {}
        for worker in range(WORKERS):
            record = json.loads(
                (root / "receipts" / f"worker-{worker}.json").read_text()
            )
            names = [
                f"{job['split']}/{job['month']}.npz"
                for job in worker_jobs(plan["jobs"], worker)
            ]
            if (
                record["plan_sha256"] != plan_digest(plan)
                or [r["file"] for r in record["records"]] != names
            ):
                raise ValueError("Worker receipt cohort differs")
            for entry in record["records"]:
                check_output(entry, root)
                monthly[entry["file"]] = entry["sha256"]
        before_stats = {*monthly, "grid.npz", "iap-grid.npz"}
        if (root / "statistics.npz").exists():
            before_stats.add("statistics.npz")
        exact_inventory(root, before_stats)
        statistics_receipt = root / "receipts" / "statistics.json"
        inputs = dict(
            train={
                name: value
                for name, value in monthly.items()
                if name.startswith("train/")
            },
            grid=plan["upstream"]["records"]["grid.npz"],
            iap_grid=plan["upstream"]["records"]["iap-grid.npz"],
            plan_sha256=plan_digest(plan),
        )
        if statistics_receipt.exists():
            saved = json.loads(statistics_receipt.read_text())
            if saved["inputs"] != inputs or saved["sha256"] != digest(
                root / "statistics.npz"
            ):
                raise ValueError("Training statistics provenance changed")
        else:
            with (
                np.load(root / "grid.npz") as grid,
                tempfile.NamedTemporaryFile(dir=root, suffix=".tmp") as temporary,
            ):
                candidate = Path(temporary.name)
                fit_statistics(
                    root / "train",
                    dict(grid),
                    candidate,
                    iap_grid=root / "iap-grid.npz",
                )
                if (root / "statistics.npz").exists():
                    with np.load(candidate) as computed:
                        save_npz(root / "statistics.npz", dict(computed))
                else:
                    os.link(candidate, root / "statistics.npz")
            receipt(
                statistics_receipt,
                dict(inputs=inputs, sha256=digest(root / "statistics.npz")),
            )
        with np.load(root / "statistics.npz") as statistics:
            if int(statistics["training_month_count"]) != 243:
                raise ValueError("Statistics must contain exactly 243 training months")
        expected = {
            **monthly,
            **{
                name: digest(root / name)
                for name in ("grid.npz", "iap-grid.npz", "statistics.npz")
            },
        }
        exact_inventory(root, expected)
        if len(expected) != 351:
            raise ValueError("Expected exactly 351 native-quarter NPZ files")
        manifest = "".join(
            f"{value}  {name}\n" for name, value in sorted(expected.items())
        )
        immutable_bytes(root / "SHA256SUMS", manifest.encode())
        # The worker hash checks above reread all monthly payloads; static files
        # and statistics were reread here. No readiness label substitutes for IO.
        check_plan(plan)
        receipt(
            root / "DATA_READY.json",
            dict(
                verification="all 351 NPZ files checked against source SHA256SUMS",
                source_sha256sums=digest(root / "SHA256SUMS"),
                plan_sha256=plan_digest(plan),
                prepared_ready_sha256=plan["upstream"]["ready_sha256"],
                source_commit=plan["source_commit"],
                runtime_sha256=plan["runtime_sha256"],
            ),
        )


def monthly_ready(plan):
    root = Path(plan["output"])
    ready = json.loads((root / "DATA_READY.json").read_text())
    if (
        ready["verification"] != "all 351 NPZ files checked against source SHA256SUMS"
        or ready["plan_sha256"] != plan_digest(plan)
        or ready["source_sha256sums"] != digest(root / "SHA256SUMS")
    ):
        raise ValueError("Monthly DATA_READY provenance differs")
    return digest(root / "DATA_READY.json")


def annual_bin(archive, start):
    """Identical five-day operations to ObservationArchive.sample, one bin only."""
    end = start + pd.Timedelta(days=5)
    adt = archive.interval("adt", start, end)
    return dict(
        surface=np.concatenate([archive.interval("sst", start, end), adt[:1]]),
        velocity=adt[1:],
        atmosphere=archive.interval("era5", start, end),
        start=np.array(start.isoformat()),
        end=np.array(end.isoformat()),
        midpoint=np.array((start + pd.Timedelta(days=2.5)).isoformat()),
    )


def validate_bin(sample, start, shape):
    if set(sample) != {"surface", "velocity", "atmosphere", "start", "end", "midpoint"}:
        raise ValueError("Annual bin array inventory differs")
    for name, channels in (("surface", 2), ("velocity", 2), ("atmosphere", 8)):
        field = sample[name]
        if (
            field.shape != (channels, *shape)
            or field.dtype != np.float32
            or np.isinf(field).any()
            or not np.isfinite(field).any()
        ):
            raise ValueError(f"Invalid annual {name}")
    if not np.isfinite(sample["atmosphere"]).all():
        raise ValueError("Nonfinite annual forcing")
    for name, delta in (("start", 0), ("end", 5), ("midpoint", 2.5)):
        if str(sample[name]) != (start + pd.Timedelta(days=delta)).isoformat():
            raise ValueError("Annual bin timestamp differs")


def audit_annual_origin(plan, origin, description):
    """Full hash/shape/forcing readback with one bin resident, plus monthly parity."""
    directory = Path(plan["annual"]) / "test" / origin
    if (
        description["origin"] != origin
        or (
            description["history_bins"],
            description["forecast_bins"],
            description["bin_days"],
        )
        != (19, 73, 5)
        or len(description["records"]) != 92
        or description["plan_sha256"] != plan_digest(plan)
        or description["monthly_ready_sha256"] != monthly_ready(plan)
    ):
        raise ValueError("Annual origin contract differs")
    with np.load(Path(plan["output"]) / "grid.npz") as grid:
        shape = (len(grid["lat"]), len(grid["lon"]))
    for index, record in enumerate(description["records"]):
        if record["file"] != f"bin-{index:03d}.npz":
            raise ValueError("Annual bin order differs")
        check_output(record, directory)
        with np.load(directory / record["file"]) as saved:
            validate_bin(
                dict(saved),
                pd.Timestamp(origin) + pd.Timedelta(days=5 * (index - 19)),
                shape,
            )
    months = [str(m) for m in pd.period_range(origin, periods=12, freq="M")]
    if [entry["file"] for entry in description["monthly_interiors"]] != [
        f"{month}.npz" for month in months
    ]:
        raise ValueError("Annual monthly target cohort differs")
    for month, record in zip(months, description["monthly_interiors"], strict=True):
        check_output(record, directory)
        with (
            np.load(directory / record["file"]) as interior,
            np.load(Path(plan["output"]) / "test" / f"{month}.npz") as monthly,
        ):
            for target, source in (("interior", "values"), ("ohc", "ohc")):
                np.testing.assert_array_equal(interior[source], monthly[target])
    with np.load(Path(plan["output"]) / "test" / f"{origin[:7]}.npz") as monthly:
        # NPZ members cannot be mmap'd; load one monthly field at a time (~0.80
        # GiB for atmosphere), never a whole 92-bin annual aggregate (~4.26 GiB).
        for key in ("surface", "velocity", "atmosphere"):
            expected = monthly[key]
            for index, record in enumerate(description["records"][:25]):
                with np.load(directory / record["file"]) as sample:
                    np.testing.assert_array_equal(sample[key], expected[index])
            del expected
        np.testing.assert_array_equal(
            monthly["midpoints"][:25],
            [
                (
                    pd.Timestamp(origin) + pd.Timedelta(days=5 * (i - 19) + 2.5)
                ).isoformat()
                for i in range(25)
            ],
        )
    expected_files = [
        record["file"]
        for record in description["records"] + description["monthly_interiors"]
    ]
    exact_inventory(directory, expected_files)


def annual_origin(plan, origin):
    check_plan(plan)
    if origin not in plan["annual_origins"]:
        raise ValueError("Origin is outside the fixed three-year cohort")
    ready_hash = monthly_ready(plan)
    directory = Path(plan["annual"]) / "test" / origin
    with exclusive(directory / "materialize.lock"):
        complete = directory / "COMPLETE.json"
        if complete.exists():
            audit_annual_origin(plan, origin, json.loads(complete.read_text()))
            return
        archive = BoundedArchive(
            Path(plan["upstream"]["root"]) / "daily", plan["upstream"]["records"]
        )
        records = []
        for index in range(92):
            start = pd.Timestamp(origin) + pd.Timedelta(days=5 * (index - 19))
            sample = annual_bin(archive, start)
            validate_bin(sample, start, archive.grid["mask"].shape[-2:])
            name = f"bin-{index:03d}.npz"
            save_npz(directory / name, sample)
            records.append(dict(file=name, sha256=digest(directory / name)))
            del sample
        interiors = []
        for month in pd.period_range(origin, periods=12, freq="M"):
            label = str(month)
            name = f"iap/{label[:4]}/{label}.npz"
            expected = plan["upstream"]["records"][name]
            copy_verified(archive.root / name, directory / f"{label}.npz", expected)
            interiors.append(dict(file=f"{label}.npz", sha256=expected))
        description = dict(
            origin=origin,
            history_bins=19,
            forecast_bins=73,
            bin_days=5,
            records=records,
            monthly_interiors=interiors,
            plan_sha256=plan_digest(plan),
            monthly_ready_sha256=ready_hash,
            source_commit=plan["source_commit"],
            runtime_sha256=plan["runtime_sha256"],
            monthly_exact_prefix_bins=25,
            monthly_exact_interior_targets=12,
        )
        audit_annual_origin(plan, origin, description)
        check_plan(plan)
        receipt(complete, description)


def verify_annual(plan):
    check_plan(plan)
    root = Path(plan["annual"])
    expected = {}
    for origin in plan["annual_origins"]:
        path = root / "test" / origin / "COMPLETE.json"
        description = json.loads(path.read_text())
        audit_annual_origin(plan, origin, description)
        for record in description["records"] + description["monthly_interiors"]:
            expected[f"test/{origin}/{record['file']}"] = record["sha256"]
    exact_inventory(root, expected)
    if len(expected) != 312:
        raise ValueError("Expected 312 annual NPZ files")
    manifests = {
        f"test/{origin}/COMPLETE.json": digest(root / "test" / origin / "COMPLETE.json")
        for origin in plan["annual_origins"]
    }
    if {
        str(path.relative_to(root)) for path in root.glob("test/*/COMPLETE.json")
    } != set(manifests):
        raise ValueError("Annual origin manifest inventory differs")
    hashes = {**expected, **manifests}
    immutable_bytes(
        root / "SHA256SUMS",
        "".join(
            f"{value}  {name}\n" for name, value in sorted(hashes.items())
        ).encode(),
    )
    receipt(
        root / "DATA_READY.json",
        dict(
            verification="312 annual NPZ files and three origin manifests verified",
            source_sha256sums=digest(root / "SHA256SUMS"),
            plan_sha256=plan_digest(plan),
            monthly_ready_sha256=monthly_ready(plan),
            origins=plan["annual_origins"],
            source_commit=plan["source_commit"],
            runtime_sha256=plan["runtime_sha256"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan")
    for name in ("prepared", "output", "annual", "runtime-contract", "source-commit"):
        planner.add_argument(f"--{name}", required=True)
    for name in ("monthly", "finalize", "annual", "verify-annual"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True)
        if name == "monthly":
            command.add_argument(
                "--worker", type=int, choices=range(WORKERS), required=True
            )
        if name == "annual":
            command.add_argument("--origin", choices=ANNUAL_ORIGINS, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(
            args.prepared,
            args.output,
            args.annual,
            args.runtime_contract,
            args.source_commit,
        )
    else:
        plan = json.loads(Path(args.plan).read_text())
        if args.command == "monthly":
            monthly_worker(plan, args.worker)
        elif args.command == "annual":
            annual_origin(plan, args.origin)
        elif args.command == "verify-annual":
            verify_annual(plan)
        else:
            finalize(plan)


if __name__ == "__main__":
    main()
