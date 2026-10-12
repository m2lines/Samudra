# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pandas as pd
import pytest

from samudra.experiments.observation_data import ObservationArchive, month_intervals
from scripts import materialize_quarter_observations as materialize


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "daily"
    root.mkdir()
    np.savez(
        root / "grid.npz",
        lat=[-45, 45],
        lon=[45, 135, 225, 315],
        mask=np.ones((77, 2, 4), bool),
    )
    starts, ends, _ = month_intervals("2015-01")
    records = {}
    for product, channels in (("sst", 1), ("adt", 3), ("era5", 8)):
        for index, day in enumerate(
            pd.date_range(starts[0], ends[-1], inclusive="left")
        ):
            path = root / product / str(day.year) / f"{day.date()}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            values = (np.arange(channels * 8).reshape(channels, 2, 4) + index).astype(
                np.float32
            )
            if product != "era5" and index == 4:
                values[0, 0, 0] = np.nan
            np.savez(path, values=values)
            records[str(path.relative_to(root))] = materialize.digest(path)
    path = root / "iap" / "2015" / "2015-01.npz"
    path.parent.mkdir(parents=True)
    np.savez(
        path,
        values=np.arange(112, dtype=np.float32).reshape(2, 14, 2, 2),
        ohc=np.ones((2, 2, 2), np.float32),
    )
    records[str(path.relative_to(root))] = materialize.digest(path)
    return root, records


def test_four_strides_are_exact_unique_cohorts():
    jobs = materialize.month_jobs()
    workers = [materialize.worker_jobs(jobs, i) for i in range(4)]
    assert [len(worker) for worker in workers] == [87] * 4
    assert (
        len({(job["split"], job["month"]) for worker in workers for job in worker})
        == 348
    )
    assert {
        split: sum(job["split"] == split for job in jobs)
        for split in ("train", "validation", "test")
    } == {"train": 243, "validation": 9, "test": 96}
    assert all(workers[i][j] == jobs[4 * j + i] for i in range(4) for j in range(87))
    with pytest.raises(ValueError, match="Worker"):
        materialize.worker_jobs(jobs, 4)


def test_bounded_archive_exact_legacy_arrays_and_nan_support(archive):
    root, records = archive
    actual_archive = materialize.BoundedArchive(root, records)
    actual = actual_archive.sample("2015-01")
    expected = ObservationArchive(root).sample("2015-01")
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert np.isnan(actual["surface"][0, 0, 0, 0])
    assert np.isfinite(actual["atmosphere"]).all()
    assert actual_archive.day.cache_info().maxsize == 15
    assert actual_archive.day.cache_info().currsize == 15
    materialize.validate_sample(actual, "2015-01", (2, 4), (2, 2))
    actual["atmosphere"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="forcing"):
        materialize.validate_sample(actual, "2015-01", (2, 4), (2, 2))


def test_changed_upstream_payload_fails_before_sampling(archive):
    root, records = archive
    path = root / "sst" / "2014" / "2014-09-28.npz"
    assert path.exists()
    path.write_bytes(b"changed")
    bounded = materialize.BoundedArchive(root, records)
    with pytest.raises(ValueError, match="payload changed"):
        bounded.sample("2015-01")


def test_month_receipt_resume_and_changed_output(archive, tmp_path):
    daily, records = archive
    output = tmp_path / "monthly"
    output.mkdir()
    plan = {"output": str(output)}
    materialize.receipt(output / "MATERIALIZATION_PLAN.json", plan)
    job = {"split": "test", "month": "2015-01"}
    source = materialize.BoundedArchive(daily, records)
    receipt = materialize.materialize_month(plan, job, source, (2, 4), (2, 2))
    assert receipt == materialize.materialize_month(plan, job, None, (2, 4), (2, 2))
    with np.load(output / receipt["file"]) as actual:
        expected = ObservationArchive(daily).sample("2015-01")
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
    (output / receipt["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload changed"):
        materialize.materialize_month(plan, job, source, (2, 4), (2, 2))
    assert not (output / "DATA_READY.json").exists()


def test_exact_inventory_and_immutable_publication(tmp_path):
    np.savez(tmp_path / "expected.npz", value=1)
    materialize.exact_inventory(tmp_path, ["expected.npz"])
    np.savez(tmp_path / "extra.npz", value=2)
    with pytest.raises(ValueError, match="inventory differs"):
        materialize.exact_inventory(tmp_path, ["expected.npz"])
    marker = tmp_path / "receipt.json"
    materialize.receipt(marker, {"hash": "first"})
    materialize.receipt(marker, {"hash": "first"})
    with pytest.raises(ValueError, match="artifact differs"):
        materialize.receipt(marker, {"hash": "changed"})
    assert json.loads(marker.read_text()) == {"hash": "first"}
    dtype_path = tmp_path / "wrong-dtype.npz"
    np.savez(dtype_path, values=np.ones(2, np.float64))
    with pytest.raises(AssertionError, match="dtype"):
        materialize.save_npz(dtype_path, dict(values=np.ones(2, np.float32)))


def test_runtime_change_blocks_materialization(tmp_path):
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    plan = dict(
        schema=1,
        workers=4,
        jobs=materialize.month_jobs(),
        annual_origins=materialize.ANNUAL_ORIGINS,
        code=materialize.code_hashes(),
        runtime_contract=str(runtime),
        runtime_sha256=materialize.digest(runtime),
        runtime_identity={"machine": "different-architecture"},
    )
    with pytest.raises(ValueError, match="source/runtime changed"):
        materialize.check_plan(plan)


def test_missing_prepared_ready_blocks_planning(tmp_path):
    with pytest.raises(FileNotFoundError, match="PREPARED_READY"):
        materialize.prepared_contract(tmp_path)


def test_prepared_receipt_chain_checks_hashes_and_exact_inventory(
    tmp_path, monkeypatch
):
    calendar = {"daily_dates": ["2015-01-01"]}
    monkeypatch.setattr(materialize, "required_calendar", lambda: calendar)
    monkeypatch.setattr(
        materialize, "month_jobs", lambda: [{"split": "test", "month": "2015-01"}]
    )
    daily = tmp_path / "daily"
    daily.mkdir()
    (daily / "grid.npz").write_bytes(b"model grid")
    (daily / "iap-grid.npz").write_bytes(b"native grid")
    materialize.receipt(
        daily / "iap-operator.json",
        {
            "model_grid_sha256": materialize.digest(daily / "grid.npz"),
            "iap_grid_sha256": materialize.digest(daily / "iap-grid.npz"),
        },
    )
    for product in ("sst", "adt", "era5", "iap"):
        path = (
            daily
            / product
            / "2015"
            / ("2015-01.npz" if product == "iap" else "2015-01-01.npz")
        )
        path.parent.mkdir(parents=True)
        path.write_bytes(b"payload")
    raw = tmp_path / "raw-metadata.json"
    raw.write_text("{}")
    materialize.receipt(
        tmp_path / "PLAN.json",
        dict(
            output=str(tmp_path),
            calendar=calendar,
            tasks=[{}],
            input_hashes={str(raw): materialize.digest(raw)},
        ),
    )
    task_path = tmp_path / "receipts" / "000.json"
    materialize.receipt(
        task_path,
        dict(
            plan_sha256=materialize.digest(tmp_path / "PLAN.json"),
            records=[
                dict(file=str(path), sha256=materialize.digest(path))
                for path in daily.rglob("*")
                if path.is_file()
            ],
        ),
    )
    materialize.receipt(
        tmp_path / "PREPARED_READY.json",
        dict(
            plan_sha256=materialize.digest(tmp_path / "PLAN.json"),
            task_receipts={"000": materialize.digest(task_path)},
        ),
    )
    assert len(materialize.prepared_contract(tmp_path)["records"]) == 7
    task_path.write_text(task_path.read_text() + "\n")
    with pytest.raises(ValueError, match="receipt changed"):
        materialize.prepared_contract(tmp_path)


def test_finalize_fits_training_only_and_publishes_exact_351(tmp_path, monkeypatch):
    root = tmp_path / "monthly"
    root.mkdir()
    np.savez(
        root / "grid.npz",
        lat=[-45, 45],
        lon=[45, 135, 225, 315],
        mask=np.ones((77, 2, 4), bool),
    )
    np.savez(root / "iap-grid.npz", lat=[-45, 45], lon=[90, 270])
    plan = dict(
        output=str(root),
        jobs=materialize.month_jobs(),
        upstream=dict(
            records={
                name: materialize.digest(root / name)
                for name in ("grid.npz", "iap-grid.npz")
            },
            ready_sha256="prepared",
        ),
        source_commit="a" * 40,
        runtime_sha256="b" * 64,
    )
    materialize.receipt(root / "MATERIALIZATION_PLAN.json", plan)
    monkeypatch.setattr(materialize, "check_plan", lambda plan: None)
    for worker in range(4):
        records = []
        for job in materialize.worker_jobs(plan["jobs"], worker):
            name = f"{job['split']}/{job['month']}.npz"
            (root / job["split"]).mkdir(exist_ok=True)
            np.savez(root / name, month=job["month"])
            records.append(dict(file=name, sha256=materialize.digest(root / name)))
        materialize.receipt(
            root / "receipts" / f"worker-{worker}.json",
            dict(plan_sha256=materialize.plan_digest(plan), records=records),
        )
    calls = []

    def fit(samples, grid, output, *, iap_grid):
        assert samples == root / "train"
        months = sorted(path.stem for path in samples.glob("*.npz"))
        assert months == [
            str(month) for month in pd.period_range("1993-05", "2013-07", freq="M")
        ]
        assert iap_grid == root / "iap-grid.npz"
        calls.append(months)
        with output.open("wb") as stream:
            np.savez(stream, training_month_count=243)

    monkeypatch.setattr(materialize, "fit_statistics", fit)
    materialize.finalize(plan)
    assert len((root / "SHA256SUMS").read_text().splitlines()) == 351
    assert (
        json.loads((root / "DATA_READY.json").read_text())["verification"]
        == "all 351 NPZ files checked against source SHA256SUMS"
    )
    materialize.finalize(plan)
    assert len(calls) == 1  # Verified statistics reuse, no recomputation/replacement.
    np.savez(root / "train" / "2015-01.npz", validation_leak=1)
    with pytest.raises(ValueError, match="inventory differs"):
        materialize.finalize(plan)
    assert len(calls) == 1


def test_annual_stream_matches_read_origin_and_monthly_arrays(
    archive, tmp_path, monkeypatch
):
    from samudra.experiments.observation_annual import read_origin

    daily, records = archive
    for day in pd.date_range("2014-09-28", "2016-01-04"):
        for product, channels in (("sst", 1), ("adt", 3), ("era5", 8)):
            path = daily / product / str(day.year) / f"{day.date()}.npz"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    path, values=np.full((channels, 2, 4), day.dayofyear, np.float32)
                )
                records[str(path.relative_to(daily))] = materialize.digest(path)
    for month in pd.period_range("2015-02", "2015-12", freq="M"):
        path = daily / "iap" / "2015" / f"{month}.npz"
        np.savez(
            path,
            values=np.full((2, 14, 2, 2), month.month, np.float32),
            ohc=np.full((2, 2, 2), month.month, np.float32),
        )
        records[str(path.relative_to(daily))] = materialize.digest(path)
    output = tmp_path / "monthly"
    output.mkdir()
    materialize.copy_verified(
        daily / "grid.npz", output / "grid.npz", materialize.digest(daily / "grid.npz")
    )
    source = materialize.BoundedArchive(daily, records)
    for month in pd.period_range("2015-01", "2015-12", freq="M"):
        materialize.save_npz(
            output / "test" / f"{month}.npz", source.sample(str(month))
        )
    plan = dict(
        output=str(output),
        annual=str(tmp_path / "annual"),
        annual_origins=["2015-01-01"],
        upstream=dict(root=str(daily.parent), records=records),
        source_commit="a" * 40,
        runtime_sha256="b" * 64,
    )
    materialize.receipt(output / "MATERIALIZATION_PLAN.json", plan)
    # Source and full-cohort gates have separate tests; use tiny physical grids
    # here to exercise the real archive, writer, auditor and annual read_origin.
    monkeypatch.setattr(materialize, "check_plan", lambda plan: None)
    monkeypatch.setattr(materialize, "monthly_ready", lambda plan: "qualified-monthly")
    original_stack = np.stack

    def bounded_stack(arrays, *args, **kwargs):
        assert len(arrays) <= 26, "Materialization stacked an annual aggregate"
        return original_stack(arrays, *args, **kwargs)

    with monkeypatch.context() as bounded:
        bounded.setattr(np, "stack", bounded_stack)
        materialize.annual_origin(plan, "2015-01-01")
    directory = tmp_path / "annual" / "test" / "2015-01-01"
    description, actual = read_origin(directory)
    assert len(description["monthly_interiors"]) == 12
    assert len(description["records"]) == 92
    assert actual["atmosphere"].shape == (92, 8, 2, 4)
    with np.load(output / "test" / "2015-01.npz") as monthly:
        for name in ("surface", "atmosphere", "velocity"):
            np.testing.assert_array_equal(actual[name][:25], monthly[name][:25])
    path = directory / "bin-000.npz"
    with np.load(path) as saved:
        corrupted = dict(saved)
    corrupted["atmosphere"][0, 0, 0] = np.nan
    np.savez(path, **corrupted)
    description["records"][0]["sha256"] = materialize.digest(path)
    with pytest.raises(ValueError, match="forcing"):
        materialize.audit_annual_origin(plan, "2015-01-01", description)
