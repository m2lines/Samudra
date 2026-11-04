# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ocean_preprocessing.obs_preprocessing import full_range as full


def make_archive(tmp_path, product, count=27):
    first = "2000-12-20" if product != "argo-iap" else "1999-01-01"
    frequency = "MS" if product == "argo-iap" else "D"
    dates = pd.date_range(first, periods=count, freq=frequency)
    plan = dict(
        schema_version=1,
        product=product,
        frequency=frequency,
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        time_count=count,
        variables=["ugos", "vgos", "ugosa", "vgosa"]
        if product == "duacs"
        else ["sst"]
        if product == "oisst"
        else ["temp", "salt"],
    )
    root = tmp_path / "raw" / product
    root.mkdir(parents=True)
    values = np.arange(count * 6, dtype="float32").reshape(count, 2, 3) ** 2
    values[:, 0, 0] = np.nan
    times = dates + (pd.Timedelta(hours=12) if product == "oisst" else pd.Timedelta(0))
    fields = {
        name: (("time", "latitude", "longitude"), values + i)
        for i, name in enumerate(plan["variables"])
    }
    expected = xr.Dataset(
        fields,
        coords=dict(time=times, latitude=[20.0, -20.0], longitude=[-120.0, 0.0, 120.0]),
    )
    if product == "duacs":
        expected = expected.astype("float64") / 10000.0
        for name in expected.data_vars:
            expected[name].attrs["units"] = "m/s"
        plan.update(dataset_version="202411", dataset_id=full.raw.DUACS_DATASET_ID)
        for a, b in full.raw._yearly_ranges(plan["start_date"], plan["end_date"]):
            expected.sel(time=slice(str(a), str(b))).to_zarr(
                root / f"duacs_{a:%Y%m%d}_{b:%Y%m%d}.zarr", zarr_format=2
            )
    else:
        records = []
        for i, date in enumerate(dates):
            if product == "oisst":
                filename = f"oisst-avhrr-v02r01.{date:%Y%m%d}.nc"
                expected.isel(time=slice(i, i + 1)).expand_dims(zlev=[0]).to_netcdf(
                    root / filename, engine="h5netcdf"
                )
                records.append(
                    dict(
                        date=str(date.date()),
                        path=filename,
                        url="https://example.invalid/" + filename,
                    )
                )
            else:
                for field, name in [("temperature", "temp"), ("salinity", "salt")]:
                    filename = f"{field}/IAP_05_2000m_{field}_year_{date.year}_month_{date.month:02}.nc"
                    (root / field).mkdir(exist_ok=True)
                    expected[[name]].isel(time=i, drop=True).expand_dims(
                        depth_std=[1.0]
                    ).to_netcdf(root / filename, engine="h5netcdf")
                    records.append(
                        dict(
                            date=str(date.date()),
                            path=filename,
                            url="http://example.invalid/" + filename,
                        )
                    )
        plan["records"] = sorted(records, key=lambda r: (r["date"], r["path"]))
    manifest = tmp_path / "inventory.json"
    full._atomic_json(manifest, plan)
    full._atomic_json(root / "inventory.json", plan)
    return manifest, plan, expected


@pytest.mark.parametrize("product", full.PRODUCTS)
def test_native_values_and_time_survive_preparation(tmp_path, product):
    manifest, plan, expected = make_archive(tmp_path, product)
    out = tmp_path / "prepared"
    full.prepare(str(manifest), str(tmp_path / "raw"), str(out), threads=2)
    store = out / f"{product}.zarr"
    with xr.open_zarr(store) as actual:
        expected = full.prep._standardize_daily(expected)
        if product == "argo-iap":
            expected = expected.expand_dims(depth_std=[1.0]).transpose(
                "time", "depth_std", "lat", "lon"
            )
            actual = actual.transpose("time", "depth_std", "lat", "lon")
        xr.testing.assert_allclose(actual, expected)
        assert "none;" in actual.attrs["m2lines/time_alignment"]
        assert actual.attrs["m2lines/inventory_sha256"] == full._digest(plan)
    # Completed reruns validate rather than overwrite.
    full.prepare(str(manifest), str(tmp_path / "raw"), str(out))
    assert not (out / f".{product}.progress.json").exists()


def test_positive_360_longitude_is_wrapped_with_its_data():
    source = xr.Dataset(
        {"temp": (("time", "lat", "lon"), np.array([[[10.0, 20.0, 30.0]]]))},
        coords={
            "time": pd.date_range("2023-01-01", periods=1),
            "lat": [0.0],
            "lon": [0.5, 180.0, 360.0],
        },
    )
    actual = full.prep._standardize_daily(source)
    np.testing.assert_array_equal(actual.lon.values, [0.0, 0.5, 180.0])
    np.testing.assert_array_equal(actual.temp.values, [[[30.0, 10.0, 20.0]]])
    np.testing.assert_array_equal(source.lon.values, [0.5, 180.0, 360.0])


def test_interrupted_block_resumes_without_dropping_or_averaging(tmp_path, monkeypatch):
    manifest, plan, expected = make_archive(tmp_path, "oisst")
    original = full._open_block

    @contextlib.contextmanager
    def interrupted(plan, root, dates):
        if dates[0] == pd.Timestamp("2001-01-13"):
            raise RuntimeError("simulated interruption")
        with original(plan, root, dates) as ds:
            yield ds

    out = tmp_path / "prepared"
    monkeypatch.setattr(full, "_open_block", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        full.prepare(str(manifest), str(tmp_path / "raw"), str(out))
    assert json.loads((out / ".oisst.progress.json").read_text())["written"] == 24
    assert not (out / "oisst.zarr").exists()
    monkeypatch.setattr(full, "_open_block", original)
    full.prepare(str(manifest), str(tmp_path / "raw"), str(out))
    with xr.open_zarr(out / "oisst.zarr") as actual:
        xr.testing.assert_allclose(actual, full.prep._standardize_daily(expected))


def test_missing_chunk_is_not_silently_accepted_as_land(tmp_path):
    manifest, _, _ = make_archive(tmp_path, "duacs")
    out = tmp_path / "prepared"
    full.prepare(str(manifest), str(tmp_path / "raw"), str(out))
    (out / "duacs.zarr" / "ugos" / "0.0.0").unlink()
    with pytest.raises(ValueError, match="Missing chunk"):
        full.validate(str(manifest), str(out / "duacs.zarr"))


def test_missing_month_in_inventory_is_rejected(tmp_path):
    manifest, plan, _ = make_archive(tmp_path, "argo-iap")
    plan["records"].pop(3)
    full._atomic_json(manifest, plan)
    with pytest.raises(ValueError, match="missing, duplicate"):
        full.prepare(str(manifest), str(tmp_path / "raw"), str(tmp_path / "prepared"))


def test_final_only_oisst_discovery_and_partial_years(tmp_path, monkeypatch):
    base = full.raw.OISST_BASE_URL + "/"
    listings = {
        base: ["198109", "198110/"],
        base + "198109/": [f"oisst-avhrr-v02r01.198109{d:02}.nc" for d in range(1, 31)],
        base + "198110/": [
            "oisst-avhrr-v02r01.19811001.nc",
            "oisst-avhrr-v02r01.19811002_preliminary.nc",
        ],
    }
    monkeypatch.setattr(full, "_links", lambda url: listings[url])
    path = tmp_path / "plan.json"
    plan = full.discover("oisst", str(path))
    assert plan["time_count"] == 31
    assert plan["end_date"] == "1981-10-01"
    with pytest.raises(FileExistsError):
        full.discover("oisst", str(path))


def test_iap_discovery_refuses_different_variable_coverage(monkeypatch):
    def links(url):
        field = "temperature" if "Temperature" in url else "salinity"
        return [
            f"IAP_05_2000m_{field}_year_1960_month_{m:02}.nc"
            for m in ([1, 2] if field == "temperature" else [1])
        ]

    monkeypatch.setattr(full, "_links", links)
    with pytest.raises(ValueError, match="months differ"):
        full._iap_records()


def test_copernicus_epoch_units():
    assert full._catalogue_time(
        dict(
            coordinate_unit="milliseconds since 1970-01-01 00:00:00Z (no leap seconds)",
            minimum_value=725846400000.0,
        ),
        "minimum_value",
    ) == pd.Timestamp("1993-01-01")
    with pytest.raises(ValueError, match="Unsupported numeric"):
        full._catalogue_time(
            dict(coordinate_unit="days", minimum_value=1), "minimum_value"
        )


def test_missing_raw_duacs_chunk_is_rejected_before_preparing(tmp_path):
    manifest, _, _ = make_archive(tmp_path, "duacs")
    next((tmp_path / "raw" / "duacs").glob("*.zarr/ugos/0.0.0")).unlink()
    with pytest.raises(ValueError, match="Missing chunk"):
        full.prepare(str(manifest), str(tmp_path / "raw"), str(tmp_path / "prepared"))


def test_download_failure_never_promotes_partial_netcdf(tmp_path, monkeypatch):
    manifest, plan, _ = make_archive(tmp_path, "oisst", count=1)
    root = tmp_path / "raw" / "oisst"
    target = root / plan["records"][0]["path"]
    target.unlink()

    def fail(tasks, **kwargs):
        for _, partial in tasks:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"partial")
        return {url for url, _ in tasks}

    monkeypatch.setattr(full.raw, "_download_batch", fail)
    with pytest.raises(RuntimeError, match="downloads failed"):
        full.download(str(manifest), str(tmp_path / "raw"))
    assert not target.exists()
    assert not list(root.glob(".download-*"))


def test_corrupt_existing_download_is_replaced_atomically(tmp_path, monkeypatch):
    manifest, plan, _ = make_archive(tmp_path, "oisst", count=1)
    target = tmp_path / "raw" / "oisst" / plan["records"][0]["path"]
    valid = target.read_bytes()
    target.write_bytes(b"not a netcdf" * 100)
    monkeypatch.setattr(full.raw, "MIN_BYTES_OISST", 1)

    def succeed(tasks, **kwargs):
        for _, temporary in tasks:
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(valid)
        return set()

    monkeypatch.setattr(full.raw, "_download_batch", succeed)
    full.download(str(manifest), str(tmp_path / "raw"))
    assert target.read_bytes() == valid


def test_duacs_download_pins_version_and_optional_variables(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(full.raw, "_has_copernicus_credentials", lambda _: True)
    monkeypatch.setattr(full.raw.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    full.raw.duacs(
        tmp_path,
        start_date="1993-01-01",
        end_date="1993-01-02",
        dataset_version="202411",
        include_ssh=True,
        dry_run=True,
    )
    assert calls[0][calls[0].index("--dataset-version") + 1] == "202411"
    assert "adt" in calls[0] and "sla" in calls[0]
