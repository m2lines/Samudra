# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the package virtualization entrypoint and reference stores."""

import json
import subprocess
import sys

import fsspec
import numpy as np
import pytest
import xarray as xr
from ocean_preprocessing.virtualize_om4 import OSN, SourceArchive, build


def write_years(
    root, *, changed_grid=False, gap=False, missing_variable=False, classic=False
):
    root.mkdir()
    datasets = []
    for i in range(2):
        ds = xr.Dataset(
            {
                "field": (
                    ("time", "y", "x"),
                    np.arange(12, dtype="float32").reshape(2, 2, 3) + i,
                ),
                "time_bnds": (("time", "bnds"), np.array([[0, 5], [5, 10]]) + i * 10),
                "area": (("y", "x"), np.ones((2, 3))),
            },
            coords={"time": [2.5 + i * 10, 7.5 + i * 10], "y": [0, 1], "x": [0, 1, 2]},
        )
        ds.field.values[0, 0, 0] = np.nan
        ds["other_field"] = (
            ("time", "y", "x"),
            np.ones((2, 2, 3), dtype="float64") * (i + 0.125),
        )
        ds.time.attrs = {
            "units": "days since 1958-01-01",
            "calendar": "julian",
            "bounds": "time_bnds",
        }
        ds.time_bnds.attrs = {"units": "days since 1958-01-01", "calendar": "julian"}
        ds.field.attrs = {"units": "1", "long_name": "Synthetic test field"}
        if changed_grid and i:
            ds = ds.assign_coords(x=[0, 1, 3])
        if gap and i:
            ds["time"] = ds.time + 5
            ds.time.attrs = datasets[0].time.attrs.copy()
        if missing_variable and i:
            ds = ds.drop_vars("field")
        encoding = (
            {"field": {"chunksizes": (1, 2, 3), "zlib": True, "_FillValue": -999.0}}
            if "field" in ds
            else {}
        )
        if classic:
            ds.to_netcdf(
                root / f"{1958 + i}0101.ocean_5daily.nc",
                engine="scipy",
                format="NETCDF3_64BIT",
                unlimited_dims=["time"],
            )
        else:
            ds.to_netcdf(
                root / f"{1958 + i}0101.ocean_5daily.nc",
                engine="h5netcdf",
                encoding=encoding,
            )
        datasets.append(ds)
    return datasets


@pytest.mark.parametrize("classic", [False, True])
@pytest.mark.parametrize("gap", [False, True])
def test_parquet_roundtrip(tmp_path, classic, gap):
    src = tmp_path / "raw"
    original = write_years(src, classic=classic, gap=gap)
    output = build(str(src), tmp_path / "refs", record_size=2)
    assert list(output.rglob("*.parq"))
    metadata = json.loads((output / ".zmetadata").read_text())
    assert metadata["metadata"][".zgroup"]["zarr_format"] == 2
    mapper = fsspec.filesystem(
        "reference", fo=str(output), remote_protocol="file", asynchronous=True
    ).get_mapper("")
    with xr.open_zarr(
        mapper, consolidated=False, chunks=None, decode_times=False
    ) as actual:
        expected = xr.concat(
            original, dim="time", data_vars="minimal", coords="minimal", compat="equals"
        )
        xr.testing.assert_equal(actual.load(), expected)
    assert (output / "provenance.json").is_file()
    with pytest.raises(FileExistsError):
        build(str(src), output, expected_files=2)


@pytest.mark.parametrize("change", ["changed_grid", "missing_variable"])
def test_reject_invalid_archive(tmp_path, change):
    src = tmp_path / "raw"
    write_years(src, **{change: True})
    with pytest.raises(ValueError):
        build(str(src), tmp_path / "refs", expected_files=2)
    assert not (tmp_path / "refs").exists()


def test_explicit_file_count(tmp_path):
    src = tmp_path / "raw"
    write_years(src)
    with pytest.raises(ValueError, match="Expected 65"):
        build(str(src), tmp_path / "refs", expected_files=65)


def test_static(tmp_path):
    ds = xr.Dataset({"area": (("y", "x"), np.ones((2, 3)))})
    ds.to_netcdf(tmp_path / "ocean_static.nc", engine="h5netcdf")
    output = build(
        str(tmp_path),
        tmp_path / "refs",
        pattern="ocean_static.nc",
        expected_files=1,
        static=True,
    )
    mapper = fsspec.filesystem(
        "reference", fo=str(output), remote_protocol="file", asynchronous=True
    ).get_mapper("")
    with xr.open_zarr(mapper, consolidated=False, chunks=None) as actual:
        xr.testing.assert_equal(actual.load(), ds)


@pytest.mark.parametrize(
    "flags",
    [
        ["--expected-files", "2", "--record-size", "2", "--anonymous=True"],
        ["--expected_files=2", "--record_size=2", "--anonymous=False"],
    ],
)
def test_module_entrypoint(tmp_path, flags):
    src = tmp_path / "raw"
    write_years(src, classic=True)
    output = tmp_path / "refs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ocean_preprocessing.virtualize_om4",
            str(src),
            str(output),
            *flags,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (output / "provenance.json").is_file()
    assert "Wrote" in result.stderr


@pytest.mark.parametrize("flags", [["--help"], ["--", "--help"]])
def test_module_help(flags):
    result = subprocess.run(
        [sys.executable, "-m", "ocean_preprocessing.virtualize_om4", *flags],
        check=True,
        capture_output=True,
        text=True,
    )
    help_text = result.stdout + result.stderr
    assert "Local source directory or s3:// bucket/prefix" in help_text
    assert "Number of references per Parquet partition" in help_text
    assert "Read public sources without credentials" in help_text


def test_source_discovery_and_fresh_inventory(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    source = root / "part.nc"
    source.write_bytes(b"abcd")
    (root / "notes.txt").write_text("Not a source file")

    archive = SourceArchive.discover(str(root), "*.nc", OSN, True)
    assert archive.urls == (source.as_uri(),)
    before = archive.snapshot()
    source.write_bytes(b"abcdefgh")
    after = archive.snapshot()
    assert int(before[0]["size"]) == 4
    assert int(after[0]["size"]) == 8


def test_declared_coordinate_bounds(tmp_path):
    src = tmp_path / "raw"
    datasets = write_years(src, classic=True)
    for path, dataset in zip(sorted(src.glob("*.nc")), datasets):
        dataset = dataset.rename(time_bnds="interval_bounds")
        dataset.time.attrs["bounds"] = "interval_bounds"
        dataset.to_netcdf(
            path, engine="scipy", format="NETCDF3_64BIT", unlimited_dims=["time"]
        )
    output = build(str(src), tmp_path / "refs")
    mapper = fsspec.filesystem(
        "reference", fo=str(output), remote_protocol="file", asynchronous=True
    ).get_mapper("")
    with xr.open_zarr(
        mapper, consolidated=False, chunks=None, decode_times=False
    ) as actual:
        np.testing.assert_array_equal(
            actual.interval_bounds.values, [[0, 5], [5, 10], [10, 15], [15, 20]]
        )
        assert actual.time.attrs["bounds"] == "interval_bounds"


@pytest.mark.parametrize("failure", ["validation", "source_changed"])
def test_failed_build_does_not_publish(tmp_path, monkeypatch, failure):
    from ocean_preprocessing import virtualize_om4

    src = tmp_path / "raw"
    write_years(src)
    validate = virtualize_om4.validate_parquet_references

    def interrupt(directory, expected, source):
        validate(directory, expected, source)
        if failure == "validation":
            raise ValueError("Injected reference validation failure")
        path = next(src.glob("*.nc"))
        path.write_bytes(path.read_bytes() + b"changed")

    monkeypatch.setattr(virtualize_om4, "validate_parquet_references", interrupt)
    output = tmp_path / "refs"
    with pytest.raises(ValueError, match="validation failure|Source objects changed"):
        build(str(src), output)
    assert not output.exists()
    assert not list(tmp_path.glob(".refs-*"))


@pytest.mark.parametrize("fill_value", [0, 42])
def test_unwritten_native_fill_is_not_silently_changed(tmp_path, fill_value):
    import h5py

    source = tmp_path / "grid.nc"
    with h5py.File(source, "w") as handle:
        axis = handle.create_dataset("x", data=np.arange(2))
        axis.make_scale("x")
        field = handle.create_dataset(
            "field", shape=(2,), chunks=(1,), dtype="i4", fillvalue=fill_value
        )
        field.dims[0].attach_scale(axis)
        field[0] = 1
    with xr.open_dataset(source, engine="h5netcdf", decode_times=False) as native:
        np.testing.assert_array_equal(native.field.values, [1, fill_value])
    output = tmp_path / "refs"
    with pytest.raises(ValueError, match="unwritten native chunks"):
        build(str(tmp_path), output, pattern="grid.nc", static=True)
    assert not output.exists()
