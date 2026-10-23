# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock, sentinel

import numpy as np
import ocean_preprocessing.__main__ as main
import pytest
import xarray as xr
from numcodecs import Blosc
from ocean_preprocessing.__main__ import CLI


def test_clusterless_dry_run_computes_without_a_client(tmp_path):
    pipeline = CLI(output_path=str(tmp_path / "unused.zarr"), dry_run=True)
    ds = xr.Dataset({"value": ("time", np.arange(3))}).chunk({"time": 1})

    pipeline._collect(ds)

    assert not (tmp_path / "unused.zarr").exists()


def test_collect_explicitly_writes_zarr_v2(tmp_path):
    output = tmp_path / "output.zarr"
    pipeline = CLI(output_path=str(output))
    ds = xr.Dataset({"value": ("time", np.arange(3))}).chunk({"time": 1})

    pipeline._collect(ds)

    metadata = json.loads((output / ".zgroup").read_text(encoding="utf-8"))
    assert metadata["zarr_format"] == 2


def test_collect_keeps_physical_fields_uncompressed_by_default(tmp_path):
    output = tmp_path / "output.zarr"
    pipeline = CLI(output_path=str(output))
    ds = xr.Dataset({"value": ("time", np.arange(3))}).chunk({"time": 1})

    pipeline._collect(ds)

    metadata = json.loads((output / "value" / ".zarray").read_text(encoding="utf-8"))
    assert metadata["compressor"] is None


def test_collect_compresses_basin_masks_by_default(tmp_path: Path) -> None:
    output = tmp_path / "output.zarr"
    pipeline = CLI(output_path=str(output))
    ds = xr.Dataset({"basin_arctic": (("y", "x"), np.ones((2, 3), dtype=bool))})

    pipeline._collect(ds)

    metadata = json.loads((output / "basin_arctic" / ".zarray").read_text())
    assert metadata["compressor"]["id"] == "blosc"
    assert metadata["compressor"]["cname"] == "zstd"
    assert metadata["compressor"]["shuffle"] == Blosc.SHUFFLE


@pytest.mark.parametrize("compress", [False, True])
def test_collect_compression_only_toggles_physical_fields(
    tmp_path: Path, compress: bool
) -> None:
    output = tmp_path / "output.zarr"
    pipeline = CLI(output_path=str(output), compress=compress)
    values = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
    values[:, :, 0, 0] = np.nan
    ds = xr.Dataset(
        {
            "thetao": (("time", "lev", "y", "x"), values, {"units": "degC"}),
            "mask_0": (("y", "x"), np.isfinite(values[0, 0])),
        },
        coords={
            "time": xr.date_range(
                "0151-01-06", periods=2, freq="5D", calendar="noleap", use_cftime=True
            ),
            "lev": ("lev", [5.0, 20.0], {"units": "meters"}),
            "y": [-1.0, 1.0],
            "x": [0.0, 1.0, 2.0],
            "wetmask": (("lev", "y", "x"), np.isfinite(values[0])),
            "dz": ("lev", [10.0, 20.0], {"units": "meters"}),
        },
        attrs={"grid_type": "gaussian"},
    ).chunk({"time": 1})
    # Explicit output encoding must override codecs inherited from source stores.
    for field in ds.variables.values():
        field.encoding["compressor"] = Blosc(cname="lz4")

    pipeline._collect(ds)

    for name in ds.variables:
        metadata = json.loads((output / name / ".zarray").read_text())
        if name == "thetao" and not compress:
            assert metadata["compressor"] is None
            continue
        assert metadata["compressor"]["id"] == "blosc"
        assert metadata["compressor"]["cname"] == "zstd"
        assert metadata["compressor"]["clevel"] == 5
        assert metadata["compressor"]["shuffle"] == Blosc.SHUFFLE
    assert json.loads((output / "thetao" / ".zarray").read_text())["chunks"] == [
        1,
        2,
        2,
        3,
    ]
    with xr.open_zarr(output) as actual:
        xr.testing.assert_identical(actual.load(), ds.compute())


def test_small_run_leaves_a_dataset_without_time_alone(tmp_path):
    """Not everything written here is a time series; basin masks are not."""
    output = tmp_path / "output.zarr"
    pipeline = CLI(output_path=str(output), small_run=True)
    ds = xr.Dataset({"mask": (("lat", "lon"), np.ones((4, 6)))})

    pipeline._collect(ds)

    assert xr.open_zarr(output)["mask"].shape == (4, 6)


def test_small_run_still_truncates_a_time_series(tmp_path):
    output = tmp_path / "output.zarr"
    pipeline = CLI(output_path=str(output), small_run=True)
    ds = xr.Dataset({"value": ("time", np.arange(50))}).chunk({"time": 1})

    pipeline._collect(ds)

    assert xr.open_zarr(output).sizes["time"] == 10


def test_wfo_source_path_is_not_forwarded_to_cluster(monkeypatch, tmp_path):
    cluster_calls = []

    def fake_init_cluster(cluster, **cluster_opts):
        cluster_calls.append((cluster, cluster_opts))
        return sentinel.client

    monkeypatch.setattr(main, "init_cluster", fake_init_cluster)

    pipeline = CLI(
        output_path=str(tmp_path / "unused.zarr"),
        cluster="local",
        wfo_source_path="s3://example/wfo.zarr",
        n_workers=4,
    )

    assert pipeline.wfo_source_path == "s3://example/wfo.zarr"
    assert pipeline.dask_client is sentinel.client
    assert cluster_calls == [("local", {"n_workers": 4})]


def test_output_profile_only_applies_to_the_output_store(monkeypatch) -> None:
    cluster = MagicMock(return_value=None)
    monkeypatch.setattr(main, "init_cluster", cluster)
    writer = MagicMock()
    monkeypatch.setattr(xr.Dataset, "to_zarr", writer)
    pipeline = CLI(
        output_path="s3://destination/CM4.zarr",
        output_profile="publication",
        n_workers=4,
    )

    pipeline._collect(xr.Dataset({"value": ("time", np.arange(3))}))

    cluster.assert_called_once_with("off", n_workers=4)
    assert writer.call_args.kwargs["storage_options"] == {"profile": "publication"}


@pytest.mark.parametrize("flatten", [False, True])
def test_native_publication_preserves_vector_basis_and_angles(
    monkeypatch, tmp_path, flatten
):
    angles = np.array([[0.0, 30.0, 90.0], [-45.0, 120.0, 180.0]])
    source = xr.Dataset(
        {
            name: (("time", "lev", "y", "x"), np.full((1, 2, 2, 3), value))
            for name, value in [
                ("uo", 2.0),
                ("vo", 3.0),
                ("thetao", 10.0),
                ("so", 35.0),
            ]
        }
        | {
            name: (("time", "y", "x"), np.full((1, 2, 3), value))
            for name, value in [
                ("tauuo", 4.0),
                ("tauvo", 5.0),
                ("zos", 0.1),
                ("hfds", 1.0),
                ("wfo", 0.0),
            ]
        },
        coords={
            "time": [0],
            "lev": [5.0, 15.0],
            "x": [0, 1, 2],
            "y": [0, 1],
            "dz": ("lev", [10.0, 10.0]),
            "lon": (("y", "x"), np.zeros((2, 3))),
            "lat": (("y", "x"), np.zeros((2, 3))),
            "lon_b": (("y_b", "x_b"), np.zeros((3, 4))),
            "lat_b": (("y_b", "x_b"), np.zeros((3, 4))),
            "areacello": (("y", "x"), np.ones((2, 3))),
            "wetmask": (("lev", "y", "x"), np.ones((2, 2, 3), dtype=bool)),
            "angle": (("y", "x"), angles),
        },
    )
    source.uo.attrs["standard_name"] = "eastward_sea_water_velocity"
    monkeypatch.setattr(main, "om4_preprocessing", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        main,
        "rotate_vectors",
        lambda *args: pytest.fail("Native vectors must not rotate"),
    )
    output = tmp_path / "native.zarr"
    CLI(
        output_path=str(output),
        skip_regridding=True,
        skip_spatial_filtering=True,
        skip_validation=True,
        skip_flattening=not flatten,
    ).om4("raw", "static", "supergrid", "unused")
    with xr.open_zarr(output, chunks=None) as actual:
        np.testing.assert_array_equal(actual.angle, angles)
        assert actual.angle.dims == ("y", "x")
        assert actual.angle.attrs["units"] == "degrees"
        assert "Counterclockwise" in actual.angle.attrs["long_name"]
        for name, value in [("uo", 2), ("vo", 3), ("tauuo", 4), ("tauvo", 5)]:
            fields = (
                [f"{name}_{level}" for level in range(2)]
                if flatten and name in ("uo", "vo")
                else [name]
            )
            for field in fields:
                np.testing.assert_array_equal(actual[field], value)
                assert actual[field].attrs["vector_basis"] == "native_grid"
                assert "standard_name" not in actual[field].attrs
        provenance = actual.attrs["m2lines/preprocessing"]
        assert provenance["vector_rotation"] == []
        assert provenance["vector_basis"] == "native_grid"
        assert provenance["target_grid"] is None
