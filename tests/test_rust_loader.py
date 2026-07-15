# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

rust_loader = pytest.importorskip("samudra_rust_loader")


@pytest.fixture
def flat_om4_store(tmp_path):
    values = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    values[2, 1, 3] = np.nan
    dataset = xr.Dataset(
        {
            "first": (("time", "lat", "lon"), values),
            "second": (("time", "lat", "lon"), values + np.float32(100)),
        }
    )
    path = tmp_path / "flat-om4.zarr"
    dataset.to_zarr(path, mode="w", consolidated=False)
    return path, dataset


def test_flat_om4_reader_matches_xarray(flat_om4_store):
    path, dataset = flat_om4_store
    reader = rust_loader.FlatOm4Reader(path, ["first", "second"], 2)

    actual = np.asarray(reader.read([3, 0, 2], ["second", "first"]))
    expected = (
        dataset[["second", "first"]]
        .isel(time=[3, 0, 2])
        .to_array()
        .transpose("time", "variable", "lat", "lon")
        .to_numpy()
    )

    assert reader.shape == (4, 3, 5)
    np.testing.assert_array_equal(actual, expected)


def test_flat_om4_reader_reads_into_c_contiguous_array(flat_om4_store):
    path, dataset = flat_om4_store
    reader = rust_loader.FlatOm4Reader(path, ["first", "second"], 2)
    actual = np.empty((3, 2, 3, 5), dtype=np.float32)

    reader.read_into([3, 0, 2], ["second", "first"], actual)

    expected = (
        dataset[["second", "first"]]
        .isel(time=[3, 0, 2])
        .to_array()
        .transpose("time", "variable", "lat", "lon")
        .to_numpy()
    )
    np.testing.assert_array_equal(actual, expected)


def test_flat_om4_reader_rejects_wrong_output_shape(flat_om4_store):
    path, _ = flat_om4_store
    reader = rust_loader.FlatOm4Reader(path, ["first"], 1)
    output = np.empty((1, 1, 3, 4), dtype=np.float32)

    with pytest.raises(RuntimeError, match="output has shape"):
        reader.read_into([0], ["first"], output)


@pytest.mark.parametrize(
    ("indexes", "match"),
    [([-1], "non-negative"), ([4], "out of bounds")],
)
def test_flat_om4_reader_rejects_invalid_index(flat_om4_store, indexes, match):
    path, _ = flat_om4_store
    reader = rust_loader.FlatOm4Reader(path, ["first"], 1)

    with pytest.raises(RuntimeError, match=match):
        reader.read(indexes, ["first"])


def test_flat_om4_reader_rejects_variable_not_opened(flat_om4_store):
    path, _ = flat_om4_store
    reader = rust_loader.FlatOm4Reader(path, ["first"], 1)

    with pytest.raises(RuntimeError, match="was not opened"):
        reader.read([0], ["second"])


def test_flat_om4_reader_rejects_non_float32(tmp_path):
    path = tmp_path / "float64.zarr"
    xr.Dataset(
        {"value": (("time", "lat", "lon"), np.zeros((2, 3, 4), dtype=np.float64))}
    ).to_zarr(path, mode="w", consolidated=False)

    with pytest.raises(RuntimeError, match="requires float32"):
        rust_loader.FlatOm4Reader(path, ["value"], 1)


def test_flat_om4_reader_rejects_non_flat_shape(tmp_path):
    path = tmp_path / "rank-four.zarr"
    xr.Dataset(
        {
            "value": (
                ("time", "level", "lat", "lon"),
                np.zeros((2, 1, 3, 4), dtype=np.float32),
            )
        }
    ).to_zarr(path, mode="w", consolidated=False)

    with pytest.raises(RuntimeError, match=r"expected \(time, lat, lon\)"):
        rust_loader.FlatOm4Reader(path, ["value"], 1)
