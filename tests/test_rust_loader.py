# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

rust_loader = pytest.importorskip("samudra_rust_loader")


def open_reader(path, variables, max_concurrent_reads=2):
    pool = rust_loader.FlatOm4ReadPool(max_concurrent_reads)
    return rust_loader.FlatOm4Reader(path, variables, pool)


def open_compact_reader(path, variables, max_concurrent_reads=2):
    pool = rust_loader.FlatOm4ReadPool(max_concurrent_reads)
    return rust_loader.CompactOm4Reader(path, variables, pool)


def test_flat_om4_read_pool_requires_positive_concurrency():
    with pytest.raises(RuntimeError, match="must be positive"):
        rust_loader.FlatOm4ReadPool(0)


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


@pytest.fixture
def compact_om4_store(tmp_path):
    depth = np.arange(4 * 3 * 3 * 5, dtype=np.float32).reshape(4, 3, 3, 5)
    depth[2, 1, 1, 3] = np.nan
    surface = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    dataset = xr.Dataset(
        {
            "thetao": (("lev", "time", "y", "x"), depth.transpose(1, 0, 2, 3)),
            "so": (
                ("lev", "time", "y", "x"),
                (depth + np.float32(1000)).transpose(1, 0, 2, 3),
            ),
            "zos": (("time", "y", "x"), surface + np.float32(2000)),
            "hfds_anomalies": (
                ("time", "y", "x"),
                surface + np.float32(3000),
            ),
        }
    )
    path = tmp_path / "compact-om4.zarr"
    dataset.to_zarr(path, mode="w", consolidated=False)
    return path, dataset


def test_flat_om4_reader_matches_xarray(flat_om4_store):
    path, dataset = flat_om4_store
    reader = open_reader(path, ["first", "second"])

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
    reader = open_reader(path, ["first", "second"])
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
    reader = open_reader(path, ["first"])
    output = np.empty((1, 1, 3, 4), dtype=np.float32)

    with pytest.raises(RuntimeError, match="output has shape"):
        reader.read_into([0], ["first"], output)


@pytest.mark.parametrize(
    ("indexes", "match"),
    [([-1], "non-negative"), ([4], "out of bounds")],
)
def test_flat_om4_reader_rejects_invalid_index(flat_om4_store, indexes, match):
    path, _ = flat_om4_store
    reader = open_reader(path, ["first"])

    with pytest.raises(RuntimeError, match=match):
        reader.read(indexes, ["first"])


def test_flat_om4_reader_rejects_variable_not_opened(flat_om4_store):
    path, _ = flat_om4_store
    reader = open_reader(path, ["first"])

    with pytest.raises(RuntimeError, match="was not opened"):
        reader.read([0], ["second"])


def test_flat_om4_reader_rejects_non_float32(tmp_path):
    path = tmp_path / "float64.zarr"
    xr.Dataset(
        {"value": (("time", "lat", "lon"), np.zeros((2, 3, 4), dtype=np.float64))}
    ).to_zarr(path, mode="w", consolidated=False)

    with pytest.raises(RuntimeError, match="requires float32"):
        open_reader(path, ["value"])


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
        open_reader(path, ["value"])


def test_flat_om4_reader_rejects_zero_sized_spatial_extent(tmp_path):
    path = tmp_path / "zero-lat.zarr"
    xr.Dataset(
        {
            "value": (
                ("time", "lat", "lon"),
                np.zeros((2, 0, 4), dtype=np.float32),
            )
        }
    ).to_zarr(path, mode="w", consolidated=False)

    with pytest.raises(RuntimeError, match="zero-sized extent"):
        open_reader(path, ["value"])


def test_flat_om4_reader_rejects_transposed_spatial_dimensions(tmp_path):
    path = tmp_path / "transposed.zarr"
    xr.Dataset(
        {
            "value": (
                ("time", "lon", "lat"),
                np.zeros((2, 3, 3), dtype=np.float32),
            )
        }
    ).to_zarr(path, mode="w", consolidated=False)

    with pytest.raises(RuntimeError, match="expected \\(time, lat/y, lon/x\\)"):
        open_reader(path, ["value"])


def test_flat_om4_reader_accepts_y_x_dimension_aliases(tmp_path):
    path = tmp_path / "y-x.zarr"
    xr.Dataset(
        {
            "value": (
                ("time", "y", "x"),
                np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            )
        }
    ).to_zarr(path, mode="w", consolidated=False)

    reader = open_reader(path, ["value"])

    assert reader.shape == (2, 3, 4)


def test_compact_om4_reader_translates_channels_and_preserves_order(
    compact_om4_store,
):
    path, dataset = compact_om4_store
    channels = ["thetao_2", "zos", "so_0", "thetao_1", "hfds_anomalies"]
    reader = open_compact_reader(path, channels)

    actual = np.asarray(reader.read([3, 0, 2], channels))
    expected = np.stack(
        [
            dataset.thetao.isel(time=[3, 0, 2], lev=2).to_numpy(),
            dataset.zos.isel(time=[3, 0, 2]).to_numpy(),
            dataset.so.isel(time=[3, 0, 2], lev=0).to_numpy(),
            dataset.thetao.isel(time=[3, 0, 2], lev=1).to_numpy(),
            dataset.hfds_anomalies.isel(time=[3, 0, 2]).to_numpy(),
        ],
        axis=1,
    )

    assert reader.shape == (4, 3, 5)
    np.testing.assert_array_equal(actual, expected)


def test_compact_om4_reader_reads_into_c_contiguous_array(compact_om4_store):
    path, dataset = compact_om4_store
    channels = ["so_1", "zos"]
    reader = open_compact_reader(path, channels)
    actual = np.empty((2, 2, 3, 5), dtype=np.float32)

    reader.read_into([2, 1], channels, actual)

    expected = np.stack(
        [
            dataset.so.isel(time=[2, 1], lev=1).to_numpy(),
            dataset.zos.isel(time=[2, 1]).to_numpy(),
        ],
        axis=1,
    )
    np.testing.assert_array_equal(actual, expected)


def test_compact_om4_reader_accepts_time_before_level(tmp_path):
    path = tmp_path / "time-level.zarr"
    xr.Dataset(
        {
            "thetao": (
                ("time", "lev", "lat", "lon"),
                np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2),
            )
        }
    ).to_zarr(path, mode="w", consolidated=False)

    channels = ["thetao_2", "thetao_0", "thetao_1"]
    reader = open_compact_reader(path, channels)

    assert reader.shape == (2, 2, 2)
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    np.testing.assert_array_equal(
        np.asarray(reader.read([1], channels))[0],
        data[1, [2, 0, 1]],
    )


def test_compact_om4_reader_rejects_level_out_of_range(compact_om4_store):
    path, _ = compact_om4_store

    with pytest.raises(RuntimeError, match="selects level 3.*has 3 levels"):
        open_compact_reader(path, ["thetao_3"])


def test_compact_om4_reader_rejects_depth_channel_backed_by_surface_array(
    compact_om4_store,
):
    path, _ = compact_om4_store

    with pytest.raises(RuntimeError, match=r"expected \(time, lev, lat, lon\)"):
        open_compact_reader(path, ["zos_0"])


def test_compact_om4_reader_rejects_surface_channel_backed_by_depth_array(
    compact_om4_store,
):
    path, _ = compact_om4_store

    with pytest.raises(RuntimeError, match=r"expected \(time, lat, lon\)"):
        open_compact_reader(path, ["thetao"])


def test_compact_om4_reader_rejects_missing_physical_variable(compact_om4_store):
    path, _ = compact_om4_store

    with pytest.raises(RuntimeError, match='opening compact OM4 variable "missing"'):
        open_compact_reader(path, ["missing_0"])


def test_compact_om4_reader_rejects_non_float32(tmp_path):
    path = tmp_path / "float64-compact.zarr"
    xr.Dataset(
        {
            "thetao": (
                ("lev", "time", "lat", "lon"),
                np.zeros((2, 2, 3, 4), dtype=np.float64),
            )
        }
    ).to_zarr(path, mode="w", consolidated=False)

    with pytest.raises(RuntimeError, match="compact OM4 Rust loading requires float32"):
        open_compact_reader(path, ["thetao_0"])


def test_compact_om4_reader_rejects_inconsistent_time_or_spatial_shape(tmp_path):
    path = tmp_path / "inconsistent-compact.zarr"
    dataset = xr.Dataset(
        {
            "thetao": (
                ("lev", "time", "lat", "lon"),
                np.zeros((2, 3, 3, 4), dtype=np.float32),
            ),
            "zos": (
                ("surface_time", "lat", "lon"),
                np.zeros((4, 3, 4), dtype=np.float32),
            ),
        }
    ).rename({"surface_time": "time_surface"})
    # Give each variable an independent time dimension, then rewrite the surface
    # metadata to the canonical name so only the extent mismatch is under test.
    dataset.to_zarr(path, mode="w", consolidated=False)
    attrs_path = path / "zos" / ".zattrs"
    attrs = attrs_path.read_text().replace("time_surface", "time")
    attrs_path.write_text(attrs)

    with pytest.raises(RuntimeError, match="time/spatial shape"):
        open_compact_reader(path, ["thetao_0", "zos"])


def test_compact_om4_reader_rejects_variable_not_opened(compact_om4_store):
    path, _ = compact_om4_store
    reader = open_compact_reader(path, ["thetao_0"])

    with pytest.raises(RuntimeError, match="canonical channel.*was not opened"):
        reader.read([0], ["thetao_1"])
