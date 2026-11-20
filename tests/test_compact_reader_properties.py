# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Independent array and physical-chunk oracles for compact Python reads."""

from collections import Counter
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st
from zarr.storage import DirectoryStore  # type: ignore[import-untyped]

from samudra.utils.data import CanonicalReadRequest, XarrayCanonicalReader


@st.composite
def compact_requests(draw):
    times = draw(st.lists(st.integers(0, 3), max_size=6))
    channels = draw(
        st.lists(
            st.sampled_from(["a_0", "a_1", "a_2", "b_0", "b_1", "b_2", "s"]), max_size=8
        )
    )
    return (
        times,
        channels,
        draw(st.integers(1, 4)),
        draw(st.integers(1, 3)),
        draw(st.booleans()),
        draw(st.booleans()),
    )


@given(compact_requests())
def test_compact_reads_match_array_oracle_and_touch_each_chunk_once(case):
    times, channels, time_chunk, level_chunk, flat_b, transpose = case
    values = np.arange(4 * 3 * 2 * 3, dtype=np.float32).reshape(4, 3, 2, 3)
    surface = values[:, 0] - 10
    arrays = {
        "a": (("time", "lev", "lat", "lon"), values),
        "s": (("time", "lat", "lon"), surface),
    }
    if flat_b:
        arrays.update(
            {f"b_{k}": (("time", "lat", "lon"), values[:, k] + 200) for k in range(3)}
        )
    else:
        arrays["b"] = (("time", "lev", "lat", "lon"), values + 200)
    data = xr.Dataset(
        arrays,
        coords={
            "time": np.arange(4),
            "lev": np.arange(3),
            "lat": np.arange(2),
            "lon": np.arange(3),
        },
    )
    if transpose:
        data = data.transpose("lon", "lat", "lev", "time", missing_dims="ignore")
    chunks = {"time": time_chunk, "lev": level_chunk, "lat": 1, "lon": 2}
    expected = np.empty((len(times), len(channels), 2, 3), dtype=np.float32)
    for column, channel in enumerate(channels):
        if channel == "s":
            expected[:, column] = surface[times]
        else:
            family, level = channel.split("_")
            expected[:, column] = values[times, int(level)] + (
                200 if family == "b" else 0
            )
    expected_chunks = set()
    for channel in channels:
        if channel == "s" or (flat_b and channel.startswith("b_")):
            name, level = channel, None
        else:
            name, level_text = channel.split("_")
            level = int(level_text)
        for time in times:
            for lat in range(2):
                for lon in range(2):
                    coords = {"time": time // time_chunk, "lat": lat, "lon": lon}
                    if level is not None:
                        coords["lev"] = level // level_chunk
                    key = ".".join(str(coords[str(dim)]) for dim in data[name].dims)
                    expected_chunks.add(f"{name}/{key}")
    with TemporaryDirectory() as root:
        data.to_zarr(
            root,
            consolidated=True,
            encoding={
                name: {"chunks": tuple(chunks[str(dim)] for dim in var.dims)}
                for name, var in data.data_vars.items()
            },
        )
        with xr.open_zarr(root, chunks=None) as stored:
            reader = XarrayCanonicalReader(
                stored,
                xr.Dataset(),
                xr.Dataset(),
                tuple(["a_0", "a_1", "a_2", "b_0", "b_1", "b_2", "s"]),
            )
            reads: Counter[str] = Counter()
            original = DirectoryStore.__getitem__

            def counted(store, key):
                if key.split("/")[0] in arrays and not key.split("/")[-1].startswith(
                    "."
                ):
                    reads[key] += 1
                return original(store, key)

            with (
                patch.object(DirectoryStore, "__getitem__", counted),
                patch.object(
                    xr.DataArray, "chunk", side_effect=AssertionError("implicit Dask")
                ),
                patch.object(
                    xr.Dataset, "chunk", side_effect=AssertionError("implicit Dask")
                ),
            ):
                actual = reader.read(
                    CanonicalReadRequest(
                        np.asarray(times, dtype=np.int64), tuple(channels)
                    )
                )
            np.testing.assert_array_equal(actual, expected)
            assert set(reads) == expected_chunks
            assert all(count == 1 for count in reads.values())
            assert all(var.chunks is None for var in reader.data.data_vars.values())


@given(st.integers(1, 5), st.sampled_from(["m/s", "K", "psu"]))
def test_compact_channel_metadata_preserves_variable_attributes(levels, units):
    from samudra.constants import build_om4_layout

    channels = tuple(f"a_{level}" for level in range(levels))
    data = xr.Dataset(
        {
            "a": (
                ("time", "lev", "lat", "lon"),
                np.zeros((1, levels, 1, 1), dtype=np.float32),
            )
        }
    )
    data.a.attrs.update(long_name="Physical field", units=units)
    reader = XarrayCanonicalReader(data, xr.Dataset(), xr.Dataset(), channels)
    metadata = reader.metadata(build_om4_layout("thetao_1", "hfds"))
    for channel in channels:
        assert metadata[channel] == {"long_name": "Physical field", "units": units}


@given(
    st.sampled_from([0.25, 0.5, 2.0]),
    st.sampled_from([-10.0, 0.0, 100.0]),
    st.lists(st.integers(0, 3), min_size=1, max_size=6),
    st.lists(st.integers(0, 2), min_size=1, max_size=5),
)
def test_compact_cf_decoding_matches_physical_units(scale, offset, times, levels):
    values = (np.arange(4 * 3 * 2 * 3).reshape(4, 3, 2, 3) * scale + offset).astype(
        np.float32
    )
    values[0, 0, 0, 0] = np.nan
    data = xr.Dataset({"a": (("time", "lev", "lat", "lon"), values)})
    with TemporaryDirectory() as root:
        data.to_zarr(
            root,
            consolidated=True,
            encoding={
                "a": {
                    "dtype": "int16",
                    "_FillValue": -32768,
                    "scale_factor": scale,
                    "add_offset": offset,
                    "chunks": (2, 3, 1, 2),
                }
            },
        )
        with xr.open_zarr(root, chunks=None) as stored:
            reader = XarrayCanonicalReader(
                stored, xr.Dataset(), xr.Dataset(), ("a_0", "a_1", "a_2")
            )
            actual = reader.read(
                CanonicalReadRequest(
                    np.array(times, dtype=np.int64),
                    tuple(f"a_{level}" for level in levels),
                )
            )
            expected = values[np.array(times)[:, None], np.array(levels)[None, :]]
            np.testing.assert_array_equal(actual, expected)
