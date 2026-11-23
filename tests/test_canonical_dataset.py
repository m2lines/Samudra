# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
from zarr.storage import DirectoryStore  # type: ignore[import-untyped]

from samudra.config import JulianDate, Om4TimeConfig
from samudra.datasets import InferenceDataset, TorchTrainDataset, TrainingWindows
from samudra.utils.data import (
    CanonicalReadRequest,
    CanonicalSource,
    XarrayCanonicalReader,
)
from tests.conftest import TEST_DATA_LAYOUT, canonicalize_mock_om4


def _equivalent_om4_sources(
    compact_store_root: Path | None = None,
    *,
    dask_chunks: bool = True,
    time_chunk: int = 1,
    depth_chunk: int | None = None,
) -> tuple[CanonicalSource, CanonicalSource]:
    time = xr.CFTimeIndex(
        [cftime.datetime(2000, 1, day, 12, calendar="julian") for day in range(1, 9)]
    )
    lat = np.array([-1.0, 1.0])
    lon = np.array([0.5, 1.5, 2.5])
    levels = len(TEST_DATA_LAYOUT.depth_levels)
    shape = (len(time), levels, len(lat), len(lon))
    depth_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    surface_data = np.arange(len(time) * len(lat) * len(lon), dtype=np.float32).reshape(
        len(time), len(lat), len(lon)
    )
    wetmask = np.ones((levels, len(lat), len(lon)), dtype=bool)
    wetmask[2, 0, 0] = False

    compact_data = xr.Dataset(
        {
            "so": (("time", "lev", "lat", "lon"), depth_data),
            "zos": (("time", "lat", "lon"), surface_data + 1),
            "hfds": (("time", "lat", "lon"), surface_data + 2),
            "wetmask": (
                ("lev", "lat", "lon"),
                wetmask,
            ),
        },
        coords={"time": time, "lev": np.arange(levels), "lat": lat, "lon": lon},
    )
    compact_means = compact_data[["so", "zos", "hfds"]].mean(("time", "lat", "lon"))
    compact_stds = compact_data[["so", "zos", "hfds"]].std(("time", "lat", "lon"))

    flat_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        **{
            f"so_{level}": (
                ("time", "lat", "lon"),
                depth_data[:, level],
            )
            for level in range(levels)
        },
        **{
            f"mask_{level}": (("lat", "lon"), wetmask[level]) for level in range(levels)
        },
        "zos": (("time", "lat", "lon"), surface_data + 1),
        "hfds": (("time", "lat", "lon"), surface_data + 2),
    }
    flat_data = xr.Dataset(flat_vars, coords={"time": time, "lat": lat, "lon": lon})
    data_channels = [f"so_{level}" for level in range(levels)] + ["zos", "hfds"]
    flat_means = flat_data[data_channels].mean(("time", "lat", "lon"))
    flat_stds = flat_data[data_channels].std(("time", "lat", "lon"))

    prognostic = ["so_0", "so_2", "zos"]
    boundary = ["hfds"]
    flat_canonical = canonicalize_mock_om4(flat_data, flat_means, flat_stds)
    flat = CanonicalSource.from_datasets(
        *flat_canonical,
        name="flat",
        data_layout=TEST_DATA_LAYOUT,
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
    )
    if compact_store_root is None:
        compact_canonical = canonicalize_mock_om4(
            compact_data, compact_means, compact_stds
        )
        compact = CanonicalSource.from_datasets(
            *compact_canonical,
            name="compact",
            data_layout=TEST_DATA_LAYOUT,
            prognostic_var_names=prognostic,
            boundary_var_names=boundary,
        )
    else:
        data_path = compact_store_root / "compact-data.zarr"
        means_path = compact_store_root / "compact-means.zarr"
        stds_path = compact_store_root / "compact-stds.zarr"
        compact_data.to_zarr(
            data_path,
            mode="w",
            consolidated=True,
            encoding={
                "so": {
                    "chunks": (time_chunk, depth_chunk or levels, len(lat), len(lon))
                }
            },
        )
        compact_means.to_zarr(means_path, mode="w", consolidated=True)
        compact_stds.to_zarr(stds_path, mode="w", consolidated=True)
        compact_canonical = canonicalize_mock_om4(
            xr.open_zarr(data_path, chunks={} if dask_chunks else None),
            xr.open_zarr(means_path, chunks={} if dask_chunks else None),
            xr.open_zarr(stds_path, chunks={} if dask_chunks else None),
        )
        compact = CanonicalSource.from_datasets(
            *compact_canonical,
            data_layout=TEST_DATA_LAYOUT,
            prognostic_var_names=prognostic,
            boundary_var_names=boundary,
        )
    return flat, compact


def test_flat_and_compact_om4_have_identical_canonical_contract() -> None:
    flat, compact = _equivalent_om4_sources()

    assert flat.channels == compact.channels == ("so_0", "so_2", "zos", "hfds")
    np.testing.assert_array_equal(flat.time, compact.time)
    torch.testing.assert_close(flat.resolution[0], compact.resolution[0])
    torch.testing.assert_close(flat.resolution[1], compact.resolution[1])
    np.testing.assert_allclose(
        flat.statistics(flat.channels).mean,
        compact.statistics(compact.channels).mean,
    )
    np.testing.assert_allclose(
        flat.statistics(flat.channels).std,
        compact.statistics(compact.channels).std,
    )
    torch.testing.assert_close(flat.masks.prognostic, compact.masks.prognostic)
    torch.testing.assert_close(flat.masks.boundary, compact.masks.boundary)

    indices = np.array([[0, 2], [1, 3]], dtype=np.int64)
    np.testing.assert_allclose(
        flat.read(indices, flat.channels), compact.read(indices, compact.channels)
    )


def test_compact_zarr_has_the_same_canonical_contract(tmp_path: Path) -> None:
    flat, compact = _equivalent_om4_sources(tmp_path)
    indices = np.array([[0, 2], [1, 3]], dtype=np.int64)

    assert compact.channels == flat.channels
    np.testing.assert_array_equal(compact.time, flat.time)
    np.testing.assert_allclose(
        compact.statistics(compact.channels).mean,
        flat.statistics(flat.channels).mean,
    )
    np.testing.assert_allclose(
        compact.statistics(compact.channels).std,
        flat.statistics(flat.channels).std,
    )
    np.testing.assert_allclose(
        compact.read(indices, compact.channels), flat.read(indices, flat.channels)
    )


def test_channel_request_and_time_slice_are_immutable() -> None:
    source, _ = _equivalent_om4_sources()
    original_channels = source.channels
    original_time = source.time.copy()

    requested_channels = ("hfds", "so_2")
    sliced = source.slice_time(
        Om4TimeConfig(start=JulianDate("2000-01-02"), end=JulianDate("2000-01-03"))
    )

    assert source.channels == original_channels
    np.testing.assert_array_equal(source.time, original_time)
    assert sliced.channels == source.channels
    assert sliced.time.size == 2
    np.testing.assert_allclose(
        sliced.read(np.array([0]), requested_channels),
        source.read(np.array([1]), requested_channels),
    )


def test_read_request_owns_immutable_integer_indices() -> None:
    indices = np.array([0, 2], dtype=np.int32)
    request = CanonicalReadRequest(indices, ("so_0",))
    indices[0] = 1

    assert request.channels == ("so_0",)
    assert request.time_indices.dtype == np.int64
    np.testing.assert_array_equal(request.time_indices, [0, 2])
    assert not request.time_indices.flags.writeable


def test_source_factory_installs_reader_without_mutating_input() -> None:
    source, _ = _equivalent_om4_sources()
    data, means, stds = source._xarray_datasets_for_testing()
    selected_time = Om4TimeConfig(
        start=JulianDate("2000-01-02"), end=JulianDate("2000-01-03")
    )
    built = CanonicalSource.from_datasets(
        data,
        means,
        stds,
        data_layout=source.data_layout,
        prognostic_var_names=["so_0", "so_2", "zos"],
        boundary_var_names=["hfds"],
        reader_factory=lambda source: XarrayCanonicalReader.from_source_data(
            source
        ).slice_time(selected_time),
    )

    assert source.time.size == 8
    assert built.time.size == 2
    assert built.channels == source.channels
    np.testing.assert_array_equal(
        built.read(np.array([0, 1]), ("hfds",)),
        source.read(np.array([1, 2]), ("hfds",)),
    )


def test_flat_and_compact_cpu_training_and_inference_are_identical() -> None:
    flat, compact = _equivalent_om4_sources()
    prognostic = ["so_0", "so_2", "zos"]
    boundary = ["hfds"]

    def train_dataset(
        source: CanonicalSource, *, concurrent: bool = False
    ) -> TorchTrainDataset:
        return TorchTrainDataset(
            TrainingWindows(
                input_source=source,
                label_source=None,
                prognostic_var_names=prognostic,
                boundary_var_names=boundary,
                input_steps=2,
                output_steps=2,
                steps=1,
                normalize_before_mask=True,
                masked_fill_value=0.0,
            ),
            concurrent_compute_=concurrent,
        )

    flat_train = train_dataset(flat)
    compact_train = train_dataset(compact)
    flat_raw = flat_train[0]
    compact_raw = compact_train[0]
    for flat_step, compact_step in zip(flat_raw.steps, compact_raw.steps, strict=True):
        for flat_tensor, compact_tensor in zip(flat_step, compact_step, strict=True):
            torch.testing.assert_close(flat_tensor, compact_tensor)
    flat_batch = flat_train.to_model_batch(flat_raw, torch.device("cpu"))
    compact_batch = compact_train.to_model_batch(compact_raw, torch.device("cpu"))
    for flat_tensor, compact_tensor in zip(
        flat_batch[0], compact_batch[0], strict=True
    ):
        torch.testing.assert_close(flat_tensor, compact_tensor)

    # The concurrent CPU path submits whole canonical read requests, not one task
    # per compact level. It must retain the same combined-read semantics.
    compact_concurrent = train_dataset(compact, concurrent=True)[0]
    for compact_tensor, concurrent_tensor in zip(
        compact_raw.steps[0], compact_concurrent.steps[0], strict=True
    ):
        torch.testing.assert_close(compact_tensor, concurrent_tensor)

    def inference_dataset(source: CanonicalSource) -> InferenceDataset:
        return InferenceDataset(
            source,
            prognostic_var_names=prognostic,
            boundary_var_names=boundary,
            input_steps=2,
            output_steps=2,
            normalize_before_mask=False,
            masked_fill_value=-1.0,
            long_rollout=False,
        )

    flat_inference = inference_dataset(flat)
    compact_inference = inference_dataset(compact)
    for flat_tensor, compact_tensor in zip(
        flat_inference[0], compact_inference[0], strict=True
    ):
        torch.testing.assert_close(flat_tensor, compact_tensor)


@pytest.mark.parametrize("dask_chunks", [False, True])
@pytest.mark.parametrize("time_chunk", [1, 2])
@pytest.mark.parametrize("depth_chunk", [1, 19])
def test_compact_om4_reads_each_requested_chunk_once(
    tmp_path, monkeypatch, dask_chunks, time_chunk, depth_chunk
):
    reads = []
    original_getitem = DirectoryStore.__getitem__

    def counted_getitem(store, key):
        if (
            key.startswith("so/")
            and not key.startswith("so/.")
            and str(store.path).endswith("compact-data.zarr")
        ):
            reads.append(key)
        return original_getitem(store, key)

    monkeypatch.setattr(DirectoryStore, "__getitem__", counted_getitem)
    flat, compact = _equivalent_om4_sources(
        tmp_path,
        dask_chunks=dask_chunks,
        time_chunk=time_chunk,
        depth_chunk=depth_chunk,
    )
    assert reads == []  # Construction must not load prognostic data.
    assert isinstance(compact.reader, XarrayCanonicalReader)
    assert "so" in compact.reader.data and "so_0" not in compact.reader.data
    assert (compact.reader.data.so.chunks is not None) == dask_chunks
    assert compact.metadata == flat.metadata
    assert compact.coordinates().keys() == flat.coordinates().keys()
    period = Om4TimeConfig(start=JulianDate("2000-01-02"), end=JulianDate("2000-01-07"))
    flat, compact = flat.slice_time(period), compact.slice_time(period)
    indices = np.array([[2, 0], [2, 5]], dtype=np.int64)
    channels = ("so_2", "hfds", "so_0", "zos", "so_2", "hfds")
    actual = compact.read(indices, channels)
    np.testing.assert_array_equal(actual, flat.read(indices, channels))
    expected = sorted(
        f"so/{time}.{level}.0.0"
        for time in sorted({time // time_chunk for time in (1, 3, 6)})
        for level in sorted({level // depth_chunk for level in (0, 2)})
    )
    assert sorted(reads) == expected
    reads.clear()
    np.testing.assert_array_equal(compact.read(indices, channels), actual)
    assert sorted(reads) == expected  # No request-spanning cache or hidden load.


def test_compact_source_factory_receives_arrays_without_a_reader(tmp_path):
    from samudra.utils.data import XarraySourceData

    _, source = _equivalent_om4_sources(tmp_path, dask_chunks=False)
    assert isinstance(source.reader, XarrayCanonicalReader)
    seen = []

    def build_reader(arrays):
        assert type(arrays) is XarraySourceData
        assert "so" in arrays.data and "so_0" not in arrays.data
        assert arrays.data.so.chunks is None
        seen.append(arrays.channels)
        return XarrayCanonicalReader.from_source_data(arrays)

    CanonicalSource.from_datasets(
        source.reader.data,
        source.reader.means,
        source.reader.stds,
        data_layout=source.data_layout,
        prognostic_var_names=["so_0", "so_2", "zos"],
        boundary_var_names=["hfds"],
        reader_factory=build_reader,
    )
    assert seen == [source.channels]
