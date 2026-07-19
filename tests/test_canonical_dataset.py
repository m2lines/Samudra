# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cftime
import numpy as np
import torch
import xarray as xr

from samudra.config import JulianDate, TimeConfig
from samudra.datasets import InferenceDataset, TorchTrainDataset
from samudra.utils.data import CanonicalDataset, CanonicalReadRequest
from samudra.utils.location import LocalLocation
from tests.conftest import TEST_DATASET_SPEC


def _equivalent_om4_sources(
    compact_store_root: Path | None = None,
) -> tuple[CanonicalDataset, CanonicalDataset]:
    time = xr.CFTimeIndex(
        [cftime.datetime(2000, 1, day, 12, calendar="julian") for day in range(1, 9)]
    )
    lat = np.array([-1.0, 1.0])
    lon = np.array([0.5, 1.5, 2.5])
    levels = len(TEST_DATASET_SPEC.depth_levels)
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
            TEST_DATASET_SPEC.mask_all_levels_var: (
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
    flat = CanonicalDataset.from_datasets(
        flat_data,
        flat_means,
        flat_stds,
        name="flat",
        dataset_spec=TEST_DATASET_SPEC,
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
    )
    if compact_store_root is None:
        compact = CanonicalDataset.from_datasets(
            compact_data,
            compact_means,
            compact_stds,
            name="compact",
            dataset_spec=TEST_DATASET_SPEC,
            prognostic_var_names=prognostic,
            boundary_var_names=boundary,
        )
    else:
        data_path = compact_store_root / "compact-data.zarr"
        means_path = compact_store_root / "compact-means.zarr"
        stds_path = compact_store_root / "compact-stds.zarr"
        compact_data.to_zarr(data_path, mode="w", consolidated=True)
        compact_means.to_zarr(means_path, mode="w", consolidated=True)
        compact_stds.to_zarr(stds_path, mode="w", consolidated=True)
        compact = CanonicalDataset.from_locations(
            data_location=LocalLocation(path=data_path),
            means_location=LocalLocation(path=means_path),
            stds_location=LocalLocation(path=stds_path),
            dataset_spec=TEST_DATASET_SPEC,
            prognostic_var_names=prognostic,
            boundary_var_names=boundary,
            static_data_vars=None,
            use_dask=True,
        )
    return flat, compact


def test_flat_and_compact_om4_have_identical_canonical_contract() -> None:
    flat, compact = _equivalent_om4_sources()

    assert flat.channels == compact.channels == ("so_0", "so_2", "zos", "hfds")
    np.testing.assert_array_equal(flat.time, compact.time)
    torch.testing.assert_close(flat.resolution[0], compact.resolution[0])
    torch.testing.assert_close(flat.resolution[1], compact.resolution[1])
    np.testing.assert_allclose(flat.statistics.mean, compact.statistics.mean)
    np.testing.assert_allclose(flat.statistics.std, compact.statistics.std)
    torch.testing.assert_close(flat.masks.prognostic, compact.masks.prognostic)
    torch.testing.assert_close(flat.masks.boundary, compact.masks.boundary)

    request = CanonicalReadRequest(np.array([[0, 2], [1, 3]], dtype=np.int64))
    np.testing.assert_allclose(flat.read(request).values, compact.read(request).values)


def test_compact_zarr_has_the_same_canonical_contract(tmp_path: Path) -> None:
    flat, compact = _equivalent_om4_sources(tmp_path)
    request = CanonicalReadRequest(np.array([[0, 2], [1, 3]], dtype=np.int64))

    assert compact.channels == flat.channels
    np.testing.assert_array_equal(compact.time, flat.time)
    np.testing.assert_allclose(compact.statistics.mean, flat.statistics.mean)
    np.testing.assert_allclose(compact.statistics.std, flat.statistics.std)
    np.testing.assert_allclose(compact.read(request).values, flat.read(request).values)


def test_canonical_selection_and_time_slice_are_immutable_views() -> None:
    source, _ = _equivalent_om4_sources()
    original_channels = source.channels
    original_time = source.time.copy()

    selected = source.select_channels(["hfds", "so_2"], prefix="selected")
    sliced = selected.slice_time(
        TimeConfig(start=JulianDate("2000-01-02"), end=JulianDate("2000-01-03"))
    )

    assert source.channels == original_channels
    np.testing.assert_array_equal(source.time, original_time)
    assert selected.channels == ("hfds", "so_2")
    assert sliced.channels == selected.channels
    assert sliced.time.size == 2
    np.testing.assert_allclose(
        sliced.read(CanonicalReadRequest(np.array([0]))).values,
        selected.read(CanonicalReadRequest(np.array([1]))).values,
    )


def test_flat_and_compact_cpu_training_and_inference_are_identical() -> None:
    flat, compact = _equivalent_om4_sources()
    prognostic = ["so_0", "so_2", "zos"]
    boundary = ["hfds"]

    def train_dataset(
        source: CanonicalDataset, *, concurrent: bool = False
    ) -> TorchTrainDataset:
        return TorchTrainDataset(
            src=source,
            dst=None,
            prognostic_var_names=prognostic,
            boundary_var_names=boundary,
            hist=1,
            steps=1,
            normalize_before_mask=True,
            masked_fill_value=0.0,
            concurrent_compute_=concurrent,
        )

    flat_train = train_dataset(flat)
    compact_train = train_dataset(compact)
    flat_raw = flat_train[0]
    compact_raw = compact_train[0]
    for flat_step, compact_step in zip(
        flat_raw.raw_data, compact_raw.raw_data, strict=True
    ):
        for flat_tensor, compact_tensor in zip(flat_step, compact_step, strict=True):
            torch.testing.assert_close(flat_tensor, compact_tensor)
    flat_batch = flat_train.to_train_data(flat_raw, torch.device("cpu"))
    compact_batch = compact_train.to_train_data(compact_raw, torch.device("cpu"))
    for flat_tensor, compact_tensor in zip(
        flat_batch[0], compact_batch[0], strict=True
    ):
        torch.testing.assert_close(flat_tensor, compact_tensor)

    # The concurrent CPU path submits whole canonical read requests, not one task
    # per compact level. It must retain the same combined-read semantics.
    compact_concurrent = train_dataset(compact, concurrent=True)[0]
    for compact_tensor, concurrent_tensor in zip(
        compact_raw.raw_data[0], compact_concurrent.raw_data[0], strict=True
    ):
        torch.testing.assert_close(compact_tensor, concurrent_tensor)

    def inference_dataset(source: CanonicalDataset) -> InferenceDataset:
        return InferenceDataset(
            source,
            prognostic_var_names=prognostic,
            boundary_var_names=boundary,
            hist=1,
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
