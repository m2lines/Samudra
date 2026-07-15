# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import threading
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import xarray as xr

pytest.importorskip("samudra_rust_loader")

from samudra.config import (
    DataSourceConfig,
    Om4DatasetConfig,
    RustDataLoadingConfig,
    TimeConfig,
    TrainConfig,
)
from samudra.constants import build_om4_spec
from samudra.datasets import TorchTrainDataset
from samudra.rust_data import RustBatchDataset, RustTrainDataLoader
from samudra.train import Trainer, _dataset_batch_group_keys
from samudra.utils.data import DataSource
from samudra.utils.location import LocalLocation, UnresolvedLocation
from samudra.utils.multiton import MultitonScope
from samudra.utils.samplers import DistributedEquivalenceGroupBatchSampler
from samudra.utils.train import collate_raw_train_data


@pytest.fixture
def flat_om4_source(tmp_path):
    spec = build_om4_spec(prognostic_vars_key="thetao_1", boundary_vars_key="hfds")
    time_size, lat_size, lon_size = 20, 3, 4
    coords = {
        "time": xr.date_range(
            "2000-01-01",
            periods=time_size,
            freq="5D",
            calendar="julian",
            use_cftime=True,
        ),
        "lat": np.linspace(-60, 60, lat_size),
        "lon": np.linspace(0, 270, lon_size),
    }
    prognostic = np.arange(time_size * lat_size * lon_size, dtype=np.float32).reshape(
        time_size, lat_size, lon_size
    )
    boundary = prognostic * np.float32(0.5) + np.float32(7)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "thetao_lev_2_5": (("time", "lat", "lon"), prognostic),
        "hfds": (("time", "lat", "lon"), boundary),
    }
    wet = np.ones((lat_size, lon_size), dtype=bool)
    for mask_name in spec.mask_vars:
        data_vars[mask_name] = (("lat", "lon"), wet)
    data = xr.Dataset(data_vars, coords=coords)
    means = data[["thetao_lev_2_5", "hfds"]].mean(("time", "lat", "lon"))
    stds = data[["thetao_lev_2_5", "hfds"]].std(("time", "lat", "lon"))

    data_path = tmp_path / "data.zarr"
    means_path = tmp_path / "means.zarr"
    stds_path = tmp_path / "stds.zarr"
    data.to_zarr(data_path, mode="w", consolidated=True)
    means.to_zarr(means_path, mode="w", consolidated=True)
    stds.to_zarr(stds_path, mode="w", consolidated=True)

    return DataSource.from_locations(
        LocalLocation(path=data_path),
        LocalLocation(path=means_path),
        LocalLocation(path=stds_path),
        dataset_spec=spec,
        prognostic_var_names=spec.prognostic_var_names,
        boundary_var_names=spec.boundary_var_names,
        static_data_vars=None,
        use_dask=False,
    )


@pytest.fixture
def compact_om4_source(tmp_path):
    prognostic_variables = [
        "thetao_0",
        "thetao_1",
        "thetao_2",
        "so_0",
        "so_1",
        "so_2",
        "zos",
    ]
    spec = dataclasses.replace(
        build_om4_spec(prognostic_vars_key="thetao_1", boundary_vars_key="hfds"),
        prognostic_var_names=prognostic_variables,
    )
    time_size = 20
    level_size = len(spec.depth_levels)
    lat_size, lon_size = 3, 4
    coords = {
        "time": xr.date_range(
            "2000-01-01",
            periods=time_size,
            freq="5D",
            calendar="julian",
            use_cftime=True,
        ),
        "lev": np.asarray(spec.depth_levels),
        "y": np.linspace(-60, 60, lat_size),
        "x": np.linspace(0, 270, lon_size),
    }
    depth = np.arange(
        time_size * level_size * lat_size * lon_size, dtype=np.float32
    ).reshape(time_size, level_size, lat_size, lon_size)
    thetao = depth * np.float32(0.25) - np.float32(12)
    salinity = depth * np.float32(0.05) + np.float32(30)
    surface = np.arange(time_size * lat_size * lon_size, dtype=np.float32).reshape(
        time_size, lat_size, lon_size
    )
    zos = surface * np.float32(0.02) - np.float32(1)
    boundary = surface * np.float32(-0.5) + np.float32(7)
    thetao[4, 1, 0, 2] = np.nan
    salinity[3, 2, 2, 1] = np.nan
    zos[3, 1, 2] = np.nan
    wetmask = np.ones((level_size, lat_size, lon_size), dtype=bool)
    wetmask[0, 0, 0] = False
    wetmask[1, 1, 1] = False
    wetmask[2, 2, 2] = False
    data = xr.Dataset(
        {
            "thetao": (
                ("lev", "time", "y", "x"),
                thetao.transpose(1, 0, 2, 3),
            ),
            "so": (
                ("lev", "time", "y", "x"),
                salinity.transpose(1, 0, 2, 3),
            ),
            "zos": (("time", "y", "x"), zos),
            "hfds": (("time", "y", "x"), boundary),
            "wetmask": (("lev", "y", "x"), wetmask),
        },
        coords=coords,
    )
    means = data[["thetao", "so", "zos", "hfds"]].mean(("time", "y", "x"))
    stds = data[["thetao", "so", "zos", "hfds"]].std(("time", "y", "x"))
    data_path = tmp_path / "compact-data.zarr"
    means_path = tmp_path / "compact-means.zarr"
    stds_path = tmp_path / "compact-stds.zarr"
    data.to_zarr(data_path, mode="w", consolidated=True)
    means.to_zarr(means_path, mode="w", consolidated=True)
    stds.to_zarr(stds_path, mode="w", consolidated=True)

    return DataSource.from_locations(
        LocalLocation(path=data_path),
        LocalLocation(path=means_path),
        LocalLocation(path=stds_path),
        dataset_spec=spec,
        prognostic_var_names=spec.prognostic_var_names,
        boundary_var_names=spec.boundary_var_names,
        static_data_vars=None,
        use_dask=False,
    )


@pytest.mark.parametrize("hist", [0, 1])
@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
def test_rust_batch_matches_collated_raw_train_data(
    flat_om4_source, hist, steps, stride
):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=hist,
        steps=steps,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=stride,
    )
    batch_indices = [0, min(1, len(dataset) - 1)]
    expected = collate_raw_train_data([dataset[index] for index in batch_indices])

    actual = RustBatchDataset(dataset, max_concurrent_reads=2).load_batch(batch_indices)

    assert actual.dataset_id == expected.dataset_id
    assert len(actual.raw_data) == len(expected.raw_data)
    for actual_step, expected_step in zip(actual.raw_data, expected.raw_data):
        for actual_tensor, expected_tensor in zip(actual_step, expected_step):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True
            )


@pytest.mark.parametrize("normalize_before_mask", [True, False])
def test_rust_batch_matches_processed_train_data(
    flat_om4_source, normalize_before_mask
):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=1,
        steps=2,
        normalize_before_mask=normalize_before_mask,
        masked_fill_value=-1.0,
        stride=1,
    )
    batch_indices = [0, 1]
    expected_raw = collate_raw_train_data([dataset[index] for index in batch_indices])
    actual_raw = RustBatchDataset(dataset, max_concurrent_reads=2).load_batch(
        batch_indices
    )

    expected = dataset.to_train_data(expected_raw, torch.device("cpu"))
    actual = dataset.to_train_data(actual_raw, torch.device("cpu"))

    assert len(actual) == len(expected)
    for actual_step, expected_step in zip(
        actual.example_by_step, expected.example_by_step
    ):
        for actual_tensor, expected_tensor in zip(actual_step, expected_step):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True
            )


@pytest.mark.parametrize("hist", [0, 1])
@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("normalize_before_mask", [True, False])
def test_compact_rust_batch_matches_raw_and_processed_train_data(
    compact_om4_source, hist, steps, stride, normalize_before_mask
):
    dataset = TorchTrainDataset(
        src=compact_om4_source,
        dst=None,
        prognostic_var_names=compact_om4_source.dataset_spec.prognostic_var_names,
        boundary_var_names=compact_om4_source.dataset_spec.boundary_var_names,
        hist=hist,
        steps=steps,
        normalize_before_mask=normalize_before_mask,
        masked_fill_value=-1.0,
        stride=stride,
    )
    batch_indices = [0, 1]
    expected_raw = collate_raw_train_data([dataset[index] for index in batch_indices])
    actual_raw = RustBatchDataset(dataset, max_concurrent_reads=2).load_batch(
        batch_indices
    )

    assert actual_raw.dataset_id == expected_raw.dataset_id
    for actual_step, expected_step in zip(actual_raw.raw_data, expected_raw.raw_data):
        for actual_tensor, expected_tensor in zip(actual_step, expected_step):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True
            )

    expected = dataset.to_train_data(expected_raw, torch.device("cpu"))
    actual = dataset.to_train_data(actual_raw, torch.device("cpu"))
    for actual_step, expected_step in zip(
        actual.example_by_step, expected.example_by_step
    ):
        for actual_tensor, expected_tensor in zip(actual_step, expected_step):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True
            )


def test_compact_rust_batch_uses_canonical_variable_and_level_order(
    compact_om4_source,
):
    logical_variables = compact_om4_source.dataset_spec.prognostic_var_names
    dataset = TorchTrainDataset(
        src=compact_om4_source,
        dst=None,
        prognostic_var_names=logical_variables,
        boundary_var_names=compact_om4_source.dataset_spec.boundary_var_names,
        hist=0,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )

    actual = RustBatchDataset(dataset, max_concurrent_reads=2).load_batch([0])
    expected_input = torch.from_numpy(
        np.stack(
            [
                compact_om4_source.data.thetao.isel(time=0, lev=0).to_numpy(),
                compact_om4_source.data.thetao.isel(time=0, lev=1).to_numpy(),
                compact_om4_source.data.thetao.isel(time=0, lev=2).to_numpy(),
                compact_om4_source.data.so.isel(time=0, lev=0).to_numpy(),
                compact_om4_source.data.so.isel(time=0, lev=1).to_numpy(),
                compact_om4_source.data.so.isel(time=0, lev=2).to_numpy(),
                compact_om4_source.data.zos.isel(time=0).to_numpy(),
            ]
        )
    ).reshape(1, 1, len(logical_variables), 3, 4)

    torch.testing.assert_close(
        actual.raw_data[0][0], expected_input, rtol=0, atol=0, equal_nan=True
    )


def test_compact_rust_loader_consumes_existing_prefetch_schedule(
    compact_om4_source,
):
    dataset = TorchTrainDataset(
        src=compact_om4_source,
        dst=None,
        prognostic_var_names=compact_om4_source.dataset_spec.prognostic_var_names,
        boundary_var_names=compact_om4_source.dataset_spec.boundary_var_names,
        hist=1,
        steps=2,
        normalize_before_mask=True,
        masked_fill_value=-1.0,
        stride=2,
    )
    schedule = [[3, 1], [0, 2]]
    loader = RustTrainDataLoader(
        [dataset],
        schedule,
        torch.device("cpu"),
        max_concurrent_reads=2,
        prefetch_batches=2,
        pin_memory=False,
    )

    for actual, batch_indices in zip(loader, schedule):
        expected_raw = collate_raw_train_data(
            [dataset[index] for index in batch_indices]
        )
        expected = dataset.to_train_data(expected_raw, torch.device("cpu"))
        for actual_step, expected_step in zip(
            actual.example_by_step, expected.example_by_step
        ):
            for actual_tensor, expected_tensor in zip(actual_step, expected_step):
                torch.testing.assert_close(
                    actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True
                )


def test_train_data_device_preparation_caches_static_tensors(flat_om4_source):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=0,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )
    raw = RustBatchDataset(dataset, max_concurrent_reads=1).load_batch([0, 1])
    device = torch.device("cpu")

    first = dataset.to_train_data(raw, device)
    cached_static = dict(dataset._device_ocean_static)
    cached_ctx = dataset._device_ctx[device]
    second = dataset.to_train_data(raw, device)

    assert first.ctx is cached_ctx
    assert second.ctx is cached_ctx
    assert dataset._device_ocean_static.keys() == cached_static.keys()
    for key, expected in cached_static.items():
        assert all(
            actual is expected_tensor
            for actual, expected_tensor in zip(
                dataset._device_ocean_static[key], expected
            )
        )


def test_rust_loader_consumes_existing_batch_schedule(flat_om4_source):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=1,
        steps=2,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=1,
    )
    schedule = [[3, 1], [0, 2]]
    loader = RustTrainDataLoader(
        [dataset],
        schedule,
        torch.device("cpu"),
        max_concurrent_reads=2,
        prefetch_batches=2,
        pin_memory=False,
    )

    actual_batches = list(loader)

    assert len(actual_batches) == len(schedule)
    assert len(loader) == len(schedule)
    assert not hasattr(loader, "_dataloader")
    for actual, batch_indices in zip(actual_batches, schedule):
        expected_raw = collate_raw_train_data(
            [dataset[index] for index in batch_indices]
        )
        expected = dataset.to_train_data(expected_raw, torch.device("cpu"))
        for actual_step, expected_step in zip(
            actual.example_by_step, expected.example_by_step
        ):
            for actual_tensor, expected_tensor in zip(actual_step, expected_step):
                torch.testing.assert_close(
                    actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True
                )


def test_rust_loader_preserves_homogeneous_dataset_id_invariant(flat_om4_source):
    first = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=0,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )
    second = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=0,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )
    loader = RustTrainDataLoader(
        [first, second],
        [[0, len(first)]],
        torch.device("cpu"),
        max_concurrent_reads=1,
        prefetch_batches=1,
        pin_memory=False,
    )

    with pytest.raises(AssertionError, match="heterogenous batches"):
        next(iter(loader))


def test_distributed_sampler_never_crosses_dataset_ids(flat_om4_source):
    datasets = [
        TorchTrainDataset(
            src=flat_om4_source,
            dst=None,
            prognostic_var_names=["thetao_0"],
            boundary_var_names=["hfds"],
            hist=0,
            steps=1,
            normalize_before_mask=True,
            masked_fill_value=0.0,
            stride=stride,
        )
        for stride in (1, 2)
    ]

    schedules = []
    group_keys = _dataset_batch_group_keys(datasets)
    for rank in range(2):
        sampler = DistributedEquivalenceGroupBatchSampler(
            datasets=datasets,
            group_key=lambda dataset: group_keys[dataset.id],
            batch_size=4,
            num_replicas=2,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        loader = RustTrainDataLoader(
            datasets,
            sampler,
            torch.device("cpu"),
            max_concurrent_reads=1,
            prefetch_batches=1,
            pin_memory=False,
        )
        schedule = list(sampler)
        schedules.append(schedule)
        for batch_indices in schedule:
            loader._resolve_batch(batch_indices)

    assert len(schedules[0]) == len(schedules[1])


def test_rust_batch_uses_physical_indices_after_time_slice(flat_om4_source):
    sliced = flat_om4_source.slice(
        TimeConfig.model_validate({"start": "2000-01-11", "end": "2000-03-01"})
    )
    selected_positions = flat_om4_source.data.indexes["time"].get_indexer(
        sliced.data.indexes["time"]
    )
    np.testing.assert_array_equal(sliced.physical_time_indices, selected_positions)

    dataset = TorchTrainDataset(
        src=sliced,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=1,
        steps=2,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=1,
    )
    batch_indices = [0, 1]
    expected = collate_raw_train_data([dataset[index] for index in batch_indices])
    actual = RustBatchDataset(dataset, max_concurrent_reads=2).load_batch(batch_indices)

    for actual_step, expected_step in zip(actual.raw_data, expected.raw_data):
        for actual_tensor, expected_tensor in zip(actual_step, expected_step):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True
            )


def test_trainer_selects_rust_loader_with_no_pytorch_workers(flat_om4_source, tmp_path):
    data_root = flat_om4_source.data_location.path.parent
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/test/train_default.yaml"
    )
    config = TrainConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(data_root),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
            "--backend",
            "cpu",
        ]
    )
    config.data.sources = [
        DataSourceConfig(
            data_location=UnresolvedLocation(path="data.zarr"),
            data_means_location=UnresolvedLocation(path="means.zarr"),
            data_stds_location=UnresolvedLocation(path="stds.zarr"),
        )
    ]
    config.data.dataset = Om4DatasetConfig(
        prognostic_vars_key="thetao_1", boundary_vars_key="hfds"
    )
    config.data.loading = RustDataLoadingConfig(
        prefetch_batches=2, max_concurrent_reads=2
    )
    config.train_time = TimeConfig.model_validate(
        {"start": "2000-01-01", "end": "2000-02-10"}
    )
    config.val_time = TimeConfig.model_validate(
        {"start": "2000-02-15", "end": "2000-03-20"}
    )
    config.inference_epochs = []
    config.batch_size = 2
    config.data_stride = [1, 2]
    config.steps = [1]
    config.step_transition = []

    with MultitonScope():
        trainer = Trainer(config)
        trainer.init_data_loaders(cur_step=1)
        batch = next(iter(trainer.train_loader))

    assert trainer.num_workers == 0
    assert isinstance(trainer.train_loader, RustTrainDataLoader)
    assert isinstance(trainer.val_loader, RustTrainDataLoader)
    assert not hasattr(trainer.train_loader, "_dataloader")
    assert trainer.train_loader._read_pool is trainer.val_loader._read_pool
    assert all(
        dataset.read_pool is trainer.train_loader._read_pool
        for dataset in trainer.train_loader._batch_datasets
    )
    assert len(trainer.train_loader._batch_datasets) == 2
    for batch_indices in trainer.train_sampler:
        trainer.train_loader._resolve_batch(batch_indices)
    for batch_indices in trainer.val_sampler:
        trainer.val_loader._resolve_batch(batch_indices)
    assert len(batch) == 1


def test_trainer_selects_rust_loader_for_compact_om4(compact_om4_source, tmp_path):
    data_root = compact_om4_source.data_location.path.parent
    config_path = (
        Path(__file__).resolve().parents[1] / "configs/test/train_default.yaml"
    )
    config = TrainConfig.from_yaml_and_cli(
        [
            str(config_path),
            "--experiment.data_root",
            str(data_root),
            "--experiment.base_output_dir",
            str(tmp_path / "outputs"),
            "--backend",
            "cpu",
        ]
    )
    config.data.sources = [
        DataSourceConfig(
            data_location=UnresolvedLocation(path="compact-data.zarr"),
            data_means_location=UnresolvedLocation(path="compact-means.zarr"),
            data_stds_location=UnresolvedLocation(path="compact-stds.zarr"),
        )
    ]
    config.data.dataset = Om4DatasetConfig(
        prognostic_vars_key="thetao_1", boundary_vars_key="hfds"
    )
    config.data.loading = RustDataLoadingConfig(
        prefetch_batches=2, max_concurrent_reads=2
    )
    config.train_time = TimeConfig.model_validate(
        {"start": "2000-01-01", "end": "2000-02-10"}
    )
    config.val_time = TimeConfig.model_validate(
        {"start": "2000-02-15", "end": "2000-03-20"}
    )
    config.inference_epochs = []
    config.batch_size = 2
    config.data_stride = [1, 2]
    config.steps = [1]
    config.step_transition = []

    with MultitonScope():
        trainer = Trainer(config)
        trainer.init_data_loaders(cur_step=1)
        batch = next(iter(trainer.train_loader))

    assert isinstance(trainer.train_loader, RustTrainDataLoader)
    assert isinstance(trainer.val_loader, RustTrainDataLoader)
    assert all(
        dataset._input_store._physical_by_logical is None
        for dataset in trainer.train_loader._batch_datasets
    )
    assert len(batch) == 1


def test_rust_loader_prefetches_next_batch_during_consumption(
    flat_om4_source, monkeypatch
):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=0,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )
    loader = RustTrainDataLoader(
        [dataset],
        [[0], [1], [2]],
        torch.device("cpu"),
        max_concurrent_reads=1,
        prefetch_batches=2,
        pin_memory=False,
    )
    original_load = loader._batch_datasets[0].load_batch
    second_started = threading.Event()
    call_count = 0

    def slow_load(indices, *, pin_memory=False):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            second_started.set()
        time.sleep(0.03)
        return original_load(indices, pin_memory=pin_memory)

    monkeypatch.setattr(loader._batch_datasets[0], "load_batch", slow_load)
    iterator = cast(Any, iter(loader))

    next(iterator)

    assert second_started.wait(timeout=0.2)
    assert len(iterator._host._pending) <= 2
    iterator.close()
    assert not any(
        thread.name.startswith("samudra-rust-prefetch")
        for thread in threading.enumerate()
    )


def test_rust_loader_surfaces_prefetch_errors(flat_om4_source, monkeypatch):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=0,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )
    loader = RustTrainDataLoader(
        [dataset],
        [[0]],
        torch.device("cpu"),
        max_concurrent_reads=1,
        prefetch_batches=1,
        pin_memory=False,
    )

    def fail_load(_indices, **_kwargs):
        raise ValueError("intentional Rust producer failure")

    monkeypatch.setattr(loader._batch_datasets[0], "load_batch", fail_load)

    with pytest.raises(ValueError, match="intentional Rust producer failure"):
        next(iter(loader))

    assert not any(
        thread.name.startswith("samudra-rust-prefetch")
        for thread in threading.enumerate()
    )


@pytest.mark.cuda
def test_rust_loader_prefetches_pinned_batch_on_dedicated_cuda_stream(
    flat_om4_source, monkeypatch
):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=1,
        steps=2,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )
    loader = RustTrainDataLoader(
        [dataset],
        [[0, 1], [2, 3]],
        torch.device("cuda"),
        max_concurrent_reads=2,
        prefetch_batches=2,
        pin_memory=True,
        prefetch_to_device=True,
    )
    observed_streams = []
    observed_pinned = []
    original_prepare = loader._prepare_batch

    def record_prepare(loaded):
        _, raw = loaded
        observed_pinned.append(
            all(tensor.is_pinned() for step in raw.raw_data for tensor in step)
        )
        observed_streams.append(torch.cuda.current_stream())
        return original_prepare(loaded)

    monkeypatch.setattr(loader, "_prepare_batch", record_prepare)
    iterator = cast(Any, iter(loader))
    actual = next(iterator)
    expected_raw = collate_raw_train_data([dataset[0], dataset[1]])
    expected = dataset.to_train_data(expected_raw, torch.device("cpu"))

    # __next__ enqueues batch N+1 before returning batch N to the model stream.
    assert observed_pinned == [True, True]
    assert all(stream != torch.cuda.default_stream() for stream in observed_streams)
    assert actual.ctx.label_mask.device.type == "cuda"
    for actual_step, expected_step in zip(
        actual.example_by_step, expected.example_by_step
    ):
        for actual_tensor, expected_tensor in zip(actual_step, expected_step):
            torch.testing.assert_close(
                actual_tensor.cpu(),
                expected_tensor,
                rtol=0,
                atol=0,
                equal_nan=True,
            )
    iterator.close()


@pytest.mark.cuda
def test_rust_loader_reuses_pinned_buffers_after_cuda_event(flat_om4_source):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=0,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )
    loader = RustTrainDataLoader(
        [dataset],
        [[index] for index in range(6)],
        torch.device("cuda"),
        max_concurrent_reads=1,
        prefetch_batches=1,
        pin_memory=True,
        prefetch_to_device=True,
    )

    list(loader)
    torch.cuda.synchronize()
    stats = loader.pinned_pool_stats

    assert stats is not None
    assert stats["reuse_count"] > 0
    assert stats["in_use_bytes"] == 0
    # Three tensors per batch, with at most the configured one-batch queue plus
    # the batch currently being prepared alive at once.
    assert stats["allocation_count"] <= 6
