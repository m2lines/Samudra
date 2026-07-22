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
    InferenceDataLoadingConfig,
    Om4DatasetConfig,
    RustDataLoadingConfig,
    TimeConfig,
    TrainConfig,
)
from samudra.constants import build_om4_spec
from samudra.datasets import TorchTrainDataset
from samudra.rust_data import (
    CudaPrefetch,
    HostPrefetch,
    RustTrainDataLoader,
    create_rust_io_runtime,
    native_om4_dataset,
)
from samudra.train import Trainer
from samudra.utils.data import CanonicalDataset
from samudra.utils.location import LocalLocation, UnresolvedLocation
from samudra.utils.multiton import MultitonScope
from samudra.utils.samplers import DistributedEquivalenceGroupBatchSampler
from samudra.utils.train import collate_raw_train_data


def rust_train_loader(
    *args: Any, max_concurrent_reads: int, **kwargs: Any
) -> RustTrainDataLoader:
    del max_concurrent_reads
    datasets, *rest = args
    pin_memory = kwargs.pop("pin_memory")
    prefetch = (
        CudaPrefetch()
        if kwargs.pop("prefetch_to_device", False)
        else HostPrefetch(pin_memory=pin_memory)
    )
    return RustTrainDataLoader(
        [dataset.shard for dataset in datasets], *rest, prefetch=prefetch, **kwargs
    )


def make_flat_om4_source(
    tmp_path: Path,
    *,
    prefix: str,
    lat_size: int,
    lon_size: int,
    scale: float = 1.0,
    offset: float = 0.0,
) -> CanonicalDataset:
    spec = build_om4_spec(prognostic_vars_key="thetao_1", boundary_vars_key="hfds")
    time_size = 20
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
    ) * np.float32(scale) + np.float32(offset)
    boundary = prognostic * np.float32(0.5) + np.float32(7)
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "thetao_0": (("time", "lat", "lon"), prognostic),
        "hfds": (("time", "lat", "lon"), boundary),
    }
    wet = np.ones((lat_size, lon_size), dtype=bool)
    for mask_name in spec.mask_vars:
        data_vars[mask_name] = (("lat", "lon"), wet)
    data = xr.Dataset(data_vars, coords=coords)
    means = data[["thetao_0", "hfds"]].mean(("time", "lat", "lon"))
    stds = data[["thetao_0", "hfds"]].std(("time", "lat", "lon"))

    data_path = tmp_path / f"{prefix}data.zarr"
    means_path = tmp_path / f"{prefix}means.zarr"
    stds_path = tmp_path / f"{prefix}stds.zarr"
    data.to_zarr(data_path, mode="w", consolidated=True)
    means.to_zarr(means_path, mode="w", consolidated=True)
    stds.to_zarr(stds_path, mode="w", consolidated=True)

    source = CanonicalDataset.from_locations(
        data_location=LocalLocation(path=data_path),
        means_location=LocalLocation(path=means_path),
        stds_location=LocalLocation(path=stds_path),
        dataset_spec=spec,
        prognostic_var_names=spec.prognostic_var_names,
        boundary_var_names=spec.boundary_var_names,
        static_data_vars=None,
        use_dask=False,
    )
    return native_om4_dataset(
        source, LocalLocation(path=data_path), create_rust_io_runtime(2)
    )


@pytest.fixture
def flat_om4_source(tmp_path):
    return make_flat_om4_source(
        tmp_path,
        prefix="",
        lat_size=3,
        lon_size=4,
    )


@pytest.fixture
def flat_om4_destination(tmp_path):
    return make_flat_om4_source(
        tmp_path,
        prefix="destination-",
        lat_size=2,
        lon_size=3,
        scale=-0.25,
        offset=100.0,
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

    source = CanonicalDataset.from_locations(
        data_location=LocalLocation(path=data_path),
        means_location=LocalLocation(path=means_path),
        stds_location=LocalLocation(path=stds_path),
        dataset_spec=spec,
        prognostic_var_names=spec.prognostic_var_names,
        boundary_var_names=spec.boundary_var_names,
        static_data_vars=None,
        use_dask=False,
    )
    return native_om4_dataset(
        source, LocalLocation(path=data_path), create_rust_io_runtime(2)
    )


@pytest.mark.parametrize("hist", [0, 1])
@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("normalize_before_mask", [True, False])
def test_compact_rust_loader_consumes_existing_prefetch_schedule(
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
    schedule = [[3, 1], [0, 2]]
    loader = rust_train_loader(
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
    raw = collate_raw_train_data([dataset[0], dataset[1]])
    device = torch.device("cpu")

    first = dataset.to_train_data(raw, device)
    cached_static = dict(dataset.preparer._device_ocean_static)
    cached_ctx = dataset.preparer._device_ctx[device]
    second = dataset.to_train_data(raw, device)

    assert first.ctx is cached_ctx
    assert second.ctx is cached_ctx
    assert dataset.preparer._device_ocean_static.keys() == cached_static.keys()
    for key, expected in cached_static.items():
        assert all(
            actual is expected_tensor
            for actual, expected_tensor in zip(
                dataset.preparer._device_ocean_static[key], expected
            )
        )


def test_training_shard_exposes_shaped_full_rollout_plan(flat_om4_source):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=1,
        steps=2,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=2,
    )

    plan = dataset.shard.window_plan([2, 0])

    assert plan.dataset_id == dataset.id
    assert len(plan.steps) == 2
    np.testing.assert_array_equal(
        plan.steps[0].input.request.time_indices, [[2, 4], [0, 2]]
    )
    np.testing.assert_array_equal(
        plan.steps[0].label.request.time_indices, [[6, 8], [4, 6]]
    )
    np.testing.assert_array_equal(
        plan.steps[1].input.request.time_indices, [[6, 8], [4, 6]]
    )
    assert plan.steps[0].boundary.source is dataset.shard.boundary_src
    assert dataset.batch_compatibility_key == dataset.id
    assert not hasattr(dataset, "_device_ocean_static")


def test_training_shard_can_read_destination_at_current_input_times(flat_om4_source):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=flat_om4_source,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=1,
        steps=2,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=2,
        target_time_mode="current",
    )

    plan = dataset.shard.window_plan([2, 0])

    np.testing.assert_array_equal(
        plan.steps[0].input.request.time_indices, [[2, 4], [0, 2]]
    )
    np.testing.assert_array_equal(
        plan.steps[0].label.request.time_indices, [[2, 4], [0, 2]]
    )
    np.testing.assert_array_equal(
        plan.steps[1].input.request.time_indices, [[6, 8], [4, 6]]
    )
    np.testing.assert_array_equal(
        plan.steps[1].label.request.time_indices, [[6, 8], [4, 6]]
    )

    last_plan = dataset.shard.window_plan([len(dataset) - 1])
    np.testing.assert_array_equal(
        last_plan.steps[-1].input.request.time_indices,
        [[flat_om4_source.time.size - 3, flat_om4_source.time.size - 1]],
    )
    np.testing.assert_array_equal(
        last_plan.steps[-1].label.request.time_indices,
        last_plan.steps[-1].input.request.time_indices,
    )


@pytest.mark.parametrize("hist", [0, 1])
@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("normalize_before_mask", [True, False])
def test_rust_loader_consumes_existing_batch_schedule(
    flat_om4_source, hist, steps, stride, normalize_before_mask
):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=hist,
        steps=steps,
        normalize_before_mask=normalize_before_mask,
        masked_fill_value=0.0,
        stride=stride,
    )
    schedule = [[3, 1], [0, 2]]
    loader = rust_train_loader(
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


def test_rust_loader_matches_cpu_with_separate_destination(
    flat_om4_source, flat_om4_destination
):
    dataset = TorchTrainDataset(
        src=flat_om4_source,
        dst=flat_om4_destination,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=1,
        steps=2,
        normalize_before_mask=False,
        masked_fill_value=-1.0,
        stride=2,
    )
    batch_indices = [0, 2]
    loader = rust_train_loader(
        [dataset],
        [batch_indices],
        torch.device("cpu"),
        max_concurrent_reads=2,
        prefetch_batches=1,
        pin_memory=False,
    )

    actual = next(iter(loader))
    expected_raw = collate_raw_train_data([dataset[index] for index in batch_indices])
    expected = dataset.to_train_data(expected_raw, torch.device("cpu"))

    for actual_step, expected_step in zip(
        actual.example_by_step, expected.example_by_step
    ):
        for actual_tensor, expected_tensor in zip(actual_step, expected_step):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True
            )


def test_rust_loader_deduplicates_full_rollout_before_preprocessing(
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
        stride=1,
    )
    loader = rust_train_loader(
        [dataset],
        [[0, 1]],
        torch.device("cpu"),
        max_concurrent_reads=2,
        prefetch_batches=1,
        pin_memory=False,
    )
    chunk_batch = loader._batch_datasets[0].load_chunk_batch([0, 1])

    # The materialized layout would contain 24 planes: input, boundary, and
    # label each have two samples by two times for both rollout steps. The
    # unique plan retains only seven prognostic and five boundary time planes.
    assert sorted(group.shape[0] for group in chunk_batch.groups) == [5, 7]
    assert sum(group.shape[0] for group in chunk_batch.groups) == 12

    calls = 0
    preparer = loader._batch_datasets[0].preparer
    original = preparer.normalize_and_mask_device_planes

    def record_preprocessing(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        preparer, "normalize_and_mask_device_planes", record_preprocessing
    )
    actual = next(iter(loader))
    expected_raw = collate_raw_train_data([dataset[0], dataset[1]])
    expected = dataset.to_train_data(expected_raw, torch.device("cpu"))

    # Input and label share one unique prognostic transform; boundary uses the
    # other. Neither transform is repeated for each rollout step.
    assert calls == 2
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
    loader = rust_train_loader(
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
    for rank in range(2):
        sampler = DistributedEquivalenceGroupBatchSampler(
            datasets=datasets,
            batch_size=4,
            num_replicas=2,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        loader = rust_train_loader(
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
    sliced = flat_om4_source.slice_time(
        TimeConfig.model_validate({"start": "2000-01-11", "end": "2000-03-01"})
    )

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
    expected_raw = collate_raw_train_data([dataset[index] for index in batch_indices])
    expected = dataset.to_train_data(expected_raw, torch.device("cpu"))
    loader = rust_train_loader(
        [dataset],
        [batch_indices],
        torch.device("cpu"),
        max_concurrent_reads=2,
        prefetch_batches=1,
        pin_memory=False,
    )
    actual = next(iter(loader))

    for actual_step, expected_step in zip(
        actual.example_by_step, expected.example_by_step
    ):
        for actual_tensor, expected_tensor in zip(actual_step, expected_step):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True
            )


def test_native_decoration_maps_an_already_sliced_canonical_dataset(
    flat_om4_source,
):
    canonical = dataclasses.replace(
        flat_om4_source,
        _reader=flat_om4_source._reader.semantic,
    )
    sliced = canonical.slice_time(
        TimeConfig.model_validate({"start": "2000-01-11", "end": "2000-03-01"})
    )
    native = native_om4_dataset(
        sliced,
        LocalLocation(path=Path(flat_om4_source._reader.path)),
        create_rust_io_runtime(2),
    )
    dataset = TorchTrainDataset(
        src=native,
        dst=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        hist=1,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )
    indices = [0, 1]
    expected_raw = collate_raw_train_data([dataset[index] for index in indices])
    expected = dataset.to_train_data(expected_raw, torch.device("cpu"))
    loader = rust_train_loader(
        [dataset],
        [indices],
        torch.device("cpu"),
        max_concurrent_reads=2,
        prefetch_batches=1,
        pin_memory=False,
    )

    actual = next(iter(loader))

    for actual_step, expected_step in zip(
        actual.example_by_step, expected.example_by_step, strict=True
    ):
        for actual_tensor, expected_tensor in zip(
            actual_step, expected_step, strict=True
        ):
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True
            )


def test_trainer_selects_rust_loader_with_no_pytorch_workers(flat_om4_source, tmp_path):
    data_root = Path(flat_om4_source._reader.path).parent
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
    config.data.inference_loading = InferenceDataLoadingConfig(num_workers=3)
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
    assert trainer.inference_num_workers == 3
    assert isinstance(trainer.train_loader, RustTrainDataLoader)
    assert isinstance(trainer.val_loader, RustTrainDataLoader)
    assert not hasattr(trainer.train_loader, "_dataloader")
    assert len(trainer.train_loader._batch_datasets) == 2
    for batch_indices in trainer.train_sampler:
        trainer.train_loader._resolve_batch(batch_indices)
    for batch_indices in trainer.val_sampler:
        trainer.val_loader._resolve_batch(batch_indices)
    assert len(batch) == 1


def test_trainer_selects_rust_loader_for_compact_om4(compact_om4_source, tmp_path):
    data_root = Path(compact_om4_source._reader.path).parent
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
        dataset._input_reader._reader_variables["thetao_0"].extension_selector()
        == ("thetao", 0)
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
    loader = rust_train_loader(
        [dataset],
        [[0], [1], [2]],
        torch.device("cpu"),
        max_concurrent_reads=1,
        prefetch_batches=2,
        pin_memory=False,
    )
    original_load = loader._batch_datasets[0].load_chunk_batch
    second_started = threading.Event()
    call_count = 0

    def slow_load(indices, *, buffer_pool=None):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            second_started.set()
        time.sleep(0.03)
        return original_load(indices, buffer_pool=buffer_pool)

    monkeypatch.setattr(loader._batch_datasets[0], "load_chunk_batch", slow_load)
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
    loader = rust_train_loader(
        [dataset],
        [[0]],
        torch.device("cpu"),
        max_concurrent_reads=1,
        prefetch_batches=1,
        pin_memory=False,
    )

    def fail_load(_indices, **_kwargs):
        raise ValueError("intentional Rust producer failure")

    monkeypatch.setattr(loader._batch_datasets[0], "load_chunk_batch", fail_load)

    with pytest.raises(ValueError, match="intentional Rust producer failure"):
        next(iter(loader))

    assert not any(
        thread.name.startswith("samudra-rust-prefetch")
        for thread in threading.enumerate()
    )


def test_rust_loader_closes_prefetch_when_partial_iterator_is_abandoned(
    flat_om4_source,
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
    loader = rust_train_loader(
        [dataset],
        [[0], [1], [2]],
        torch.device("cpu"),
        max_concurrent_reads=1,
        prefetch_batches=2,
        pin_memory=False,
    )

    next(iter(loader))

    assert loader._active_iterator is not None
    assert loader._active_iterator() is None
    assert not any(
        thread.name.startswith("samudra-rust-prefetch")
        for thread in threading.enumerate()
    )


@pytest.mark.cuda
def test_rust_loader_reclaims_completed_pinned_prefetch_on_early_close(
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
    loader = rust_train_loader(
        [dataset],
        [[0], [1], [2]],
        torch.device("cpu"),
        max_concurrent_reads=1,
        prefetch_batches=2,
        pin_memory=True,
    )
    acquired: list[int] = []
    released: list[int] = []
    assert loader._pinned_pool is not None
    original_acquire = loader._pinned_pool.acquire
    original_release = loader._pinned_pool.release_tensors

    def tracked_acquire(shape):
        tensor = original_acquire(shape)
        acquired.append(id(tensor))
        return tensor

    def tracked_release(tensors, event=None):
        released.extend(id(tensor) for tensor in tensors)
        return original_release(tensors, event)

    monkeypatch.setattr(loader._pinned_pool, "acquire", tracked_acquire)
    monkeypatch.setattr(loader._pinned_pool, "release_tensors", tracked_release)

    for _ in range(3):
        iterator = cast(Any, iter(loader))
        for future in iterator._host._pending:
            future.result()
        iterator.close()

    assert sorted(released) == sorted(acquired)
    assert len(set(acquired)) < len(acquired)


@pytest.mark.cuda
def test_rust_loader_reclaims_pinned_prefetch_after_producer_error(
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
    loader = rust_train_loader(
        [dataset],
        [[0], [1]],
        torch.device("cpu"),
        max_concurrent_reads=1,
        prefetch_batches=2,
        pin_memory=True,
    )
    acquired: list[int] = []
    released: list[int] = []
    assert loader._pinned_pool is not None
    original_acquire = loader._pinned_pool.acquire
    original_release = loader._pinned_pool.release_tensors

    def tracked_acquire(shape):
        tensor = original_acquire(shape)
        acquired.append(id(tensor))
        return tensor

    def tracked_release(tensors, event=None):
        released.extend(id(tensor) for tensor in tensors)
        return original_release(tensors, event)

    monkeypatch.setattr(loader._pinned_pool, "acquire", tracked_acquire)
    monkeypatch.setattr(loader._pinned_pool, "release_tensors", tracked_release)
    original_load = loader._batch_datasets[0].load_chunk_batch
    call_count = 0

    def fail_then_load(indices, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("intentional Rust producer failure")
        return original_load(indices, **kwargs)

    monkeypatch.setattr(loader._batch_datasets[0], "load_chunk_batch", fail_then_load)

    with pytest.raises(ValueError, match="intentional Rust producer failure"):
        next(iter(loader))

    assert sorted(released) == sorted(acquired)


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
    schedule = [[0, 1], [2, 3], [4, 5]]
    loader = rust_train_loader(
        [dataset],
        schedule,
        torch.device("cuda"),
        max_concurrent_reads=2,
        prefetch_batches=2,
        pin_memory=True,
        prefetch_to_device=True,
    )
    observed_streams = []
    observed_pinned = []
    pointers: list[int] = []
    assert loader._pinned_pool is not None
    original_acquire = loader._pinned_pool.acquire
    original_prepare = loader._prepare_batch

    def tracked_acquire(shape):
        tensor = original_acquire(shape)
        pointers.append(tensor.data_ptr())
        return tensor

    def record_prepare(loaded):
        _, raw = loaded
        observed_pinned.append(all(tensor.is_pinned() for tensor in raw.groups))
        observed_streams.append(torch.cuda.current_stream())
        torch.cuda._sleep(20_000_000)
        return original_prepare(loaded)

    monkeypatch.setattr(loader._pinned_pool, "acquire", tracked_acquire)
    monkeypatch.setattr(loader, "_prepare_batch", record_prepare)
    iterator = cast(Any, iter(loader))
    actual_batches = list(iterator)

    # __next__ enqueues batch N+1 before returning batch N to the model stream.
    assert observed_pinned == [True, True, True]
    assert all(stream != torch.cuda.default_stream() for stream in observed_streams)
    # The artificial stream delay keeps batch N's H2D copies active while the host
    # producer acquires batch N+1, so their two read-group buffers must be distinct.
    assert set(pointers[:2]).isdisjoint(pointers[2:4])
    for actual, batch_indices in zip(actual_batches, schedule):
        expected_raw = collate_raw_train_data(
            [dataset[index] for index in batch_indices]
        )
        expected = dataset.to_train_data(expected_raw, torch.device("cpu"))
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
def test_rust_loader_reuses_pinned_buffers_after_cuda_event(
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
    loader = rust_train_loader(
        [dataset],
        [[0, 1, 2, 3], [4], [5, 6], [7], [8, 9, 10], [11]],
        torch.device("cuda"),
        max_concurrent_reads=1,
        prefetch_batches=1,
        pin_memory=True,
        prefetch_to_device=True,
    )
    pointers: list[int] = []
    assert loader._pinned_pool is not None
    original_acquire = loader._pinned_pool.acquire

    def tracked_acquire(shape):
        tensor = original_acquire(shape)
        pointers.append(tensor.data_ptr())
        return tensor

    monkeypatch.setattr(loader._pinned_pool, "acquire", tracked_acquire)

    list(loader)
    torch.cuda.synchronize()
    probe = loader._pinned_pool.acquire((1,))
    loader._pinned_pool.release_tensors([probe])

    assert len(loader._pinned_pool._free) <= 3
    assert len(set(pointers)) < len(pointers)
