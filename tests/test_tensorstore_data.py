# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr
import zarr  # type: ignore[import-untyped]

pytest.importorskip("tensorstore")

from samudra.config import DataConfig, Om4DataSourceConfig, TensorStoreDataLoadingConfig
from samudra.datasets import TorchTrainDataset, TrainingWindows
from samudra.native_reader import NativeOm4Reader, PhysicalVariable
from samudra.tensorstore_data import TensorStoreIoRuntime
from samudra.utils.location import LocalLocation
from samudra.utils.train import collate_host_batches


@pytest.fixture(params=["flat", "time-lev", "lev-time"])
def om4_store(tmp_path, request):
    values = np.arange(24 * 19 * 2 * 3, dtype=np.float32).reshape(24, 19, 2, 3)
    values[4, 0, 1, 2] = np.nan
    mask = np.ones((19, 2, 3), dtype=bool)
    mask[:, 0, 0] = False
    data = xr.Dataset(
        {
            "thetao": (("time", "lev", "y", "x"), values),
            "hfds": (("time", "y", "x"), values[:, 0] * np.float32(0.25)),
            "wetmask": (("lev", "y", "x"), mask),
        },
        coords={
            "time": xr.date_range(
                "2000-01-01", periods=24, freq="5D", calendar="julian", use_cftime=True
            ),
            "lev": np.arange(19),
            "y": [-30.0, 30.0],
            "x": [0.0, 120.0, 240.0],
        },
    )
    if request.param == "flat":
        for level in range(19):
            data[f"thetao_{level}"] = data.thetao.isel(lev=level, drop=True)
            data[f"mask_{level}"] = data.wetmask.isel(lev=level, drop=True)
        data = data.drop_vars(["thetao", "wetmask"]).drop_dims("lev")
    elif request.param == "lev-time":
        data = data.transpose("lev", "time", "y", "x", missing_dims="ignore")
    variables = [
        name
        for name in data.data_vars
        if str(name).startswith("thetao") or name == "hfds"
    ]
    means = data[variables].mean(("time", "y", "x"))
    stds = data[variables].std(("time", "y", "x"))
    # Compressed, time-chunked fixtures exercise real decoding and overlapping reads.
    data.chunk({"time": 2}).to_zarr(tmp_path / "data.zarr", mode="w")
    means.to_zarr(tmp_path / "means.zarr", mode="w")
    stds.to_zarr(tmp_path / "stds.zarr", mode="w")
    cfg = Om4DataSourceConfig.model_validate(
        {
            "train_time": {"start": "2000-01-11", "end": "2000-04-10"},
            "val_time": {"start": "2000-04-11", "end": "2000-04-25"},
            "data_location": "data.zarr",
            "data_means_location": "means.zarr",
            "data_stds_location": "stds.zarr",
            "prognostic_vars_key": "thetao_1",
            "boundary_vars_key": "hfds",
        }
    )
    return tmp_path, cfg


def make_loader(
    om4_store,
    monkeypatch,
    device,
    input_steps,
    output_steps,
    steps,
    normalize_before_mask,
):
    def no_rust():
        pytest.fail("TensorStore loading attempted to import the Rust extension")

    monkeypatch.setattr("samudra.rust_reader._load_extension", no_rust)
    root, source_cfg = om4_store
    loading = TensorStoreDataLoadingConfig(max_concurrent_reads=2, prefetch_batches=2)
    container = DataConfig(sources=[source_cfg], loading=loading).build(
        LocalLocation(path=root)
    )
    source = container.train_sources[0]
    windows = TrainingWindows(
        source,
        None,
        source.data_layout.prognostic_var_names,
        source.data_layout.boundary_var_names,
        input_steps=input_steps,
        output_steps=output_steps,
        steps=steps,
        stride=1,
        normalize_before_mask=normalize_before_mask,
        masked_fill_value=-7.0,
    )
    schedule = [[3, 1], [0, 2], [1, 3]]
    loader = loading.build_batch_loader([windows], schedule, device, seed=0)
    # Use genuine Xarray reads as the reference, rather than the native reader.
    reference = copy.copy(windows)
    assert isinstance(source.reader, NativeOm4Reader)
    reference.input_source = dataclasses.replace(
        source, _reader=source.reader.xarray_reader
    )
    reference.label_source = reference.input_source
    dataset = TorchTrainDataset(reference)
    return dataset, loader, schedule


@pytest.mark.parametrize(
    "input_steps,output_steps,steps", [(1, 1, 1), (2, 1, 2), (2, 1, 4), (2, 2, 4)]
)
@pytest.mark.parametrize("normalize_before_mask", [True, False])
def test_tensorstore_pipeline_matches_xarray(
    om4_store, monkeypatch, input_steps, output_steps, steps, normalize_before_mask
):
    device = torch.device("cpu")
    dataset, loader, schedule = make_loader(
        om4_store,
        monkeypatch,
        device,
        input_steps,
        output_steps,
        steps,
        normalize_before_mask,
    )
    try:
        for batch, indices in zip(loader, schedule, strict=True):
            expected = dataset.to_model_batch(
                collate_host_batches([dataset[i] for i in indices]), device
            )
            for actual_step, expected_step in zip(
                batch.steps, expected.steps, strict=True
            ):
                for actual, reference in zip(actual_step, expected_step, strict=True):
                    torch.testing.assert_close(
                        actual, reference, rtol=0, atol=0, equal_nan=True
                    )
    finally:
        loader.close()


@pytest.mark.cuda
def test_tensorstore_cuda_prefetch_matches_xarray(om4_store, monkeypatch):
    device = torch.device("cuda")
    dataset, loader, schedule = make_loader(
        om4_store, monkeypatch, device, 2, 1, 4, True
    )
    buffers = []
    original_acquire = loader._pinned_pool.acquire

    def acquire(shape):
        buffer = original_acquire(shape)
        assert buffer.is_pinned()
        buffers.append(buffer.data_ptr())
        return buffer

    monkeypatch.setattr(loader._pinned_pool, "acquire", acquire)
    try:
        for batch, indices in zip(loader, schedule, strict=True):
            expected = dataset.to_model_batch(
                collate_host_batches([dataset[i] for i in indices]), device
            )
            for actual_step, expected_step in zip(
                batch.steps, expected.steps, strict=True
            ):
                for actual, reference in zip(actual_step, expected_step, strict=True):
                    torch.testing.assert_close(
                        actual, reference, rtol=0, atol=0, equal_nan=True
                    )
            torch.cuda.synchronize()
    finally:
        loader.close()
    assert buffers


def test_tensorstore_compact_level_and_channel_order(om4_store):
    root, _ = om4_store
    group = zarr.open_group(str(root / "data.zarr"), mode="r")
    runtime = TensorStoreIoRuntime(2)
    selectors: list[PhysicalVariable]
    if "thetao" in group:
        reader = runtime.open(
            root / "data.zarr",
            [
                PhysicalVariable("thetao", 2),
                PhysicalVariable("hfds"),
                PhysicalVariable("thetao", 0),
            ],
        )
        selectors = [
            PhysicalVariable("thetao", 2),
            PhysicalVariable("hfds"),
            PhysicalVariable("thetao", 0),
        ]
        data = xr.open_zarr(root / "data.zarr", chunks=None)
        expected = np.stack(
            [
                data.thetao.isel(time=[4, 1], lev=2).transpose("time", "y", "x"),
                data.hfds.isel(time=[4, 1]),
                data.thetao.isel(time=[4, 1], lev=0).transpose("time", "y", "x"),
            ],
            axis=1,
        )
    else:
        selectors = [
            PhysicalVariable(name) for name in ["thetao_2", "hfds", "thetao_0"]
        ]
        reader = runtime.open(root / "data.zarr", selectors)
        data = xr.open_zarr(root / "data.zarr", chunks=None)
        expected = np.stack([data[v.name].isel(time=[4, 1]) for v in selectors], axis=1)
    output = np.empty((2, 3, 2, 3), dtype=np.float32)
    reader.read_into([4, 1], selectors, output)
    np.testing.assert_equal(output, expected)


@pytest.mark.parametrize(
    "attribute,value",
    [("scale_factor", 2.0), ("add_offset", 1.0), ("missing_value", -999.0)],
)
def test_tensorstore_rejects_cf_encoding(om4_store, attribute, value):
    root, _ = om4_store
    group = zarr.open_group(str(root / "data.zarr"), mode="a")
    group.hfds.attrs[attribute] = value
    with pytest.raises(ValueError, match="does not decode|requires NaN fill"):
        TensorStoreIoRuntime(2).open(root / "data.zarr", [PhysicalVariable("hfds")])


@pytest.mark.parametrize("failure", ["submission", "completion"])
@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_tensorstore_drains_writes_before_releasing_failed_buffer(
    om4_store, failure, error_type
):
    root, _ = om4_store
    reader = TensorStoreIoRuntime(2).open(
        root / "data.zarr", [PhysicalVariable("hfds")]
    )
    submitted: list[int] = []
    completed = []

    def array(output, **kwargs):
        index = len(submitted)
        submitted.append(index)

        def write(source):
            if failure == "submission" and index == 1:
                raise error_type("submission failed")

            def result():
                completed.append(index)
                if failure == "completion" and index == 0:
                    raise error_type("completion failed")
                output[:] = 5

            return SimpleNamespace(result=result, done=lambda: True)

        return SimpleNamespace(write=write)

    reader._ts = SimpleNamespace(array=array)
    output = np.empty((2, 2, 2, 3), dtype=np.float32)
    with pytest.raises(BaseExceptionGroup, match="TensorStore OM4 reads failed"):
        reader.read_into(
            [0, 1], [PhysicalVariable("hfds"), PhysicalVariable("hfds")], output
        )
    assert completed == ([0] if failure == "submission" else [0, 1])


@pytest.mark.parametrize(
    "times,variables", [([], [PhysicalVariable("hfds")]), ([0], []), ([], [])]
)
def test_tensorstore_empty_reads(om4_store, times, variables):
    root, _ = om4_store
    reader = TensorStoreIoRuntime(2).open(
        root / "data.zarr", [PhysicalVariable("hfds")]
    )
    output = np.empty((len(times), len(variables), 2, 3), dtype=np.float32)
    reader.read_into(times, variables, output)


def test_tensorstore_drains_copy_after_interrupted_wait(om4_store):
    root, _ = om4_store
    reader = TensorStoreIoRuntime(2).open(
        root / "data.zarr", [PhysicalVariable("hfds")]
    )
    waits = []

    def array(output, **kwargs):
        def result():
            waits.append(1)
            if len(waits) == 1:
                raise KeyboardInterrupt()
            output[:] = 5

        future = SimpleNamespace(result=result, done=lambda: len(waits) > 1)
        return SimpleNamespace(write=lambda source: future)

    reader._ts = SimpleNamespace(array=array)
    output = np.empty((1, 1, 2, 3), dtype=np.float32)
    with pytest.raises(BaseExceptionGroup):
        reader.read_into([0], [PhysicalVariable("hfds")], output)
    assert len(waits) == 2
    np.testing.assert_equal(output, 5)
