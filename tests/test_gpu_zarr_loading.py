import contextlib

import dask.array as da
import numpy as np
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader

import ocean_emulators.datasets as datasets_mod
from ocean_emulators.constants import OM4_DATASET_SPEC
from ocean_emulators.datasets import (
    InferenceDataset,
    TorchTrainDataset,
    TrainDataLoader,
    _dataarray_to_torch_float32,
)
from ocean_emulators.utils.data import DataSource, Masks, Normalize
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.train import collate_raw_train_data


def test_dataarray_to_torch_float32_handles_dask_arrays():
    data = np.arange(6, dtype=np.float64).reshape(2, 3)
    xarr = xr.DataArray(da.from_array(data, chunks=(1, 3)), dims=["x", "y"])

    tensor = _dataarray_to_torch_float32(xarr)

    assert tensor.dtype == torch.float32
    assert torch.equal(tensor, torch.tensor(data, dtype=torch.float32))


def test_materialize_dataarray_to_torch_float32_builds_array_inside_gpu_context(
    monkeypatch,
):
    ds = xr.Dataset({"foo": (("x", "y"), np.arange(6, dtype=np.float32).reshape(2, 3))})
    entered = False
    to_array_inside_context = False
    original_to_array = xr.Dataset.to_array

    @contextlib.contextmanager
    def fake_gpu_context(use_gpu_zarr_decode: bool):
        nonlocal entered
        assert use_gpu_zarr_decode is True
        entered = True
        try:
            yield
        finally:
            entered = False

    def wrapped_to_array(self, *args, **kwargs):
        nonlocal to_array_inside_context
        to_array_inside_context = entered
        return original_to_array(self, *args, **kwargs)

    monkeypatch.setattr(datasets_mod, "_zarr_gpu_decode_context", fake_gpu_context)
    monkeypatch.setattr(xr.Dataset, "to_array", wrapped_to_array)

    tensor = datasets_mod._materialize_dataarray_to_torch_float32(
        lambda: ds.to_array(),
        use_zarr_gpu_decode=True,
    )

    assert to_array_inside_context is True
    assert torch.equal(tensor, torch.tensor(ds["foo"].values).unsqueeze(0))


@pytest.fixture
def tiny_dataset_input():
    prognostic_var_names = ["prognostic1", "prognostic2"]
    boundary_var_names = ["boundary1", "boundary2"]
    data = xr.Dataset(
        {
            "prognostic1": (
                ["time", "lat", "lon"],
                [
                    [[0.0, 1.0], [2.0, 3.0]],
                    [[4.0, 5.0], [6.0, 7.0]],
                    [[8.0, 9.0], [10.0, 11.0]],
                    [[12.0, 13.0], [14.0, 15.0]],
                    [[16.0, 17.0], [18.0, 19.0]],
                    [[20.0, 21.0], [22.0, 23.0]],
                ],
            ),
            "prognostic2": (
                ["time", "lat", "lon"],
                [
                    [[0.5, 1.5], [2.5, 3.5]],
                    [[4.5, 5.5], [6.5, 7.5]],
                    [[8.5, 9.5], [10.5, 11.5]],
                    [[12.5, 13.5], [14.5, 15.5]],
                    [[16.5, 17.5], [18.5, 19.5]],
                    [[20.5, 21.5], [22.5, 23.5]],
                ],
            ),
            "boundary1": (
                ["time", "lat", "lon"],
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[13.0, 14.0], [15.0, 16.0]],
                    [[17.0, 18.0], [19.0, 20.0]],
                    [[21.0, 22.0], [23.0, 24.0]],
                ],
            ),
            "boundary2": (
                ["time", "lat", "lon"],
                [
                    [[1.5, 2.5], [3.5, 4.5]],
                    [[5.5, 6.5], [7.5, 8.5]],
                    [[9.5, 10.5], [11.5, 12.5]],
                    [[13.5, 14.5], [15.5, 16.5]],
                    [[17.5, 18.5], [19.5, 20.5]],
                    [[21.5, 22.5], [23.5, 24.5]],
                ],
            ),
        },
        coords={"time": range(6), "lat": [0, 1], "lon": [0, 1]},
    )
    data_mean = xr.Dataset(
        {
            "prognostic1": 0.5,
            "prognostic2": 0.5,
            "boundary1": 0.5,
            "boundary2": 0.5,
        },
        coords={"lat": [0], "lon": [0]},
    )
    data_std = xr.Dataset(
        {
            "prognostic1": 1.0,
            "prognostic2": 1.0,
            "boundary1": 1.0,
            "boundary2": 1.0,
        },
        coords={"lat": [0], "lon": [0]},
    )

    wet_surface = torch.ones(2, 2)
    wet_surface[0, 0] = 0.0
    wet_surface[1, 1] = 0.0
    wet = wet_surface.expand(2, 2, 2)
    masks = Masks(prognostic=wet, boundary=wet_surface)
    test = DataSource(
        "test",
        data,
        data_mean,
        data_std,
        masks=masks,
        dataset_spec=OM4_DATASET_SPEC,
    )

    with MultitonScope():
        _ = Normalize(
            test,
            prognostic_var_names=prognostic_var_names,
            boundary_var_names=boundary_var_names,
        )
        torch_train_dataset = TorchTrainDataset(
            src=test,
            dst=None,
            prognostic_var_names=prognostic_var_names,
            boundary_var_names=boundary_var_names,
            hist=1,
            steps=2,
            normalize_before_mask=True,
            masked_fill_value=0.0,
            stride=1,
        )
        inference_dataset = InferenceDataset(
            src=test,
            prognostic_var_names=prognostic_var_names,
            boundary_var_names=boundary_var_names,
            hist=1,
            normalize_before_mask=True,
            masked_fill_value=0.0,
            long_rollout=True,
        )
        raw_loader = DataLoader(
            torch_train_dataset,
            batch_size=1,
            collate_fn=collate_raw_train_data,
        )
        train_loader = TrainDataLoader(
            raw_loader, [torch_train_dataset], torch.device("cpu")
        )
        yield train_loader, inference_dataset


def test_train_dataset_gpu_decode_context_reentered_on_materialization(
    tiny_dataset_input, monkeypatch
):
    train_loader, _ = tiny_dataset_input
    train_loader.dataset.use_zarr_gpu_decode = True
    entered = 0

    @contextlib.contextmanager
    def fake_gpu_context(use_gpu_zarr_decode: bool):
        nonlocal entered
        assert use_gpu_zarr_decode is True
        entered += 1
        yield

    monkeypatch.setattr(datasets_mod, "_zarr_gpu_decode_context", fake_gpu_context)

    _ = train_loader[0]

    assert entered >= 3


def test_inference_dataset_gpu_decode_context_reentered_on_materialization(
    tiny_dataset_input, monkeypatch
):
    _, inference_dataset = tiny_dataset_input
    inference_dataset.use_zarr_gpu_decode = True
    entered = 0

    @contextlib.contextmanager
    def fake_gpu_context(use_gpu_zarr_decode: bool):
        nonlocal entered
        assert use_gpu_zarr_decode is True
        entered += 1
        yield

    monkeypatch.setattr(datasets_mod, "_zarr_gpu_decode_context", fake_gpu_context)

    _ = inference_dataset[0]

    assert entered >= 3
