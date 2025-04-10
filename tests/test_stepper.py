import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.constants import DEPTH_LEVELS, TensorMap
from ocean_emulators.datasets import InferenceDataset
from ocean_emulators.models.base import BaseModel
from ocean_emulators.utils.data import Normalize, extract_wet_mask, validate_data
from ocean_emulators.utils.multiton import MultitonScope


@pytest.fixture
def inf_data_init(hist: int):
    with MultitonScope():
        levels = 19
        lats = 1
        lons = 1
        total_time_steps = 100

        tensor_map = TensorMap.init_instance("thetao_surface", "hfds")

        # Even thetao, odd hfds for every time step
        # Ex, timestep 0: thetao = 0, hfds = 1
        # Ex, timestep 1: thetao = 2, hfds = 3
        # Ex, timestep 2: thetao = 4, hfds = 5
        # ...
        data = xr.Dataset(
            {
                **{
                    f"thetao_{lev}": (
                        ["time", "lat", "lon"],
                        np.tile(
                            np.arange(total_time_steps)[:, None, None] * 2,
                            (1, lats, lons),
                        ),
                    )
                    for lev in range(levels)
                },
                "hfds": (
                    ["time", "lat", "lon"],
                    np.tile(
                        np.arange(total_time_steps)[:, None, None] * 2 + 1,
                        (1, lats, lons),
                    ),
                ),
                "wetmask": (
                    ["time", "lev", "lat", "lon"],
                    np.ones((total_time_steps, levels, lats, lons)),
                ),
            },
            coords={
                "time": np.arange(total_time_steps),
                "lev": DEPTH_LEVELS,
                "lat": np.arange(lats),
                "lon": np.arange(lons),
            },
        )
        data_mean = data.mean() * 0.0
        data_std = data.std() * 0.0 + 1.0
        data, data_mean, data_std = validate_data(data, data_mean, data_std)
        wet, wet_surface = extract_wet_mask(data, tensor_map.prognostic_var_names, hist)
        wet_without_hist, _ = extract_wet_mask(data, tensor_map.prognostic_var_names, 0)

        _ = Normalize.init_instance(
            data_mean=data_mean,
            data_std=data_std,
            prognostic_var_names=tensor_map.prognostic_var_names,
            boundary_var_names=tensor_map.boundary_var_names,
            wet_mask=wet_without_hist,
        )
        inference_dataset = InferenceDataset(
            data,
            tensor_map.prognostic_var_names,
            tensor_map.boundary_var_names,
            wet,
            wet_surface,
            hist,
            long_rollout=True,
        )

        yield inference_dataset


class ModelRetBoundary(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_once(self, x):
        return x[:, :-1] * 0.0 + x[:, -1]


# These tests will fail with OHC PR
@pytest.mark.parametrize("hist", [0, 1, 2, 3, 4])
def test_inference_dataset(inf_data_init, hist):
    inference_dataset = inf_data_init
    num_input_channels = (hist + 1) + 1  # thetao * (hist + 1) + hfds
    num_prognostic_channels = hist + 1  # thetao * (hist + 1)

    input_0, target_0 = inference_dataset[0]

    # Index 0 test
    # For hist = 0, input is [0, 1]
    # For hist = 1, input is [0, 2, 3]
    assert input_0.shape == (1, num_input_channels, 1, 1)
    expected_input = torch.tensor(
        [2 * i for i in range(hist + 1)] + [2 * hist + 1], device=input_0.device
    )
    assert torch.equal(input_0.flatten(), expected_input)

    # For hist = 0, target is [2]
    # For hist = 1, target is [4, 6]
    assert target_0.shape == (1, num_prognostic_channels, 1, 1)
    expected_target = torch.tensor(
        [2 * i for i in range(hist + 1, (hist + 1) * 2)], device=target_0.device
    )
    assert torch.equal(target_0.flatten(), expected_target)

    # Loop test
    for cur_step in range(1, len(inference_dataset)):
        base_step = cur_step * (hist + 1)
        input_cur, target_cur = inference_dataset[cur_step]
        assert input_cur.shape == (1, num_input_channels, 1, 1)
        expected_input = torch.tensor(
            [2 * i for i in range(base_step, base_step + hist + 1)]
            + [2 * (base_step + hist) + 1],
            device=input_0.device,
        )
        assert torch.equal(input_cur.flatten(), expected_input)

        assert target_cur.shape == (1, num_prognostic_channels, 1, 1)
        expected_target = torch.tensor(
            [2 * i for i in range(base_step + hist + 1, base_step + 2 * (hist + 1))],
            device=target_0.device,
        )
        assert torch.equal(target_cur.flatten(), expected_target)


@pytest.mark.parametrize("hist", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("num_steps", [1, 2, 3, 10])
def test_inference_rollout(inf_data_init, hist, num_steps):
    inference_dataset = inf_data_init
    model = ModelRetBoundary(
        ch_width=[1],
        n_out=inference_dataset.num_prognostic_channels,
        wet=inference_dataset.wet,
        hist=hist,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
    )

    model.eval()
    IO = model.inference(inference_dataset, num_steps=num_steps, epoch=0)
    prediction = IO.prediction
    target = IO.target

    assert prediction.shape == target.shape

    # Test if we are extracting the correct targets
    # For hist = 0, target is [2, 4, 6]
    # For hist = 1, target is [4, 6, 8, 10, 12, 14]
    expected_target = torch.tensor(
        [2 * i for i in range(hist + 1, hist + 1 + num_steps * (hist + 1))],
        device=target.device,
    )
    assert torch.equal(target.flatten(), expected_target)

    # Test if we are extracting the correct boundary values
    # The model returns the boundary condition at the latest step for each step
    # For hist = 0, prediction is [1, 3, 5]
    # For hist = 1, prediction is [3, 3, 7, 7, 11, 11]
    # For hist = 2, prediction is [5, 5, 5, 11, 11, 11, 17, 17, 17]

    start = 2 * (hist + 1) - 1
    diff_step = (hist + 1) * 2
    repeat = hist + 1
    base = torch.arange(
        start, start + diff_step * num_steps, diff_step, device=prediction.device
    )
    base = base.repeat_interleave(repeat)
    expected_prediction = torch.tensor(base, device=prediction.device)
    assert torch.equal(prediction.flatten(), expected_prediction)


# These tests will fail with OHC PR
@pytest.mark.parametrize("hist", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("merge_step", [1, 2, 3])
def test_inference_rollout_methods(inf_data_init, hist, merge_step):
    inference_dataset = inf_data_init
    model = ModelRetBoundary(
        ch_width=[1],
        n_out=inference_dataset.num_prognostic_channels,
        wet=inference_dataset.wet,
        hist=hist,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
    )

    model.eval()
    num_input_channels = (hist + 1) + 1  # thetao * (hist + 1) + hfds
    num_prognostic_channels = hist + 1  # thetao * (hist + 1)
    input_tensor = inference_dataset.get_initial_input()

    assert input_tensor.shape == (1, num_input_channels, 1, 1)
    expected_input = torch.tensor(
        [2 * i for i in range(hist + 1)] + [2 * hist + 1], device=input_tensor.device
    )
    assert torch.equal(input_tensor.flatten(), expected_input)

    pred = model.forward_once(input_tensor)
    assert pred.shape == (1, num_prognostic_channels, 1, 1)
    expected_pred = torch.tensor([2 * hist + 1] * (hist + 1), device=pred.device)
    assert torch.equal(pred.flatten(), expected_pred)

    merged_input_tensor = inference_dataset.merge_prognostic_and_boundary(
        prognostic=pred,
        step=merge_step,
    )
    assert merged_input_tensor.shape == (1, num_input_channels, 1, 1)
    expected_merged_input = torch.tensor(
        [2 * hist + 1] * (hist + 1)
        + [(merge_step + 1) * (num_prognostic_channels * 2) - 1],
        device=merged_input_tensor.device,
    )
    assert torch.equal(merged_input_tensor.flatten(), expected_merged_input)
