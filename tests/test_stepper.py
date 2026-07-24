# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import xarray as xr

from samudra.constants import TensorMap
from samudra.datasets import InferenceDataset, ModelBatch
from samudra.models.base import BaseModel
from samudra.stepper import validate_batch
from samudra.utils.ctx import BatchGrid
from samudra.utils.data import CanonicalSource, Normalize
from samudra.utils.multiton import MultitonScope
from tests.conftest import TEST_DATA_LAYOUT


@pytest.fixture
def inf_data_init(hist: int):
    with MultitonScope():
        levels = 19
        lats = 1
        lons = 1
        total_time_steps = 100

        tensor_map = TensorMap(data_layout=TEST_DATA_LAYOUT)

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
                "lev": list(TEST_DATA_LAYOUT.depth_levels),
                "lat": np.arange(lats),
                "lon": np.arange(lons),
            },
        )
        data_mean: xr.Dataset = data.mean() * 0.0
        data_std: xr.Dataset = data.std() * 0.0 + 1.0
        val = CanonicalSource.from_datasets(
            data,
            data_mean,
            data_std,
            data_layout=TEST_DATA_LAYOUT,
            name="test-data",
            prognostic_var_names=tensor_map.prognostic_var_names,
            boundary_var_names=tensor_map.boundary_var_names,
        )

        _ = Normalize(
            val,
            prognostic_var_names=tensor_map.prognostic_var_names,
            boundary_var_names=tensor_map.boundary_var_names,
        )
        inference_dataset = InferenceDataset(
            val,
            tensor_map.prognostic_var_names,
            tensor_map.boundary_var_names,
            hist,
            normalize_before_mask=True,
            masked_fill_value=0.0,
            long_rollout=True,
        )

        yield inference_dataset, val.masks.prognostic_with_hist(hist)


class MockModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_once(
        self,
        prognostic: torch.Tensor,
        boundary: torch.Tensor,
        ctx: BatchGrid,
    ):
        # Exercises the two streams independently: scale prog, add the last
        # boundary channel.
        return prognostic * 10.0 + boundary[:, -1]


class ConstantResidualModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_once(self, prognostic, boundary, ctx: BatchGrid):
        return torch.ones_like(prognostic)


def test_validate_batch_uses_absolute_predictions_for_residual_models():
    wet = torch.ones((1, 1, 1, 1), dtype=torch.bool)
    grid = torch.zeros(1)
    ctx = BatchGrid(wet, (grid, grid), (grid, grid))
    batch = ModelBatch(num_prognostic_channels=1, num_boundary_channels=1, ctx=ctx)
    prog_input = torch.tensor([[[[10.0]]]])
    boundary_input = torch.tensor([[[[5.0]]]])
    label = torch.tensor([[[[11.0]]]])
    batch.append(prog_input, boundary_input, label)

    model = ConstantResidualModel(
        in_channels=2,
        out_channels=1,
        hist=0,
        pred_residuals=True,
        last_kernel_size=3,
        pad="circular",
        gradient_detach_interval=0,
    )

    output = validate_batch(
        model,
        batch,
        lambda pred, target, ctx: ((pred - target) ** 2).mean(dim=(0, 2, 3)),
    )

    assert torch.equal(
        output.input_data, torch.cat((prog_input, boundary_input), dim=1)
    )
    assert torch.equal(output.target_data, label)
    assert torch.equal(output.gen_data, label)
    assert torch.equal(output.loss_per_channel, torch.zeros(1))
    assert output.loss.item() == 0.0


# These tests will fail with OHC PR
@pytest.mark.parametrize("hist", [0, 1, 2, 3, 4])
def test_inference_dataset(inf_data_init, hist):
    inference_dataset, _ = inf_data_init
    num_input_channels = (hist + 1) * 2  # (hist + 1) * (thetao + hfds)
    num_prognostic_channels = hist + 1  # (hist + 1) * thetao

    prog_0, boundary_0, target_0 = inference_dataset[0]
    input_0 = torch.cat((prog_0, boundary_0), dim=1)

    # Index 0 test
    # For hist = 0, input is [0, 1]
    # For hist = 1, input is [0, 2, 1, 3]
    assert input_0.shape == (1, num_input_channels, 1, 1)
    expected_input = torch.tensor(
        [2 * i for i in range(hist + 1)] + [2 * i + 1 for i in range(hist + 1)],
        device=input_0.device,
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
        prog_cur, boundary_cur, target_cur = inference_dataset[cur_step]
        input_cur = torch.cat((prog_cur, boundary_cur), dim=1)
        assert input_cur.shape == (1, num_input_channels, 1, 1)
        expected_input = torch.tensor(
            [2 * i for i in range(base_step, base_step + hist + 1)]
            + [2 * i + 1 for i in range(base_step, base_step + hist + 1)],
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
@pytest.mark.parametrize("num_steps", [1, 2, 3])
def test_inference_rollout(inf_data_init, hist, num_steps):
    inference_dataset, wet = inf_data_init
    model = MockModel(
        in_channels=1,
        out_channels=inference_dataset.num_prognostic_channels,
        hist=hist,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        gradient_detach_interval=0,
    )

    model.eval()
    initial_prognostic = inference_dataset.initial_prognostic
    IO = model.inference(
        inference_dataset, initial_prognostic, num_steps=num_steps, epoch=0
    )
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
    # For hist = 0, prediction is [0*10+1=1, 1*10+3=13, 13*10+5=135]
    # For hist = 1, prediction is [0*10+3=3, 2*10+3=23, 3*10+7=37, 23*10+7=237, ...]
    # For hist = 2, prediction is [0*10+5=5, 2*10+5=25, 4*10+5=45, 5*10+11=61, ...]

    expected_prediction = torch.tensor(
        [20 * i for i in range(hist + 1)], device=prediction.device
    )
    expected_prediction = expected_prediction + 2 * hist + 1
    base = expected_prediction.clone()
    for i in range(1, num_steps):
        cur_acc = 2 * hist + 1 + 2 * i * (hist + 1)
        base = 10 * base + cur_acc
        expected_prediction = torch.cat((expected_prediction, base))

    assert torch.equal(prediction.flatten(), expected_prediction)


# These tests will fail with OHC PR
@pytest.mark.parametrize("hist", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("merge_step", [1, 2, 3])
def test_inference_rollout_methods(inf_data_init, hist, merge_step):
    inference_dataset, wet = inf_data_init
    model = MockModel(
        in_channels=1,
        out_channels=inference_dataset.num_prognostic_channels,
        hist=hist,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        gradient_detach_interval=0,
    )

    model.eval()
    num_input_channels = (hist + 1) * 2  # (hist + 1) * (thetao + hfds)
    num_prognostic_channels = hist + 1  # (hist + 1) * thetao
    prog_tensor, boundary_tensor = inference_dataset.get_initial_input()
    input_tensor = torch.cat((prog_tensor, boundary_tensor), dim=1)

    assert input_tensor.shape == (1, num_input_channels, 1, 1)
    expected_input = torch.tensor(
        [2 * i for i in range(hist + 1)] + [2 * i + 1 for i in range(hist + 1)],
        device=input_tensor.device,
    )
    assert torch.equal(input_tensor.flatten(), expected_input)

    pred = model.forward_once(
        prog_tensor,
        boundary_tensor,
        BatchGrid(wet, inference_dataset.input_res, inference_dataset.input_res),
    )
    assert pred.shape == (1, num_prognostic_channels, 1, 1)
    expected_pred = torch.tensor(
        [2 * hist + 1 + 2 * i * 10 for i in range(hist + 1)], device=pred.device
    )
    assert torch.equal(pred.flatten(), expected_pred)

    merged_prog = pred
    merged_boundary = inference_dataset.get_boundary(merge_step)
    merged_input_tensor = torch.cat((merged_prog, merged_boundary), dim=1)
    assert merged_input_tensor.shape == (1, num_input_channels, 1, 1)

    # For hist = 0, merge_step = 1, need to merge [3]
    # 0, 1 -> 2, 3
    # For hist = 0, merge_step = 2, need to merge [5]
    # 0, 1 -> 2, 3 -> 4, 5
    # For hist = 1, merge_step = 1, need to merge [5, 7]
    # 0, 2, 1, 3 -> 4, 6, 5, 7
    # For hist = 1, merge_step = 2, need to merge [9, 11]
    # 0, 2, 1, 3 -> 4, 6, 5, 7 -> 8, 10, 9, 11
    # For hist = 2, merge_step = 1, need to merge [7, 9, 11]
    # 0, 2, 4, 1, 3, 5 -> 6, 8, 10, 7, 9, 11
    # For hist = 2, merge_step = 2, need to merge [13, 15, 17]
    # 0, 2, 4, 1, 3, 5 -> 6, 8, 10, 7, 9, 11 -> 12, 14, 16, 13, 15, 17

    expected_merged_input = torch.tensor(
        [2 * hist + 1 + 2 * i * 10 for i in range(hist + 1)]
        + [2 * (hist + 1) * merge_step - 1 + 2 * (i + 1) for i in range(hist + 1)],
        device=merged_input_tensor.device,
    )
    assert torch.equal(merged_input_tensor.flatten(), expected_merged_input)


@pytest.mark.parametrize("hist", [0, 1, 2, 3])
@pytest.mark.parametrize("num_steps", [1, 2, 3])
@pytest.mark.parametrize("start_time", [0, 6, 15])
def test_inference_rollout_time(inf_data_init, hist, num_steps, start_time):
    inference_dataset, _ = inf_data_init
    target_time = inference_dataset.get_target_time(start_time, num_steps)

    # Base time
    # Hist = 0, start_time = 0, base_time = 1
    # Hist = 0, start_time = 1, base_time = 2
    # Hist = 1, start_time = 0, base_time = 2
    # Hist = 1, start_time = 2, base_time = 6
    # Hist = 2, start_time = 0, base_time = 3
    base_time = (start_time + 1) * (hist + 1)
    times = [i for i in range(base_time, base_time + num_steps * (hist + 1))]
    expected_target_time = xr.DataArray(
        data=times,
        dims=["time"],
        coords={"time": times},
    )
    assert target_time.size == expected_target_time.size
    assert target_time.equals(expected_target_time)
