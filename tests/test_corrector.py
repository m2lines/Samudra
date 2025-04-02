from functools import partial

import pytest
import torch
import xarray as xr

from ocean_emulators.aggregator.metrics import area_weighted_sum
from ocean_emulators.models.corrector import (
    OceanHeatCorrector,
    ReLUCorrector,
    compute_ocean_heat_content,
)
from ocean_emulators.utils.data import Normalize
from ocean_emulators.utils.device import get_device
from ocean_emulators.utils.multiton import MultitonScope


@pytest.fixture
def corrector_init():
    # Create test data with mean and std
    data_mean = xr.Dataset(
        {
            "var_0": (["lat", "lon"], [[0.0]]),
            "var_1": (["lat", "lon"], [[0.0]]),
            "var_2": (["lat", "lon"], [[0.0]]),
        },
        coords={"lat": [0], "lon": [0]},
    )
    data_std = xr.Dataset(
        {
            "var_0": (["lat", "lon"], [[1.0]]),
            "var_1": (["lat", "lon"], [[1.0]]),
            "var_2": (["lat", "lon"], [[1.0]]),
        },
        coords={"lat": [0], "lon": [0]},
    )

    # Create test wet mask
    wet_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    class MockTensorMap:
        def __init__(self):
            self.VAR_3D_IDX = {
                "var_0": torch.tensor([0]),
                "var_1": torch.tensor([1]),
            }
            self.INPT_BOUNDARY_IDX = {
                "var_2": torch.tensor([0]),
            }
            self.DP_3D_IDX = {
                "0": torch.tensor([0]),
                "1": torch.tensor([1]),
            }
            self.prognostic_var_names = ["var_0", "var_1"]
            self.boundary_var_names = ["var_2"]
            self.dz = torch.tensor([1.0, 1.0])

    tensor_map = MockTensorMap()
    with MultitonScope():
        normalize = Normalize.init_instance(
            data_mean=data_mean,
            data_std=data_std,
            prognostic_var_names=["var_0", "var_1"],
            boundary_var_names=["var_2"],
            wet_mask=wet_mask,
            wet_mask_surface=wet_mask,
        )

    return normalize, tensor_map, wet_mask


def test_relu_corrector(corrector_init):
    normalize, tensor_map, wet_mask = corrector_init
    # area_weights = torch.ones(wet_mask.shape)
    mock_input = torch.randn([2, 3, *wet_mask.shape]).to(get_device())
    data = torch.ones([2, 2, *wet_mask.shape]).to(get_device())
    data[:, 0, :, :] = -1.0
    data[:, 1, :, :] = -2.0
    data = data * wet_mask.to(get_device())
    corrector = ReLUCorrector(
        non_negative_corrector_names=["var_1"],
        hist=0,
        tensor_map=tensor_map,
        normalize=normalize,
    )
    output = corrector(mock_input, data)
    assert torch.all(output[:, 0, 0, 0] == -1), (
        "var0 should still be negative after correction"
    )
    assert torch.all(output[:, 1, 0, 0] == 0), (
        "var1 should be non-negative after correction"
    )


def test_hist_corrector(corrector_init):
    normalize, tensor_map, wet_mask = corrector_init
    hist = 1
    corrector = ReLUCorrector(
        non_negative_corrector_names=["var_1"],
        hist=hist,
        tensor_map=tensor_map,
        normalize=normalize,
    )
    data = torch.ones([2, 2 * (hist + 1), *wet_mask.shape]).to(get_device())
    data[:, 0, :, :] = -1.0
    data[:, 2, :, :] = -1.0
    data[:, 1, :, :] = -2.0
    data[:, 3, :, :] = -2.0
    data = data * wet_mask.to(get_device())
    output = corrector(data, data)
    assert output.shape == (2, 2 * (hist + 1), *wet_mask.shape)
    assert torch.all(output[:, 0, 0, 0] == -1), (
        "var0 at step 0 should still be negative after correction"
    )
    assert torch.all(output[:, 1, 0, 0] == 0), (
        "var1 at step 0 should be non-negative after correction"
    )
    assert torch.all(output[:, 2, 0, 0] == -1), (
        "var0 at step 1 should be negative after correction"
    )
    assert torch.all(output[:, 3, 0, 0] == 0), (
        "var1 at step 1 should be non-negative after correction"
    )


@pytest.fixture
def ocean_heat_init():
    # Create test data with mean and std
    data_mean = xr.Dataset(
        {
            "thetao_0": (["lat", "lon"], [[0.0]]),
            "thetao_1": (["lat", "lon"], [[0.0]]),
            "thetao_2": (["lat", "lon"], [[0.0]]),
            "hfds": (["lat", "lon"], [[0.0]]),
        },
        coords={"lat": [0], "lon": [0]},
    )
    data_std = xr.Dataset(
        {
            "thetao_0": (["lat", "lon"], [[1.0]]),
            "thetao_1": (["lat", "lon"], [[1.0]]),
            "thetao_2": (["lat", "lon"], [[1.0]]),
            "hfds": (["lat", "lon"], [[1.0]]),
        },
        coords={"lat": [0], "lon": [0]},
    )

    # Create test wet mask
    wet_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])

    class MockTensorMap:
        def __init__(self):
            self.VAR_3D_IDX = {
                "thetao": torch.tensor([0, 1, 2]),
            }
            self.INPT_BOUNDARY_IDX = {
                "hfds": torch.tensor([0]),
            }
            self.DP_3D_IDX = {
                "0": torch.tensor([0]),
                "1": torch.tensor([1]),
                "2": torch.tensor([2]),
            }
            self.prognostic_var_names = ["thetao_0", "thetao_1", "thetao_2"]
            self.boundary_var_names = ["hfds"]
            self.dz: torch.Tensor = torch.tensor([1.0, 2.0, 4.0]).to(get_device())

    tensor_map = MockTensorMap()
    with MultitonScope():
        normalize = Normalize.init_instance(
            data_mean=data_mean,
            data_std=data_std,
            prognostic_var_names=["thetao_0", "thetao_1", "thetao_2"],
            boundary_var_names=["hfds"],
            wet_mask=wet_mask,
            wet_mask_surface=wet_mask,
        )

    wet_mask = wet_mask.to(get_device())
    return normalize, tensor_map, wet_mask


def test_ocean_heat_content(ocean_heat_init):
    normalize, tensor_map, wet_mask = ocean_heat_init
    T = torch.ones([1, 3, *wet_mask.shape]).to(get_device())
    T = T * wet_mask
    dz = tensor_map.dz
    area_weights = torch.ones(wet_mask.shape).to(get_device())
    area_weighted_func = partial(area_weighted_sum, area_weights=area_weights)

    global_HC_t, HC_t = compute_ocean_heat_content(T, dz, area_weighted_func)
    assert global_HC_t == 86766120
    assert HC_t[0, 0, 0, 0] == 4131720
    assert HC_t[0, 1, 0, 0] == 8263440
    assert HC_t[0, 2, 0, 0] == 16526880


def test_ocean_heat_corrector(ocean_heat_init):
    normalize, tensor_map, wet_mask = ocean_heat_init
    hfgeou_tensor = torch.ones_like(wet_mask)
    sea_surface_fraction_tensor = torch.ones_like(wet_mask)
    corrector = OceanHeatCorrector(
        hist=0,
        area_weights=torch.ones(wet_mask.shape),
        tensor_map=tensor_map,
        normalize=normalize,
        hfgeou_tensor=hfgeou_tensor,
        sea_surface_fraction_tensor=sea_surface_fraction_tensor,
    )
    input_tensor = torch.ones([1, 4, *wet_mask.shape]).to(get_device())
    input_tensor = input_tensor * wet_mask
    pred_tensor = torch.ones([1, 3, *wet_mask.shape]).to(get_device())
    pred_tensor = pred_tensor * wet_mask
    output = corrector(input_tensor, pred_tensor)

    total_next_step_heat = 86766120 + 3024000
    pred_heat = 86766120
    ratio = total_next_step_heat / pred_heat
    corrected_pred_tensor = pred_tensor[:, :3, :, :] * ratio
    assert torch.all(corrected_pred_tensor[:, 0, 0, 0] == output[:, 0, 0, 0])
