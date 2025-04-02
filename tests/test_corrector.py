import pytest
import torch
import xarray as xr

from ocean_emulators.models.corrector import ReLUCorrector
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
                "var_2": torch.tensor([2]),
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
    output = corrector(data, data)
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
