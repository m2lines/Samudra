# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
import torch
import xarray as xr

from samudra.aggregator.metrics import area_weighted_sum
from samudra.constants import DataLayout
from samudra.derived_variables import compute_global_ocean_heat_content
from samudra.models.corrector import Correctors, OceanHeatCorrector, ReLUCorrector
from samudra.utils.data import BatchPreprocessor, CanonicalSource, Masks
from samudra.utils.device import get_device


def _relu_data_layout() -> DataLayout:
    return DataLayout(
        depth_levels=(1.0,),
        depth_thickness=(1.0,),
        prognostic_var_names=["thetao", "so"],
        boundary_var_names=["hfds"],
        default_metadata={},
        ocean_heat_temperature_var="thetao",
        surface_heat_flux_var="hfds",
        seconds_per_time_step=1,
    )


def _corrector_init(data_layout: DataLayout):
    # Create test data with mean and std
    data = xr.Dataset(
        {var: (["lat", "lon"], np.random.randn(2, 2)) for var in data_layout.variables}
        | {"hfds": (["lat", "lon"], np.random.randn(2, 2))},
        coords={"lat": np.arange(2), "lon": np.arange(2)},
    )
    data_mean = xr.Dataset(
        {var: 0.0 for var in data_layout.variables} | {"hfds": 0.0},
        coords={"lat": [0], "lon": [0]},
    )
    data_std = xr.Dataset(
        {var: 1.0 for var in data_layout.variables} | {"hfds": 1.0},
        coords={"lat": [0], "lon": [0]},
    )

    # Create test wet mask
    wet_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    masks = Masks(prognostic=wet_mask, boundary=wet_mask)

    source = CanonicalSource.from_canonical_datasets(
        name="test",
        data=data,
        means=data_mean,
        stds=data_std,
        masks=masks,
        data_layout=data_layout,
    )
    normalize = BatchPreprocessor(
        source,
        prognostic_var_names=data_layout.prognostic_var_names,
        boundary_var_names=data_layout.boundary_var_names,
    )

    return normalize, data_layout, wet_mask


def test_relu_corrector():
    data_layout = _relu_data_layout()
    normalize, data_layout, wet_mask = _corrector_init(data_layout)
    mock_input = torch.randn([2, 3, *wet_mask.shape]).to(get_device())
    data = torch.ones([2, 2, *wet_mask.shape]).to(get_device())
    data[:, 0, :, :] = -1.0
    data[:, 1, :, :] = -2.0
    data = data * wet_mask.to(get_device())
    corrector = ReLUCorrector(
        non_negative_corrector_names=["so"],
        input_steps=1,
        data_layout=data_layout,
        normalize=normalize,
    )
    output = corrector(mock_input, data)
    assert torch.all(output[:, 0, 0, 0] == -1), (
        "thetao should still be negative after correction"
    )
    assert torch.all(output[:, 1, 0, 0] == 0), (
        "so should be non-negative after correction"
    )


def test_hist_corrector():
    data_layout = _relu_data_layout()
    normalize, data_layout, wet_mask = _corrector_init(data_layout)
    input_steps = 2
    corrector = ReLUCorrector(
        non_negative_corrector_names=["so"],
        input_steps=input_steps,
        data_layout=data_layout,
        normalize=normalize,
    )
    data = torch.ones([2, 2 * input_steps, *wet_mask.shape]).to(get_device())
    data[:, 0, :, :] = -1.0
    data[:, 2, :, :] = -1.0
    data[:, 1, :, :] = -2.0
    data[:, 3, :, :] = -2.0
    data = data * wet_mask.to(get_device())
    output = corrector(data, data)
    assert output.shape == (2, 2 * input_steps, *wet_mask.shape)
    assert torch.all(output[:, 0, 0, 0] == -1), (
        "thetao at step 0 should still be negative after correction"
    )
    assert torch.all(output[:, 1, 0, 0] == 0), (
        "so at step 0 should be non-negative after correction"
    )
    assert torch.all(output[:, 2, 0, 0] == -1), (
        "thetao at step 1 should be negative after correction"
    )
    assert torch.all(output[:, 3, 0, 0] == 0), (
        "so at step 1 should be non-negative after correction"
    )


def _ocean_heat_data_layout() -> DataLayout:
    return DataLayout(
        depth_levels=(2.5, 10.0, 22.5),
        depth_thickness=(1.0, 2.0, 4.0),
        prognostic_var_names=["thetao_0", "thetao_1", "thetao_2"],
        boundary_var_names=["hfds"],
        default_metadata={},
        ocean_heat_temperature_var="thetao",
        surface_heat_flux_var="hfds",
        seconds_per_time_step=5 * 24 * 60 * 60,
    )


def _ocean_heat_init():
    data_layout = _ocean_heat_data_layout()
    data = xr.Dataset(
        {
            "thetao_0": (["lat", "lon"], np.random.randn(2, 2)),
            "thetao_1": (["lat", "lon"], np.random.randn(2, 2)),
            "thetao_2": (["lat", "lon"], np.random.randn(2, 2)),
            "hfds": (["lat", "lon"], np.random.randn(2, 2)),
        },
        coords={"lat": np.arange(2), "lon": np.arange(2)},
    )
    data_mean = xr.Dataset(
        {"thetao_0": 0.0, "thetao_1": 0.0, "thetao_2": 0.0, "hfds": 0.0},
        coords={"lat": [0], "lon": [0]},
    )
    data_std = xr.Dataset(
        {"thetao_0": 1.0, "thetao_1": 1.0, "thetao_2": 1.0, "hfds": 1.0},
        coords={"lat": [0], "lon": [0]},
    )
    # Create test wet mask
    wet_mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    masks = Masks(prognostic=wet_mask, boundary=wet_mask)
    source = CanonicalSource.from_canonical_datasets(
        name="test",
        data=data,
        means=data_mean,
        stds=data_std,
        masks=masks,
        data_layout=data_layout,
    )
    normalize = BatchPreprocessor(
        source,
        prognostic_var_names=data_layout.prognostic_var_names,
        boundary_var_names=data_layout.boundary_var_names,
    )

    wet_mask = wet_mask.to(get_device())
    return normalize, data_layout, wet_mask


def test_ocean_heat_content():
    normalize, data_layout, wet_mask = _ocean_heat_init()
    T = torch.ones([1, 3, *wet_mask.shape]).to(get_device())
    T = T * wet_mask
    dz = data_layout.dz.to(get_device())
    area_weights = torch.ones(wet_mask.shape).to(get_device())
    area_weighted_func = partial(area_weighted_sum, area_weights=area_weights)

    global_HC_t = compute_global_ocean_heat_content(T, dz, area_weighted_func)
    """
    Global heat = RHO * CP * T * int(dz) * int(area) * mask
     = 1035 * 3992 * 1 * (1+2+4) * (4-1)
     = 86766120
    """
    assert global_HC_t == 86766120


def test_ocean_heat_corrector():
    normalize, data_layout, wet_mask = _ocean_heat_init()
    hfgeou_tensor = torch.ones_like(wet_mask)
    sea_surface_fraction_tensor = torch.ones_like(wet_mask)
    corrector = OceanHeatCorrector(
        input_steps=1,
        area_weights=torch.ones(wet_mask.shape),
        data_layout=data_layout,
        normalize=normalize,
        hfgeou_tensor=hfgeou_tensor,
        sea_surface_fraction_tensor=sea_surface_fraction_tensor,
        seconds_per_time_step=data_layout.seconds_per_time_step,
    )
    input_tensor = torch.ones([1, 4, *wet_mask.shape]).to(get_device())
    input_tensor = input_tensor * wet_mask
    pred_tensor = torch.ones([1, 3, *wet_mask.shape]).to(get_device())
    pred_tensor = pred_tensor * wet_mask
    output = corrector(input_tensor, pred_tensor)

    # Global heat = RHO * CP * T * int(dz) * int(area) * mask
    # Total next step heat = Global heat + Heat flux from atmosphere and sea floor
    total_next_step_heat = 86766120 + 3024000
    pred_heat = 86766120
    ratio = total_next_step_heat / pred_heat
    corrected_pred_tensor = pred_tensor[:, :3, :, :] * ratio
    assert torch.all(corrected_pred_tensor[:, 0, 0, 0] == output[:, 0, 0, 0])

    # The raw prediction under-predicts the expected heat content here (ratio
    # != 1), so the corrector should record a nonzero imbalance -- this is
    # what a soft imbalance penalty trains against, since the delivered
    # (corrected) field above closes the budget exactly regardless. See
    # https://arxiv.org/abs/2607.18416.
    assert corrector.last_imbalance_sq is not None
    expected_imbalance_sq = torch.tensor(
        [(ratio - 1) ** 2], dtype=corrector.last_imbalance_sq.dtype
    )
    assert torch.allclose(corrector.last_imbalance_sq, expected_imbalance_sq)


def test_correctors_exposes_imbalance_penalty():
    normalize, data_layout, wet_mask = _ocean_heat_init()
    static_data = xr.Dataset(
        {
            "hfgeou": (["lat", "lon"], np.ones(wet_mask.shape)),
            "sea_surface_fraction": (["lat", "lon"], np.ones(wet_mask.shape)),
        },
        coords={"lat": np.arange(wet_mask.shape[0]), "lon": np.arange(wet_mask.shape[1])},
    )
    correctors = Correctors(
        non_negative_corrector_names=None,
        ocean_heat_corrector=True,
        input_steps=1,
        area_weights=torch.ones(wet_mask.shape),
        static_data=static_data,
        data_layout=data_layout,
        normalize=normalize,
        imbalance_penalty_weight=0.5,
    )
    assert correctors.last_imbalance_penalty is None

    input_tensor = (torch.ones([1, 4, *wet_mask.shape]) * wet_mask).to(get_device())
    pred_tensor = (torch.ones([1, 3, *wet_mask.shape]) * wet_mask).to(get_device())
    correctors(input_tensor, pred_tensor)

    assert correctors.last_imbalance_penalty is not None
    assert correctors.last_imbalance_penalty > 0
    assert correctors.imbalance_penalty_weight == 0.5


def test_correctors_imbalance_penalty_none_without_ocean_heat_corrector():
    data_layout = _relu_data_layout()
    normalize, data_layout, wet_mask = _corrector_init(data_layout)
    correctors = Correctors(
        non_negative_corrector_names=["so"],
        ocean_heat_corrector=False,
        input_steps=1,
        area_weights=torch.ones(wet_mask.shape),
        static_data=None,
        data_layout=data_layout,
        normalize=normalize,
    )
    mock_input = torch.randn([2, 3, *wet_mask.shape]).to(get_device())
    data = (torch.ones([2, 2, *wet_mask.shape]) * wet_mask).to(get_device())
    correctors(mock_input, data)

    assert correctors.last_imbalance_penalty is None
