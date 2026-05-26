import numpy as np
import torch
import xarray as xr

from ocean_emulators.config import GradientLossConfig, build_loss_fn
from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.loss import (
    decomposed_mse_mae,
    gradient_h_l1_loss,
    ts_gradient_z_l1_loss,
)
from ocean_emulators.utils.multiton import MultitonScope


def test_gradient_h_loss_supports_mse_mae_metric():
    pred = torch.tensor(
        [[[[1.0, 3.0], [2.0, 4.0]]]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[[[0.0, 1.0], [1.0, 2.0]]]],
        dtype=torch.float32,
    )
    wet = torch.ones((1, 2, 2), dtype=torch.float32)
    y_coord = xr.DataArray(np.array([0.0, 1.0]), dims=["lat"])
    lambda_h = 0.25

    loss_fn = build_loss_fn(
        GradientLossConfig(type="gradient_h", metric="mse_mae", lambda_h=lambda_h),
        wet=wet,
        y_coord=y_coord,
        device=torch.device("cpu"),
        num_channels=1,
        pad_mode="constant",
    )

    actual = loss_fn(pred, target)
    expected = decomposed_mse_mae(pred, target, wet) + lambda_h * gradient_h_l1_loss(
        pred, target, wet, pad_mode="constant"
    )

    assert torch.allclose(actual, expected)


def test_gradient_h_and_ts_gradient_z_can_be_combined():
    with MultitonScope():
        tensor_map = TensorMap.init_instance("single_2", "all")
        pred = torch.tensor(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[2.0, 4.0], [6.0, 8.0]],
                ]
            ],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[2.0, 2.0], [2.0, 2.0]],
                ]
            ],
            dtype=torch.float32,
        )
        wet = torch.ones(
            (len(tensor_map.prognostic_var_names), 2, 2), dtype=torch.float32
        )
        y_coord = xr.DataArray(np.array([0.0, 1.0]), dims=["lat"])
        lambda_h = 0.25
        lambda_z = 0.5

        loss_fn = build_loss_fn(
            GradientLossConfig(
                type=["gradient_h", "TS-gradient_z"],
                metric="mse_mae",
                lambda_h=lambda_h,
                lambda_z=lambda_z,
            ),
            wet=wet,
            y_coord=y_coord,
            device=torch.device("cpu"),
            num_channels=len(tensor_map.prognostic_var_names),
            pad_mode="constant",
        )

        actual = loss_fn(pred, target)
        expected = (
            decomposed_mse_mae(pred, target, wet)
            + lambda_h * gradient_h_l1_loss(pred, target, wet, pad_mode="constant")
            + lambda_z * ts_gradient_z_l1_loss(pred, target, wet)
        )

        assert torch.allclose(actual, expected)


def test_gradient_config_without_type_uses_base_metric_only():
    pred = torch.tensor(
        [[[[1.0, 3.0], [2.0, 4.0]]]],
        dtype=torch.float32,
    )
    target = torch.zeros_like(pred)
    wet = torch.ones((1, 2, 2), dtype=torch.float32)
    y_coord = xr.DataArray(np.array([0.0, 1.0]), dims=["lat"])

    loss_fn = build_loss_fn(
        GradientLossConfig(metric="mse_mae"),
        wet=wet,
        y_coord=y_coord,
        device=torch.device("cpu"),
        num_channels=1,
        pad_mode="constant",
    )

    assert torch.allclose(loss_fn(pred, target), decomposed_mse_mae(pred, target, wet))


def test_legacy_gradient_config_maps_to_gradient_h():
    cfg = GradientLossConfig.model_validate(
        {
            "type": ["gradient", "gradient_v"],
            "metric": "mse_mae",
            "alpha": 0.25,
            "lambda_v": 0.5,
        }
    )

    assert cfg.type == ["gradient_h", "TS-gradient_z"]
    assert cfg.lambda_h == 0.25
    assert cfg.lambda_z == 0.5
