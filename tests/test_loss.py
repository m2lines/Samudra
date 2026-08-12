import numpy as np
import pytest
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


def _ts_gradient_z_setup(batch: int):
    """A batch of multi-channel fields wired up like the real training loss."""
    tensor_map = TensorMap.init_instance("single_2", "all")
    num_channels = len(tensor_map.prognostic_var_names)
    torch.manual_seed(0)
    pred = torch.randn(batch, num_channels, 3, 3)
    target = torch.randn(batch, num_channels, 3, 3)
    wet = torch.ones((num_channels, 3, 3), dtype=torch.float32)
    return tensor_map, num_channels, pred, target, wet


def test_ts_gradient_z_accepts_a_per_channel_sample_weight():
    """Regression: the per-sample weight is [batch, channel, lat, lon], not a
    surface field. Reshaping it as if it had one channel made every grouped
    training run die at the first optimizer step."""
    with MultitonScope():
        _, num_channels, pred, target, wet = _ts_gradient_z_setup(batch=8)
        sample_weight = torch.ones(8, num_channels, 3, 3)

        loss = ts_gradient_z_l1_loss(
            pred, target, wet, sample_weight=sample_weight
        )
        assert loss.shape == (num_channels,)
        assert torch.isfinite(loss).all()


def test_ts_gradient_z_all_ones_sample_weight_matches_no_weight():
    """A weight of one everywhere must be a no-op, or the weighted form is not a
    generalization of the unweighted one."""
    with MultitonScope():
        _, num_channels, pred, target, wet = _ts_gradient_z_setup(batch=4)
        unweighted = ts_gradient_z_l1_loss(pred, target, wet)
        weighted = ts_gradient_z_l1_loss(
            pred, target, wet, sample_weight=torch.ones(4, num_channels, 3, 3)
        )
        torch.testing.assert_close(unweighted, weighted)


def test_ts_gradient_z_sample_weight_can_zero_out_a_sample():
    """Dropping one member of the batch must equal never having supplied it --
    this is what lets a tile be scored only on the cells it owns."""
    with MultitonScope():
        _, num_channels, pred, target, wet = _ts_gradient_z_setup(batch=2)
        weight = torch.ones(2, num_channels, 3, 3)
        weight[1] = 0.0

        masked = ts_gradient_z_l1_loss(pred, target, wet, sample_weight=weight)
        first_only = ts_gradient_z_l1_loss(pred[:1], target[:1], wet)
        torch.testing.assert_close(masked, first_only)


def test_ts_gradient_z_rejects_a_weight_with_the_wrong_channel_count():
    with MultitonScope():
        _, _, pred, target, wet = _ts_gradient_z_setup(batch=2)
        with pytest.raises(ValueError, match="TS-gradient_z needs"):
            ts_gradient_z_l1_loss(
                pred, target, wet, sample_weight=torch.ones(2, 7, 3, 3)
            )


def test_full_gradient_loss_runs_with_a_per_sample_weight():
    """The end-to-end path the training loop actually takes: metric plus both
    gradient terms, with a per-tile weight threaded through all three."""
    with MultitonScope():
        tensor_map, num_channels, pred, target, wet = _ts_gradient_z_setup(batch=4)
        loss_fn = build_loss_fn(
            GradientLossConfig(
                type=["gradient_h", "TS-gradient_z"],
                metric="mse_mae",
                lambda_h=0.1,
                lambda_z=0.1,
            ),
            wet=wet,
            y_coord=xr.DataArray(np.array([0.0, 1.0, 2.0]), dims=["lat"]),
            device=torch.device("cpu"),
            num_channels=num_channels,
            pad_mode="constant",
        )

        weight = torch.ones(4, num_channels, 3, 3)
        weight[:, :, 0, 0] = 0.0  # a cell this sample does not own
        loss = loss_fn(pred, target, sample_weight=weight)

        assert loss.shape == (num_channels,)
        assert torch.isfinite(loss).all()
        # And it is genuinely a different number from the unweighted loss.
        assert not torch.allclose(loss, loss_fn(pred, target))


def test_ts_gradient_z_is_a_mean_not_a_batch_sum():
    """Regression: the denominator sums over the batch axis, so it must count
    every sample's cells. Broadcasting a size-1 axis there scaled the whole term
    by the batch size -- invisible at batch=1, which is all the older tests used.

    Duplicating a sample cannot change a mean.
    """
    with MultitonScope():
        _, _, pred, target, wet = _ts_gradient_z_setup(batch=1)
        single = ts_gradient_z_l1_loss(pred, target, wet)
        quadrupled = ts_gradient_z_l1_loss(
            pred.repeat(4, 1, 1, 1), target.repeat(4, 1, 1, 1), wet
        )
        torch.testing.assert_close(single, quadrupled)
