import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.config import GradientLossConfig, build_loss_fn
from ocean_emulators.constants import DEPTH_I_LEVELS, PROGNOSTIC_VARS, TensorMap
from ocean_emulators.utils.loss import (
    decomposed_mse_mae,
    gradient_h_l1_loss,
    gradient_z_l1_loss,
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


def test_gradient_h_and_gradient_z_can_be_combined():
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
                type=["gradient_h", "gradient_z"],
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
            + lambda_z * gradient_z_l1_loss(pred, target, wet)
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


def test_legacy_gradient_config_maps_to_current_names():
    cfg = GradientLossConfig.model_validate(
        {
            "type": ["gradient", "gradient_v"],
            "metric": "mse_mae",
            "alpha": 0.25,
            "lambda_v": 0.5,
        }
    )

    assert cfg.type == ["gradient_h", "gradient_z"]
    assert cfg.lambda_h == 0.25
    assert cfg.lambda_z == 0.5


def test_legacy_ts_gradient_z_name_maps_to_gradient_z():
    """The vertical term was `TS-gradient_z` while it only covered T and S.
    Configs and job scripts on disk still spell it that way."""
    cfg = GradientLossConfig.model_validate(
        {"type": ["gradient_h", "TS-gradient_z"], "lambda_z": 0.1}
    )

    assert cfg.type == ["gradient_h", "gradient_z"]


def _gradient_z_setup(batch: int):
    """A batch of multi-channel fields wired up like the real training loss.

    Uses the full `all` variable set -- U, V, Theta, Salt over every depth level
    plus the 2D Eta -- because that is the configuration that trains, and the
    vertical term has to behave for all of it.
    """
    tensor_map = TensorMap.init_instance("all", "all")
    num_channels = len(tensor_map.prognostic_var_names)
    torch.manual_seed(0)
    pred = torch.randn(batch, num_channels, 3, 3)
    target = torch.randn(batch, num_channels, 3, 3)
    wet = torch.ones((num_channels, 3, 3), dtype=torch.float32)
    return tensor_map, num_channels, pred, target, wet


def test_gradient_z_covers_every_3d_variable():
    """The whole point of the generalization: U and V are penalized alongside T
    and S, and the 2D channels stay out of it because they have no column."""
    with MultitonScope():
        tensor_map, _, pred, target, wet = _gradient_z_setup(batch=2)

        loss = gradient_z_l1_loss(pred, target, wet)

        assert tensor_map.VAR_SET_3D == ["U", "V", "Theta", "Salt"]
        for variable in tensor_map.VAR_SET_3D:
            indices = tensor_map.VAR_3D_IDX[variable].long()
            assert (loss[indices] > 0).all(), f"{variable} is not penalized"
        eta = tensor_map.VAR_3D_IDX["Eta"].long()
        torch.testing.assert_close(loss[eta], torch.zeros_like(loss[eta]))


def test_gradient_z_picks_up_a_newly_added_3d_variable(monkeypatch):
    """The variable list is read off the tensor map rather than hard-coded, so a
    new 3D variable -- vertical velocity, say -- joins the penalty without any
    change to the loss."""
    monkeypatch.setitem(
        PROGNOSTIC_VARS,
        "with_w_51",
        [
            f"{name}_{level}"
            for name in ("U", "V", "Theta", "Salt", "W")
            for level in DEPTH_I_LEVELS
        ]
        + ["Eta"],
    )
    with MultitonScope():
        tensor_map = TensorMap.init_instance("with_w_51", "all")
        num_channels = len(tensor_map.prognostic_var_names)
        torch.manual_seed(0)
        pred = torch.randn(1, num_channels, 2, 2)
        target = torch.randn(1, num_channels, 2, 2)
        wet = torch.ones((num_channels, 2, 2), dtype=torch.float32)

        loss = gradient_z_l1_loss(pred, target, wet)

        assert "W" in tensor_map.VAR_SET_3D
        vertical_velocity = tensor_map.VAR_3D_IDX["W"].long()
        assert (loss[vertical_velocity] > 0).all()


def test_gradient_z_divides_by_level_spacing():
    """The load-bearing property. A level-to-level difference is a derivative,
    so the same jump between two levels 1.07 m apart and two levels 45.5 m apart
    must not cost the same. Without the division the tightly stacked near-surface
    levels -- the ones actually free to drift apart -- end up the least
    constrained channels in the column.
    """
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        num_channels = len(tensor_map.prognostic_var_names)
        theta = tensor_map.VAR_3D_IDX["Theta"].long()
        spacing = tensor_map.vertical_spacing("Theta")
        pair_weight = spacing.reciprocal()
        pair_weight = pair_weight / pair_weight.mean()
        wet = torch.ones((num_channels, 2, 2), dtype=torch.float32)
        delta = 0.25

        def loss_for_error_at(level: int) -> torch.Tensor:
            target = torch.zeros(1, num_channels, 2, 2)
            pred = target.clone()
            pred[:, theta[level]] = delta
            return gradient_z_l1_loss(pred, target, wet)

        # An interior level is touched by the pair above it and the pair below,
        # and its channel loss is the mean of the two.
        for level in (1, 25, 49):
            expected = 0.5 * delta * (pair_weight[level - 1] + pair_weight[level])
            torch.testing.assert_close(loss_for_error_at(level)[theta[level]], expected)

        # The surface and the floor each have only one neighbouring pair.
        torch.testing.assert_close(
            loss_for_error_at(0)[theta[0]], delta * pair_weight[0]
        )
        torch.testing.assert_close(
            loss_for_error_at(50)[theta[50]], delta * pair_weight[49]
        )

        # And the reweighting genuinely bites: the finest pair is charged ~42x
        # the coarsest for an identical jump, matching the spacing ratio.
        torch.testing.assert_close(
            loss_for_error_at(0)[theta[0]] / loss_for_error_at(50)[theta[50]],
            spacing[49] / spacing[0],
        )


def test_gradient_z_keeps_the_scale_of_the_unweighted_term():
    """The 1/dz weights average one, so an error that is uniform in the vertical
    derivative costs what it cost before any weighting went in. That is what lets
    an already-tuned `lambda_z` carry over instead of silently changing strength.
    """
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        num_channels = len(tensor_map.prognostic_var_names)
        theta = tensor_map.VAR_3D_IDX["Theta"].long()
        wet = torch.ones((num_channels, 2, 2), dtype=torch.float32)
        delta = 0.2

        # An error growing linearly in the level index puts the same jump,
        # delta, between every adjacent pair of levels.
        target = torch.zeros(1, num_channels, 2, 2)
        pred = target.clone()
        for level in range(theta.numel()):
            pred[:, theta[level]] = level * delta

        column_mean = gradient_z_l1_loss(pred, target, wet)[theta].mean()

        # Unweighted, every level would score exactly delta. The weighting shifts
        # the column mean only by the two edge levels' share, a few percent.
        assert column_mean == pytest.approx(delta, rel=0.05)


def test_gradient_z_is_scale_free_in_the_level_spacing():
    """Weighting by 1/dz rescaled to average one, rather than by 1/dz in raw
    metres, is what keeps `lambda_z` meaningful: stretching the whole depth axis
    must not change the penalty."""
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        num_channels = len(tensor_map.prognostic_var_names)
        torch.manual_seed(0)
        pred = torch.randn(2, num_channels, 3, 3)
        target = torch.randn(2, num_channels, 3, 3)
        wet = torch.ones((num_channels, 3, 3), dtype=torch.float32)

        original = gradient_z_l1_loss(pred, target, wet)
        tensor_map.channel_depth_centers = tensor_map.channel_depth_centers * 3.0
        rescaled = gradient_z_l1_loss(pred, target, wet)

        torch.testing.assert_close(original, rescaled)


def test_gradient_z_accepts_a_per_channel_sample_weight():
    """Regression: the per-sample weight is [batch, channel, lat, lon], not a
    surface field. Reshaping it as if it had one channel made every grouped
    training run die at the first optimizer step."""
    with MultitonScope():
        _, num_channels, pred, target, wet = _gradient_z_setup(batch=8)
        sample_weight = torch.ones(8, num_channels, 3, 3)

        loss = gradient_z_l1_loss(pred, target, wet, sample_weight=sample_weight)
        assert loss.shape == (num_channels,)
        assert torch.isfinite(loss).all()


def test_gradient_z_all_ones_sample_weight_matches_no_weight():
    """A weight of one everywhere must be a no-op, or the weighted form is not a
    generalization of the unweighted one."""
    with MultitonScope():
        _, num_channels, pred, target, wet = _gradient_z_setup(batch=4)
        unweighted = gradient_z_l1_loss(pred, target, wet)
        weighted = gradient_z_l1_loss(
            pred, target, wet, sample_weight=torch.ones(4, num_channels, 3, 3)
        )
        torch.testing.assert_close(unweighted, weighted)


def test_gradient_z_sample_weight_can_zero_out_a_sample():
    """Dropping one member of the batch must equal never having supplied it --
    this is what lets a tile be scored only on the cells it owns."""
    with MultitonScope():
        _, num_channels, pred, target, wet = _gradient_z_setup(batch=2)
        weight = torch.ones(2, num_channels, 3, 3)
        weight[1] = 0.0

        masked = gradient_z_l1_loss(pred, target, wet, sample_weight=weight)
        first_only = gradient_z_l1_loss(pred[:1], target[:1], wet)
        torch.testing.assert_close(masked, first_only)


def test_gradient_z_rejects_a_weight_with_the_wrong_channel_count():
    with MultitonScope():
        _, _, pred, target, wet = _gradient_z_setup(batch=2)
        with pytest.raises(ValueError, match="gradient_z needs"):
            gradient_z_l1_loss(pred, target, wet, sample_weight=torch.ones(2, 7, 3, 3))


def test_full_gradient_loss_runs_with_a_per_sample_weight():
    """The end-to-end path the training loop actually takes: metric plus both
    gradient terms, with a per-tile weight threaded through all three."""
    with MultitonScope():
        tensor_map, num_channels, pred, target, wet = _gradient_z_setup(batch=4)
        loss_fn = build_loss_fn(
            GradientLossConfig(
                type=["gradient_h", "gradient_z"],
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


def test_gradient_z_with_history_channels():
    """Channels arrive time-major as (hist+1) * var, and the wet mask is the
    per-variable mask concatenated once per time. Every history copy has to get
    its own vertical term rather than the reshape silently misaligning."""
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        num_vars = len(tensor_map.prognostic_var_names)
        hist = 1
        num_times = hist + 1
        torch.manual_seed(0)
        pred = torch.randn(2, num_times * num_vars, 3, 3)
        target = torch.randn(2, num_times * num_vars, 3, 3)
        wet = torch.ones((num_times * num_vars, 3, 3), dtype=torch.float32)

        loss = gradient_z_l1_loss(pred, target, wet)

        assert loss.shape == (num_times * num_vars,)
        # The same fields repeated as history must score identically.
        repeated = gradient_z_l1_loss(
            pred[:, :num_vars].repeat(1, num_times, 1, 1),
            target[:, :num_vars].repeat(1, num_times, 1, 1),
            wet,
        ).reshape(num_times, num_vars)
        torch.testing.assert_close(repeated[0], repeated[1])


def test_gradient_z_rejects_channels_that_are_not_a_multiple_of_the_var_count():
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        num_vars = len(tensor_map.prognostic_var_names)
        pred = torch.zeros(1, num_vars + 1, 2, 2)
        wet = torch.ones((num_vars + 1, 2, 2), dtype=torch.float32)
        with pytest.raises(ValueError, match="gradient_z expected"):
            gradient_z_l1_loss(pred, pred.clone(), wet)


def test_gradient_z_is_a_mean_not_a_batch_sum():
    """Regression: the denominator sums over the batch axis, so it must count
    every sample's cells. Broadcasting a size-1 axis there scaled the whole term
    by the batch size -- invisible at batch=1, which is all the older tests used.

    Duplicating a sample cannot change a mean.
    """
    with MultitonScope():
        _, _, pred, target, wet = _gradient_z_setup(batch=1)
        single = gradient_z_l1_loss(pred, target, wet)
        quadrupled = gradient_z_l1_loss(
            pred.repeat(4, 1, 1, 1), target.repeat(4, 1, 1, 1), wet
        )
        torch.testing.assert_close(single, quadrupled)


def test_gradient_z_ignores_dry_pairs():
    """A vertical difference spans two levels, so it is scored only where both
    are wet. A level that is land everywhere must not drag its neighbours."""
    with MultitonScope():
        tensor_map = TensorMap.init_instance("all", "all")
        num_channels = len(tensor_map.prognostic_var_names)
        theta = tensor_map.VAR_3D_IDX["Theta"].long()
        torch.manual_seed(0)
        pred = torch.randn(1, num_channels, 2, 2)
        target = torch.randn(1, num_channels, 2, 2)
        wet = torch.ones((num_channels, 2, 2), dtype=torch.float32)
        wet[theta[30:]] = 0.0

        loss = gradient_z_l1_loss(pred, target, wet)

        # Pairs (29, 30) onward are dry, so levels 30 and below score nothing
        # while the wet part of the column is untouched.
        torch.testing.assert_close(loss[theta[30:]], torch.zeros_like(loss[theta[30:]]))
        assert (loss[theta[:29]] > 0).all()
