# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Residual units, autoregressive history, and checkpoint compatibility."""

import io

import numpy as np
import pytest
import torch
import xarray as xr

from samudra.config import SamudraConfig, TrainConfig
from samudra.datasets import ModelBatch
from samudra.models.base import BaseModel
from samudra.stepper import validate_batch
from samudra.utils.ctx import BatchGrid
from samudra.utils.location import LocalLocation, UnresolvedLocation
from samudra.utils.loss import TendencyLoss, loss_fn_from_metric, tendency_loss_weights


class UnitChangeModel(BaseModel):
    def forward_once(self, prognostic, boundary, ctx):
        return torch.ones_like(prognostic[:, -self.out_channels :])


def model(channels=2, output_steps=1, residuals=True):
    return UnitChangeModel(
        in_channels=2 * channels,
        out_channels=output_steps * channels,
        input_steps=2,
        pred_residuals=residuals,
        last_kernel_size=3,
        pad="circular",
        gradient_detach_interval=0,
    )


def test_scaled_residuals_use_latest_state_during_training_and_validation():
    net = model()
    net.set_residual_scale(torch.tensor([0.5, 0.125]))
    ctx = BatchGrid(
        torch.ones((2, 1, 1), dtype=torch.bool),
        (torch.zeros(1),) * 2,
        (torch.zeros(1),) * 2,
    )
    initial = torch.tensor([1.0, 2.0, 10.0, 20.0]).reshape(1, 4, 1, 1)
    batch = ModelBatch(ctx)
    for target in ([10.5, 20.125], [11.0, 20.25]):
        batch.append(
            initial, torch.zeros_like(initial), torch.tensor(target).reshape(1, 2, 1, 1)
        )
    outputs = net(batch)
    torch.testing.assert_close(outputs[0].flatten(), torch.tensor([10.5, 20.125]))
    torch.testing.assert_close(outputs[1].flatten(), torch.tensor([11.0, 20.25]))
    first = ModelBatch(ctx)
    first.append(initial, torch.zeros_like(initial), outputs[0])
    validation = validate_batch(net, first, loss_fn_from_metric("mse"))
    torch.testing.assert_close(validation.gen_data, outputs[0])
    assert validation.loss.item() == 0


@pytest.mark.parametrize("output_steps", [1, 2])
def test_config_converts_physical_scales_and_repeats_time_major(tmp_path, output_steps):
    xr.Dataset({"uo_0": xr.DataArray(0.2), "thetao_0": xr.DataArray(0.005)}).to_zarr(
        tmp_path / "rms.zarr"
    )
    cfg = SamudraConfig(
        pred_residuals=True, residual_scale_location=UnresolvedLocation(path="rms.zarr")
    )
    net = model(output_steps=output_steps)
    cfg.configure_residual_scaling(
        net,
        data_root=LocalLocation(path=tmp_path),
        channels=("uo_0", "thetao_0"),
        state_stds=np.array([0.4, 0.5]),
    )
    torch.testing.assert_close(
        net.residual_scale, torch.tensor([0.5, 0.01] * output_steps)
    )
    # Unit changes from the decoder are interpreted in each channel's own units.
    prediction = net._assemble_prediction(
        torch.zeros(1, 4, 1, 1), torch.ones(1, 2 * output_steps, 1, 1)
    )
    torch.testing.assert_close(prediction.flatten(), net.residual_scale)


def test_scaled_tendency_gradient_matches_normalized_residual_mse():
    # The uncapped objective agrees up to its single weight-normalization
    # constant, including gradients through the output parameterization.
    net = model()
    scale = torch.tensor([0.5, 0.0004])
    net.set_residual_scale(scale)
    raw = (
        torch.tensor([0.3, 0.7], dtype=torch.float64)
        .reshape(1, 2, 1, 1)
        .requires_grad_()
    )
    state = torch.tensor([1.0, 2.0, 10.0, 20.0], dtype=torch.float64).reshape(
        1, 4, 1, 1
    )
    target = state[:, -2:] + scale.view(1, 2, 1, 1) * 0.9
    ctx = BatchGrid(
        torch.ones((2, 1, 1), dtype=torch.bool),
        (torch.zeros(1),) * 2,
        (torch.zeros(1),) * 2,
    )
    loss = TendencyLoss(
        loss_fn_from_metric("mse"),
        weights=tendency_loss_weights(scale, max_weight_ratio=None),
        device=torch.device("cpu"),
    )
    actual = loss(net._assemble_prediction(state, raw), target, ctx).mean()
    actual_grad = torch.autograd.grad(actual, raw)[0]
    reference = ((raw - 0.9) ** 2).mean() / scale.double().square().reciprocal().mean()
    expected_grad = torch.autograd.grad(reference, raw)[0]
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-15)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-15)


def test_checkpoint_round_trip_restores_scaling_and_legacy_keys_stay_unchanged():
    legacy = model()
    assert "residual_scale" not in legacy.state_dict()
    model().load_state_dict(legacy.state_dict(), strict=True)
    scaled = model()
    scaled.set_residual_scale(torch.tensor([0.5, 0.0004]))
    checkpoint = io.BytesIO()
    torch.save(scaled.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = model()
    restored.set_residual_scale(torch.ones(2))
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    torch.testing.assert_close(restored.residual_scale, scaled.residual_scale)
    assert not list(
        restored.parameters()
    )  # Fixed scales are buffers, not trained parameters.


@pytest.mark.parametrize(
    "scale",
    [
        torch.ones(1),
        torch.ones(1, 2),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, float("nan")]),
        torch.tensor([1.0, -1.0]),
    ],
)
def test_invalid_scales_fail_loudly(scale):
    with pytest.raises(ValueError):
        model().set_residual_scale(scale)


def test_scaling_requires_residual_prediction():
    with pytest.raises(ValueError, match="requires pred_residuals"):
        model(residuals=False).set_residual_scale(torch.ones(2))
    with pytest.raises(ValueError, match="requires pred_residuals"):
        SamudraConfig(residual_scale_location=UnresolvedLocation(path="rms.zarr"))


def test_scaled_model_rejects_multiple_training_normalizations():
    cfg = TrainConfig.from_yaml("samudra_om4/train.yaml").model_dump(mode="json")
    cfg["model"]["pred_residuals"] = True
    cfg["model"]["residual_scale_location"] = "rms.zarr"
    cfg["data"]["sources"] *= 2
    with pytest.raises(ValueError, match="single data source"):
        TrainConfig.model_validate(cfg)
