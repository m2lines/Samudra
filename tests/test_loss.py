import numpy as np
import torch
import xarray as xr

from ocean_emulators.config import GradientLossConfig, build_loss_fn
from ocean_emulators.utils.loss import decomposed_mse_mae, gradient_l1_loss


def test_gradient_loss_supports_mse_mae_metric():
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
    alpha = 0.25

    loss_fn = build_loss_fn(
        GradientLossConfig(metric="mse_mae", alpha=alpha),
        wet=wet,
        y_coord=y_coord,
        device=torch.device("cpu"),
        num_channels=1,
        pad_mode="constant",
    )

    actual = loss_fn(pred, target)
    expected = decomposed_mse_mae(pred, target, wet) + alpha * gradient_l1_loss(
        pred, target, wet, pad_mode="constant"
    )

    assert torch.allclose(actual, expected)
