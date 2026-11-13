# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""The normalized_delta loss and scaled residual head.

Reference: Yuan's ``normalized_delta`` run (Ocean_Emulator,
``revision_experiments/normalized_delta``). The network emits ``d`` and the
state is ``x[t] + q * d``, with ``q = tau / sigma`` per channel. Her loss is an
unweighted masked MSE between ``d`` and ``(label - x[t]) / q``.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import xarray as xr

from samudra.config import NormalizedDeltaLossConfig, build_loss_fn
from samudra.models.base import BaseModel
from samudra.train import Trainer
from samudra.utils.ctx import BatchGrid
from samudra.utils.location import LocalLocation, UnresolvedLocation
from samudra.utils.multiton import MultitonScope
from scripts.compute_tendency_stats import persistence_rms

B, H, W = 3, 6, 8
CHANNELS = ("uo_0", "thetao_0", "thetao_18")
TAU = np.array([0.05, 0.3, 8e-4])  # physical RMS change over wet cells
SIGMA = np.array([0.13, 11.6, 0.5])  # state normalization stds


def residual_model(n: int) -> BaseModel:
    return BaseModel(
        in_channels=2 * n,
        out_channels=n,
        input_steps=2,
        pred_residuals=True,
        last_kernel_size=3,
        pad="circular",
        gradient_detach_interval=0,
        scale_residuals=True,
    )


def yuan_loss(anchor, d, label, wet, q):
    """Her NormalizedDeltaTarget + LossFnWithMask("mse"), one output head."""
    q = q.view(1, -1, 1, 1)
    target = torch.where(wet, (label - anchor) / q, 0.0)
    return F.mse_loss(d * wet, target * wet, reduction="none").mean(dim=(0, 2, 3))


def test_matches_yuans_normalized_delta_objective(tmp_path):
    xr.Dataset({c: xr.DataArray(t) for c, t in zip(CHANNELS, TAU)}).to_zarr(
        tmp_path / "tau.zarr"
    )
    loss_fn = build_loss_fn(
        NormalizedDeltaLossConfig(
            increment_rms_location=UnresolvedLocation(path="tau.zarr")
        ),
        device=torch.device("cpu"),
        num_channels=len(CHANNELS),
        pad_mode="circular",
        data_root=LocalLocation(path=tmp_path),
        channels=CHANNELS,
        state_stds=SIGMA,
    )
    q = torch.tensor(TAU / SIGMA, dtype=torch.float32)
    model = residual_model(len(CHANNELS))
    model.set_residual_scale(q)

    g = torch.Generator().manual_seed(0)
    n = len(CHANNELS)
    wet = torch.rand(n, H, W, generator=g) > 0.3  # some land
    prev, anchor, label = (torch.randn(B, n, H, W, generator=g) * wet for _ in range(3))
    raw = torch.randn(B, n, H, W, generator=g)

    lat, lon = torch.linspace(-80, 80, H), torch.linspace(0, 359, W)
    d_ours = raw.clone().requires_grad_()
    # forward_once zeroes land before assembly.
    pred = model._assemble_prediction(
        torch.cat((prev, anchor), dim=1), torch.where(wet, d_ours, 0.0)
    )
    ours = loss_fn(pred, label, BatchGrid(wet, (lat, lon), (lat, lon)))
    ours.sum().backward()

    d_ref = raw.clone().requires_grad_()
    ref = yuan_loss(anchor, d_ref, label, wet, q)
    ref.sum().backward()

    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(d_ours.grad, d_ref.grad, rtol=1e-5, atol=1e-9)


def test_residual_scale_restores_from_checkpoint_and_is_nan_until_set():
    q = torch.tensor([0.4, 0.02, 0.001])
    trained = residual_model(3)
    assert torch.isnan(trained.residual_scale).all()
    trained.set_residual_scale(q)

    evaluated = residual_model(3)
    evaluated.load_state_dict(trained.state_dict())
    torch.testing.assert_close(evaluated.residual_scale, q)


def test_persistence_rms_over_wet_cells_ignores_land():
    time = xr.date_range(
        "2000-01-01", periods=3, freq="5D", calendar="noleap", use_cftime=True
    )
    deep = np.zeros((3, 2, 2))
    deep[1:, 0, 0] = 4.0  # its only wet cell changes by 4 at one origin
    data = xr.Dataset(
        {
            "thetao_1": (("time", "y", "x"), deep),
            "mask_1": (("y", "x"), np.array([[True, False], [False, False]])),
        },
        coords={"time": time},
    )
    stats = persistence_rms(data, ["thetao_1"], time_stride=1, over="wet")
    # Two origins x one wet cell.
    assert float(stats["thetao_1"]) == pytest.approx(np.sqrt(16 / 2))


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_default_2step.yaml")],
    indirect=True,
)
def test_trainer_fills_residual_scale_and_trains_2in1out(train_config, tmp_path):
    train_config.data.input_steps = 2
    train_config.data.output_steps = 1
    train_config.data.loading.num_workers = 0
    with MultitonScope():
        probe = Trainer(train_config)
        channels = probe.prognostic_var_names
        stds = probe.primary_source.statistics(channels).std
    train_config.model.pred_residuals = True
    train_config.model.scale_residuals = True
    tau = np.linspace(0.01, 0.1, len(channels))
    xr.Dataset({c: xr.DataArray(t) for c, t in zip(channels, tau)}).to_zarr(
        tmp_path / "tau.zarr"
    )
    train_config.loss = NormalizedDeltaLossConfig(
        increment_rms_location=LocalLocation(path=tmp_path / "tau.zarr")
    )

    with MultitonScope():
        trainer = Trainer(train_config)
        torch.testing.assert_close(
            trainer.model.residual_scale,
            torch.tensor(tau / stds, dtype=torch.float32),
        )
        trainer.init_data_loaders(cur_step=train_config.steps[0])
        logs = trainer.train_one_epoch(epoch=1)

    channel_losses = [v for k, v in logs.items() if k.startswith("train/loss/channel/")]
    assert len(channel_losses) == len(channels)
    assert np.isfinite(np.array(channel_losses, dtype=float)).all()
