# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""The normalized_delta loss and scaled residual head.

The network emits a normalized delta ``d`` and the state is ``x[t] + q * d``,
with ``q = tau / sigma`` per channel. ``tau`` is a normalization statistic of
the data source, like the means and stds. The reference objective, written out
in ``reference_loss`` below, is an unweighted masked MSE between ``d`` and
``(label - x[t]) / q``.
"""

import pathlib

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import xarray as xr
import yaml
from pydantic import ValidationError

from samudra.config import TrainConfig
from samudra.models.base import BaseModel
from samudra.train import Trainer
from samudra.utils.ctx import BatchGrid
from samudra.utils.location import LocalLocation, sibling_location
from samudra.utils.loss import NormalizedDeltaLoss
from samudra.utils.multiton import MultitonScope

TEST_CONFIGS_DIR = pathlib.Path(__file__).parent / "configs"
B, H, W = 3, 6, 8
N = 3  # channels
TRAINER_CASE = pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "train_default_2step.yaml")],
    indirect=True,
)
CPU_BACKEND = pytest.mark.parametrize("backend", ["cpu"], indirect=True)


def residual_model(q: torch.Tensor) -> BaseModel:
    return BaseModel(
        in_channels=2 * q.numel(),
        out_channels=q.numel(),
        input_steps=2,
        pred_residuals=True,
        last_kernel_size=3,
        pad="circular",
        gradient_detach_interval=0,
        residual_scale=q,
    )


def reference_loss(anchor, d, label, wet, q):
    """Masked MSE between d and the true normalized delta, one head."""
    q = q.view(1, -1, 1, 1)
    target = torch.where(wet, (label - anchor) / q, 0.0)
    return F.mse_loss(d * wet, target * wet, reduction="none").mean(dim=(0, 2, 3))


def test_matches_the_reference_normalized_delta_objective():
    q = torch.tensor([0.4, 0.026, 0.0016])  # velocity-, SST- and deep-like
    loss_fn = NormalizedDeltaLoss(increment_rms=q, device=torch.device("cpu"))
    model = residual_model(q)

    g = torch.Generator().manual_seed(0)
    wet = torch.rand(N, H, W, generator=g) > 0.3  # some land
    prev, anchor, label = (torch.randn(B, N, H, W, generator=g) * wet for _ in range(3))
    raw = torch.randn(B, N, H, W, generator=g)

    lat, lon = torch.linspace(-80, 80, H), torch.linspace(0, 359, W)
    d_ours = raw.clone().requires_grad_()
    # forward_once zeroes land before assembly.
    pred = model._assemble_prediction(
        torch.cat((prev, anchor), dim=1), torch.where(wet, d_ours, 0.0)
    )
    ours = loss_fn(pred, label, BatchGrid(wet, (lat, lon), (lat, lon)))
    ours.sum().backward()

    d_ref = raw.clone().requires_grad_()
    ref = reference_loss(anchor, d_ref, label, wet, q)
    ref.sum().backward()

    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(d_ours.grad, d_ref.grad, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("bad", [0.0, float("nan")])
def test_zero_or_nan_increment_rms_fails_loudly(bad):
    q = torch.tensor([0.4, bad, 0.0016])
    with pytest.raises(ValueError, match="positive, finite"):
        NormalizedDeltaLoss(increment_rms=q, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="positive and finite"):
        residual_model(q)


def test_residual_scale_is_a_data_statistic_not_checkpoint_state():
    # Like the means and stds, the scale comes from the data source when the
    # model is built, so a checkpoint never carries a stale copy of it.
    model = residual_model(torch.tensor([0.4, 0.026, 0.0016]))
    assert "residual_scale" not in model.state_dict()


def train_yaml(**changes) -> dict:
    with open(TEST_CONFIGS_DIR / "train_default_2step.yaml") as f:
        raw = yaml.safe_load(f)
    raw["data"].update(input_steps=2, output_steps=1)
    raw["model"].update(pred_residuals=True, scale_residuals=True)
    raw["loss"] = {"type": "normalized_delta"}
    raw["data"]["sources"][0]["data_increment_rms_location"] = "rms.zarr"
    for key, value in changes.items():
        section, field = key.split("__")
        if section == "source":
            raw["data"]["sources"][0][field] = value
        else:
            raw[section][field] = value
    return raw


def test_config_rejects_normalized_delta_with_two_output_steps():
    TrainConfig.model_validate(train_yaml())
    with pytest.raises(ValidationError, match="output_steps == 1"):
        TrainConfig.model_validate(train_yaml(data__output_steps=2))


def test_increment_rms_is_read_by_convention_only_when_needed():
    needed = TrainConfig.model_validate(
        train_yaml(source__data_increment_rms_location=None)
    ).data.sources[0]
    assert needed.data_increment_rms_location == sibling_location(
        needed.data_location, "increment_rms"
    )
    # A config that does not use it never opens a store that may not exist.
    raw = train_yaml(source__data_increment_rms_location=None)
    raw["model"].update(scale_residuals=False)
    raw["loss"] = "mse"
    unneeded = TrainConfig.model_validate(raw).data.sources[0]
    assert "increment_rms" not in unneeded.statistic_locations()


def write_increment_rms(path, channels, tau, over="wet") -> None:
    stats = xr.Dataset({c: xr.DataArray(t) for c, t in zip(channels, tau)})
    if over is not None:
        stats.attrs["over"] = over
    stats.to_zarr(path)


def normalized_delta_config(train_config, tmp_path, *, over="wet", drop=0):
    """Switch a 2-in-1-out test config to normalized_delta with known tau."""
    train_config.data.input_steps = 2
    train_config.data.output_steps = 1
    train_config.data.loading.num_workers = 0
    with MultitonScope():
        probe = Trainer(train_config)
        channels = probe.prognostic_var_names
        stds = probe.primary_source.statistics(channels).std
    tau = np.linspace(0.01, 0.1, len(channels))
    # Written outside the shared test-data cache that serves as data_root.
    kept = len(channels) - drop
    write_increment_rms(tmp_path / "rms.zarr", channels[:kept], tau[:kept], over)
    source = train_config.data.sources[0]
    source.data_increment_rms_location = LocalLocation(path=tmp_path / "rms.zarr")
    train_config.model.pred_residuals = True
    train_config.model.scale_residuals = True
    train_config.loss = TrainConfig.model_validate(train_yaml()).loss
    return channels, torch.tensor(tau / stds, dtype=torch.float32)


@CPU_BACKEND
@TRAINER_CASE
def test_trainer_takes_one_scale_from_the_source_and_trains_2in1out(
    train_config, tmp_path
):
    channels, q = normalized_delta_config(train_config, tmp_path)
    with MultitonScope():
        trainer = Trainer(train_config)
        torch.testing.assert_close(trainer.model.residual_scale, q)
        assert isinstance(trainer.loss_fn, NormalizedDeltaLoss)
        torch.testing.assert_close(
            trainer.loss_fn.loss_scale_per_channel(),
            q.double().square().reciprocal().float(),
        )
        trainer.init_data_loaders(cur_step=train_config.steps[0])
        logs = trainer.train_one_epoch(epoch=1)

    channel_losses = [v for k, v in logs.items() if k.startswith("train/loss/channel/")]
    assert len(channel_losses) == len(channels)
    assert np.isfinite(np.array(channel_losses, dtype=float)).all()


@CPU_BACKEND
@TRAINER_CASE
@pytest.mark.parametrize("over", ["full", None])
def test_increment_rms_not_over_wet_cells_fails_loudly(train_config, tmp_path, over):
    # A full-grid RMS (land counted as zero) is smaller by sqrt(wet fraction)
    # per level; a run given one would train normally with wrong weights.
    normalized_delta_config(train_config, tmp_path, over=over)
    with MultitonScope(), pytest.raises(ValueError, match="over='wet'"):
        Trainer(train_config)


@CPU_BACKEND
@TRAINER_CASE
def test_increment_rms_missing_a_channel_fails_loudly(train_config, tmp_path):
    normalized_delta_config(train_config, tmp_path, drop=1)
    with MultitonScope(), pytest.raises(ValueError, match="No increment RMS"):
        Trainer(train_config)
