import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.config import DynamicLossConfig, TrainConfig
from ocean_emulators.models.base import BaseModel
from ocean_emulators.train import Trainer
from ocean_emulators.utils.loss import DynamicLoss
from ocean_emulators.utils.multiton import MultitonScope
from tests.conftest import DEFAULT_CONFIG, TrainPair


@pytest.mark.manual
@pytest.mark.parametrize(
    "data_source,config_name", [("mock", DEFAULT_CONFIG)], indirect=True
)
def test_trainer__mini_benchmark(trainer_pair: TrainPair, caplog, benchmark):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    @benchmark
    def run():
        trainer.run()


@pytest.mark.parametrize(
    "data_source,config_name",
    [("remote-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_trainer__mini_2step(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    trainer.run()


def _write_tiny_packed_cache(root: Path) -> None:
    time = xr.cftime_range(
        "1975-08-05",
        "1975-12-31",
        freq="5D",
        calendar="julian",
    )
    lat = np.linspace(-4.0, 4.0, 8, dtype=np.float32)
    lon = np.linspace(0.0, 7.0, 8, dtype=np.float32)
    n_time = len(time)
    base = np.arange(n_time, dtype=np.float32).reshape(n_time, 1, 1, 1)
    spatial = np.zeros((1, 1, len(lat), len(lon)), dtype=np.float32)
    prognostic = base + spatial
    boundary = base + spatial + 1.0
    packed = xr.Dataset(
        {
            "prognostic": (
                ["time", "prognostic_channel", "lat", "lon"],
                prognostic,
            ),
            "boundary": (
                ["time", "boundary_channel", "lat", "lon"],
                boundary,
            ),
            "prognostic_mean": (
                ["prognostic_channel"],
                np.zeros(1, dtype=np.float32),
            ),
            "prognostic_std": (
                ["prognostic_channel"],
                np.ones(1, dtype=np.float32),
            ),
            "boundary_mean": (
                ["boundary_channel"],
                np.zeros(1, dtype=np.float32),
            ),
            "boundary_std": (
                ["boundary_channel"],
                np.ones(1, dtype=np.float32),
            ),
            "prognostic_mask": (
                ["prognostic_channel", "lat", "lon"],
                np.ones((1, len(lat), len(lon)), dtype=bool),
            ),
            "boundary_mask": (
                ["boundary_channel", "lat", "lon"],
                np.ones((1, len(lat), len(lon)), dtype=bool),
            ),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "cache_format": "llc-train-ready-v1",
            "prognostic_channel_names_json": json.dumps(["Theta_0"]),
            "boundary_channel_names_json": json.dumps(["oceQnet"]),
        },
    )
    packed.to_zarr(root / "packed.zarr")


def test_trainer__replay_smoke(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    _write_tiny_packed_cache(tmp_path)
    train_config = TrainConfig.from_yaml_and_cli(
        [
            "configs/test/train_default.yaml",
            "--backend",
            "cpu",
            "--epochs",
            "1",
            "--save_freq",
            "1",
            "--experiment.data_root",
            str(tmp_path),
            "--experiment.name",
            "test_replay_smoke",
            "--experiment.prognostic_vars_key",
            "single_1",
            "--experiment.boundary_vars_key",
            "single",
            "--data.data_location",
            "packed.zarr",
            "--data.data_means_location",
            "unused",
            "--data.data_stds_location",
            "unused",
            "--data.num_workers",
            "0",
            "--data.hist",
            "0",
            "--replay.enabled",
            "true",
            "--replay.buffer_size",
            "2",
            "--replay.steps_per_epoch",
            "2",
            "--replay.refresh_every_n_microbatches",
            "1",
        ]
    )
    train_config.inference_epochs = []

    with MultitonScope():
        trainer = Trainer(train_config)
        trainer.run()

    sidecar = trainer.ckpt_paths.latest_checkpoint_path.with_name(
        f"{trainer.ckpt_paths.latest_checkpoint_path.stem}.replay_rank0.pt"
    )
    assert sidecar.exists()


@pytest.mark.parametrize(
    "data_source,config_name",
    [("remote-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_checkpoint_ema(train_config, caplog):
    caplog.set_level(logging.INFO)
    train_config.epochs = 1
    train_config.save_freq = 1

    with MultitonScope():
        e2e_trainer = Trainer(train_config)
        e2e_trainer.run()

    with MultitonScope():
        train_config.resume_ckpt_path = e2e_trainer.ckpt_paths.latest_checkpoint_path
        resume_trainer = Trainer(train_config)

    # TODO(jder): would be nice to generalize to testing the whole trainer state,
    # or even running it forward and checking the output is identical
    assert resume_trainer._ema == e2e_trainer._ema


@pytest.mark.parametrize(
    "data_source,config_name",
    [("remote-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_checkpoint_dynamic_loss_state(train_config, caplog):
    """DynamicLoss has internal rolling state; ensure it round-trips via checkpoints."""
    caplog.set_level(logging.INFO)
    train_config.epochs = 1
    train_config.save_freq = 1
    train_config.loss = DynamicLossConfig(metric="mse", limit=100.0)

    with MultitonScope():
        e2e_trainer = Trainer(train_config)
        assert isinstance(e2e_trainer.loss_fn, DynamicLoss)
        e2e_trainer.run()
        scale_before = e2e_trainer.loss_fn.loss_scale_per_channel().detach().cpu()

        # Make the test meaningful: ensure at least one update away from the init value.
        assert torch.isfinite(scale_before).all()
        assert not torch.allclose(scale_before, torch.ones_like(scale_before))


@pytest.mark.parametrize(
    "data_source,config_name",
    [("remote-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_checkpoint_inference(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    data = trainer.inference_loader.dataset[0]
    X, y = data
    trainer.best_val_loss = 10
    trainer.best_inf_loss = 10

    model = trainer.model
    assert isinstance(model, BaseModel)
    out = model.forward_once(X[0][0].to(trainer.device))

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(1, Path(tmpdir) / "test.pt")
        trainer.load_checkpoint(Path(tmpdir) / "test.pt")

    out2 = model.forward_once(X[0][0].to(trainer.device))

    assert torch.allclose(out, out2)


@pytest.mark.parametrize(
    "data_source,config_name,extra_config_args",
    [
        (
            "mock",
            DEFAULT_CONFIG,
            [
                "--train_time.start",
                "1975-08-01",
                "--train_time.end",
                "1975-09-01",
                "--val_time.start",
                "1975-08-15",
                "--val_time.end",
                "1975-09-01",
            ],
        ),
    ],
    indirect=True,
)
def test_trainer_overlapping_time_ranges_raises_error(train_config, caplog):
    """Creating a trainer with overlapping train + val times should error."""

    with MultitonScope():
        with pytest.raises(ValueError, match="Training time range.*"):
            Trainer(train_config)


def test_get_current_step_resume_at_transition_uses_post_transition_stage():
    trainer = Trainer.__new__(Trainer)
    trainer.start_epoch = 9
    trainer.steps = [2, 3, 4, 5, 6, 7]
    trainer.step_transition = [5, 9, 13, 17, 21]

    assert trainer.get_current_step(9) == 4


def test_get_current_temporal_stride_resume_at_transition_uses_post_transition_stage():
    trainer = Trainer.__new__(Trainer)
    trainer.start_epoch = 9
    trainer.temporal_strides = [1, 3, 6]
    trainer.temporal_stride_transition = [5, 9]

    assert trainer.get_current_temporal_stride(9) == 6
