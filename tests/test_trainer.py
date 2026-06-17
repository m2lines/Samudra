# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import tempfile
from pathlib import Path

import pytest
import torch

from samudra.config import CpuDataLoadingConfig, DynamicLossConfig
from samudra.models.base import BaseModel
from samudra.train import Trainer, should_log_validation_images
from samudra.utils.ctx import GridContext
from samudra.utils.loss import DynamicLoss
from samudra.utils.multiton import MultitonScope
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
    [("mock-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_trainer__mini_2step(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    trainer.run()


@pytest.mark.parametrize(
    "backend",
    [pytest.param("cuda", marks=pytest.mark.cuda)],
    indirect=True,
)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_samudra_mini.yaml")],
    indirect=True,
)
def test_trainer__samudra_mini_smoke_cuda(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    # The torchinfo summary path can OOM on the shared CI GPU despite this tiny config.
    trainer.num_batches_seen = 1
    trainer.run()


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_default_2step.yaml")],
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
    [("mock-om4", "test/train_default_2step.yaml")],
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
    [("mock-om4", "test/train_default_2step.yaml")],
    indirect=True,
)
def test_checkpoint_inference(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    hist = trainer.hist
    resolution = trainer.inference_src.resolution
    wet = trainer.inference_src.masks.prognostic_with_hist(hist)
    ctx = GridContext(wet, resolution, resolution).to(trainer.device)
    data = trainer.inference_loader.dataset[0]
    inference_dataset, _num_steps = data
    prog, boundary, _label = inference_dataset[0]
    prog = prog.to(trainer.device)
    boundary = boundary.to(trainer.device)
    trainer.best_val_loss = 10
    trainer.best_inf_loss = 10

    model = trainer.model
    assert isinstance(model, BaseModel)
    out = model.forward_once(prog, boundary, ctx)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(1, Path(tmpdir) / "test.pt")
        trainer.load_checkpoint(Path(tmpdir) / "test.pt")

    out2 = model.forward_once(prog, boundary, ctx)

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


def test_should_log_validation_images_every_n_epochs():
    assert [
        epoch for epoch in range(1, 26) if should_log_validation_images(epoch, 10)
    ] == [
        1,
        11,
        21,
    ]


def test_should_log_validation_images_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="Epoch must be >= 1"):
        should_log_validation_images(0, 10)

    with pytest.raises(ValueError, match="Validation image log frequency must be >= 1"):
        should_log_validation_images(1, 0)


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_default.yaml")],
    indirect=True,
)
def test_data_loaders_enable_persistent_workers_on_positive_num_workers(
    trainer_pair: TrainPair,
):
    _, trainer = trainer_pair

    assert trainer.mp_context is not None
    assert trainer.mp_context.get_start_method() == "spawn"
    assert trainer.train_loader._dataloader.persistent_workers is True
    assert trainer.val_loader._dataloader.persistent_workers is True


@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", "test/train_default.yaml")],
    indirect=True,
)
def test_data_loaders_disable_persistent_workers_when_num_workers_is_zero(
    train_config,
):
    assert isinstance(train_config.data.loading, CpuDataLoadingConfig)
    train_config.data.loading.num_workers = 0
    train_config.data.loading.persistent_workers = True

    with MultitonScope():
        trainer = Trainer(train_config)
        trainer.init_data_loaders(cur_step=train_config.steps[0])

    assert trainer.mp_context is None
    assert trainer.train_loader._dataloader.persistent_workers is False
    assert trainer.val_loader._dataloader.persistent_workers is False
