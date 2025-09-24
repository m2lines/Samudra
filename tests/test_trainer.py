import logging
import tempfile
from pathlib import Path

import pytest
import torch

from ocean_emulators.models.base import BaseModel
from ocean_emulators.train import Trainer
from ocean_emulators.utils.multiton import MultitonScope
from tests.conftest import DEFAULT_CONFIG, FOMO_MINIMAL_CONFIG, TrainPair


@pytest.mark.manual
@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock", DEFAULT_CONFIG), ("mock", FOMO_MINIMAL_CONFIG)],
    indirect=True,
)
def test_trainer__mini_benchmark(trainer_pair: TrainPair, caplog, benchmark):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    @benchmark
    def run():
        trainer.run()


@pytest.mark.parametrize(
    "data_source,config_name",
    [("remote-om4", "train_default_2step.test.yaml")],
    indirect=True,
)
def test_trainer__mini_2step(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    trainer.run()


@pytest.mark.parametrize(
    "data_source,config_name",
    [("remote-om4", "train_default_2step.test.yaml")],
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
    [("remote-om4", "train_default_2step.test.yaml")],
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
