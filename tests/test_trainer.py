import logging
import os
import tempfile

import pytest
import torch

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
def test_checkpoint(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    data = trainer.inference_loader.dataset[0]
    X, y = data
    trainer.best_val_loss = 10
    trainer.best_inf_loss = 10

    model = trainer.model
    out = model.forward_once(X[0][0].to(trainer.device))

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(1, os.path.join(tmpdir, "test.pt"))
        trainer.load_checkpoint(os.path.join(tmpdir, "test.pt"))

    out2 = trainer.model.forward_once(X[0][0].to(trainer.device))

    assert torch.allclose(out, out2)
