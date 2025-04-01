import logging

import pytest

from ocean_emulators.config import TrainConfig
from tests.test_datasets import TrainPair


@pytest.fixture(autouse=True, scope="function")
def set_scope(train_config: TrainConfig):
    """Automatically sets up the correct Multiton scope for each test.

    NB you must still do this manually for session-scoped fixtures.
    """
    with getattr(train_config, "_multiton_scope"):
        yield


@pytest.mark.manual
def test_trainer__mini_benchmark(trainer_pair: TrainPair, caplog, benchmark):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    @benchmark
    def run():
        trainer.run()


@pytest.mark.only_configs("train_cm4_2step.test.yaml")
def test_trainer__mini_2step(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    trainer.run()
