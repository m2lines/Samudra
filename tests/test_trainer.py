import logging

import pytest

from tests.test_datasets import TrainPair


@pytest.mark.manual
@pytest.mark.parametrize("data_source", ["mock"], indirect=True)
def test_trainer__mini_benchmark(trainer_pair: TrainPair, caplog, benchmark):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    @benchmark
    def run():
        trainer.run()


@pytest.mark.only_configs("train_default_2step.test.yaml")
def test_trainer__mini_2step(trainer_pair: TrainPair, caplog):
    caplog.set_level(logging.INFO)
    _, trainer = trainer_pair

    trainer.run()
