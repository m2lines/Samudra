import logging

import pytest

from config import TrainConfig
from train_3D import Trainer


@pytest.mark.manual
def test_trainer__run(train_config: TrainConfig, caplog, benchmark):
    caplog.set_level(logging.INFO)
    trainer = Trainer(train_config)

    @benchmark
    def run():
        trainer.run()
