"""Test core Datasets and DataLoaders."""

from typing import Tuple

import numpy as np
import pytest

from constants import EXTRA_VARS, INPT_VARS, OUT_VARS
from datasets import TrainData
from train_3D import Trainer

# Note: Refactoring data loaders is planned for the near-term. Ideally,
# these fixtures allow us to isolate data loader tests from their setup.


# This micro-fixture is cached by pytest. Thus, we don't have to change
# the factory methods that throw errors during double initialization.
@pytest.fixture(scope="session")
def trainer_pair(train_config):
    trainer = Trainer(train_config)

    # cur_step will set the number of pairs in the input/output sample
    trainer.init_data_loaders(cur_step=train_config.steps[0])

    return train_config, trainer


@pytest.fixture
def train_loader_pair(trainer_pair):
    cfg, trainer = trainer_pair
    return cfg, trainer.train_loader


@pytest.fixture
def val_loader_pair(trainer_pair):
    cfg, trainer = trainer_pair
    return cfg, trainer.val_loader


def extract_sample_arrays(td: TrainData, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract underlying X, y pairs from TrainData object."""
    x_arrays = [td.get_input(s).numpy(force=True) for s in range(steps)]
    y_arrays = [td.get_label(s).numpy(force=True) for s in range(steps)]

    return np.stack(x_arrays, axis=0), np.stack(y_arrays, axis=0)


# TODO(alxmrs): How can we determine `n_samples` from the input config? Timeslice?
def test_train__loads_correct_number_of_samples(train_loader_pair):
    cfg, loader = train_loader_pair
    n_samples = 6
    assert (
        len(list(loader)) == n_samples
    ), f"Current config {cfg} only supports {n_samples} examples; got {len(loader)}."


# TODO(alxmrs): What does the "2" dimension represent? Where is it in the config?
def test_train__data_shape(train_loader_pair):
    cfg, loader = train_loader_pair

    exp = cfg.experiment

    input_var_dim = len(INPT_VARS[exp.exp_num_in]) + len(EXTRA_VARS[exp.exp_num_extra])
    output_var_dim = len(OUT_VARS[cfg.experiment.exp_num_out])

    for sample in loader:
        X, y = extract_sample_arrays(sample, cfg.steps[0])
        assert X.shape == (cfg.steps[0], 2, input_var_dim, 180, 360)
        assert y.shape == (cfg.steps[0], 2, output_var_dim, 180, 360)


# TODO(alxmrs): How can we determine `n_samples` from the input config? Timeslice?
def test_val__loads_correct_number_of_samples(val_loader_pair):
    cfg, loader = val_loader_pair
    n_samples = 2
    assert (
        len(list(loader)) == n_samples
    ), f"Current config {cfg} only supports {n_samples} examples; got {len(loader)}."


# TODO(alxmrs): What does the "2" dimension represent? Where is it in the config?
def test_val__data_shape(val_loader_pair):
    cfg, loader = val_loader_pair

    exp = cfg.experiment

    input_var_dim = len(INPT_VARS[exp.exp_num_in]) + len(EXTRA_VARS[exp.exp_num_extra])
    output_var_dim = len(OUT_VARS[cfg.experiment.exp_num_out])

    for sample in loader:
        X, y = extract_sample_arrays(sample, 1)
        assert X.shape == (1, 2, input_var_dim, 180, 360)
        assert y.shape == (1, 2, output_var_dim, 180, 360)
