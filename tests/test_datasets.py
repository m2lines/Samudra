"""Test core Datasets and DataLoaders."""

import numpy as np
import pytest
from torch.utils.data import DataLoader

from config import TrainConfig
from constants import EXTRA_VARS, INPT_VARS, OUT_VARS
from datasets import TrainData
from train_3D import Trainer

# Note: Refactoring data loaders is planned for the near-term. Ideally,
# these fixtures allow us to isolate data loader tests from their setup.

TrainPair = tuple[TrainConfig, Trainer]
LoaderPair = tuple[TrainConfig, DataLoader]


# This micro-fixture is cached by pytest. Thus, we don't have to change
# the factory methods that throw errors during double initialization.
@pytest.fixture(scope="session")
def trainer_pair(train_config: TrainConfig) -> TrainPair:
    trainer = Trainer(train_config)

    # cur_step will set the number of pairs in the input/output sample
    trainer.init_data_loaders(cur_step=train_config.steps[0])

    return train_config, trainer


@pytest.fixture
def train_loader_pair(trainer_pair: TrainPair) -> LoaderPair:
    cfg, trainer = trainer_pair
    return cfg, trainer.train_loader


@pytest.fixture
def val_loader_pair(trainer_pair: TrainPair) -> LoaderPair:
    cfg, trainer = trainer_pair
    return cfg, trainer.val_loader


@pytest.fixture
def inference_loader_pair(trainer_pair: TrainPair) -> LoaderPair:
    cfg, trainer = trainer_pair
    return cfg, trainer.inference_loader


# The Inference loader is not included here because it doesn't store data
# in a `TrainData` object.
@pytest.fixture(params=["train", "val"])
def td_loader_pair(request, train_loader_pair, val_loader_pair) -> LoaderPair:
    if request.param == "train":
        return train_loader_pair
    else:
        return val_loader_pair


def extract_sample_arrays(td: TrainData) -> tuple[np.ndarray, np.ndarray]:
    """Extract underlying X, y pairs from TrainData object."""
    steps = len(td.td_dict.keys())
    x_arrays = [td.get_input(s).numpy(force=True) for s in range(steps)]
    y_arrays = [td.get_label(s).numpy(force=True) for s in range(steps)]

    return np.stack(x_arrays, axis=0), np.stack(y_arrays, axis=0)


# TODO(alxmrs): How can we determine `n_samples` from the input config? Timeslice?
#  Changing the "hist" parameter breaks this test.
def test_train__loads_correct_number_of_samples(train_loader_pair: LoaderPair):
    cfg, loader = train_loader_pair
    n_samples = 13
    assert (
        len(list(loader)) == n_samples
    ), f"Current config {cfg} only supports {n_samples} examples; got {len(loader)}."


def test_train__data_shape(train_loader_pair: LoaderPair):
    cfg, loader = train_loader_pair

    exp = cfg.experiment
    batch_size = cfg.batch_size
    hist = cfg.data.hist + 1

    input_var_dim = len(INPT_VARS[exp.exp_num_in]) * hist + len(
        EXTRA_VARS[exp.exp_num_extra]
    )
    output_var_dim = len(OUT_VARS[cfg.experiment.exp_num_out]) * hist

    for sample in loader:
        X, y = extract_sample_arrays(sample)
        assert X.shape == (cfg.steps[0], batch_size, input_var_dim, 180, 360)
        assert y.shape == (cfg.steps[0], batch_size, output_var_dim, 180, 360)


# TODO(alxmrs): How can we determine `n_samples` from the input config? Timeslice?
#  Changing the "hist" parameter breaks this test.
def test_val__loads_correct_number_of_samples(val_loader_pair):
    cfg, loader = val_loader_pair
    n_samples = 5
    assert (
        len(list(loader)) == n_samples
    ), f"Current config {cfg} only supports {n_samples} examples; got {len(loader)}."


def test_val__data_shape(val_loader_pair: LoaderPair):
    cfg, loader = val_loader_pair

    exp = cfg.experiment
    batch_size = cfg.batch_size
    hist = cfg.data.hist + 1

    input_var_dim = len(INPT_VARS[exp.exp_num_in]) * hist + len(
        EXTRA_VARS[exp.exp_num_extra]
    )
    output_var_dim = len(OUT_VARS[cfg.experiment.exp_num_out]) * hist

    for sample in loader:
        X, y = extract_sample_arrays(sample)
        assert X.shape == (1, batch_size, input_var_dim, 180, 360)
        assert y.shape == (1, batch_size, output_var_dim, 180, 360)


def test_inference__loads_correct_number_of_samples(inference_loader_pair: LoaderPair):
    cfg, loader = inference_loader_pair
    n_samples = 1
    assert (
        len(list(loader)) == n_samples
    ), f"Current config {cfg} only supports {n_samples} examples; got {len(loader)}."


def test_inference__data_shape(inference_loader_pair: LoaderPair):
    cfg, loader = inference_loader_pair

    exp = cfg.experiment
    batch_size = cfg.batch_size
    hist = cfg.data.hist + 1

    input_var_dim = len(INPT_VARS[exp.exp_num_in]) * hist + len(
        EXTRA_VARS[exp.exp_num_extra]
    )
    output_var_dim = len(OUT_VARS[cfg.experiment.exp_num_out]) * hist

    for sample in loader:
        inference_dataset, n = sample
        for X, y in inference_dataset:
            assert X.shape == (batch_size, input_var_dim, 180, 360)
            assert y.shape == (batch_size, output_var_dim, 180, 360)


def test__data_is_not_zeros(td_loader_pair: LoaderPair):
    cfg, loader = td_loader_pair

    for sample in loader:
        X, y = extract_sample_arrays(sample)
        assert np.count_nonzero(np.zeros(X.shape)) == 0, "Sanity check: Zero is zero."
        assert np.count_nonzero(X) != 0, "Input data should not be a zeros matrix!"
        assert np.count_nonzero(y) != 0, "Label data should not be a zeros matrix!"


def test_inference__data_is_not_zero(inference_loader_pair: LoaderPair):
    cfg, loader = inference_loader_pair

    for sample in loader:
        dataset, n = sample
        for X, y in dataset:
            assert (
                np.count_nonzero(np.zeros(X.shape)) == 0
            ), "Sanity check: Zero is zero."
            assert (
                np.count_nonzero(X.numpy()) != 0
            ), "Input data should not be a zeros matrix!"
            assert (
                np.count_nonzero(y.numpy()) != 0
            ), "Label data should not be a zeros matrix!"
