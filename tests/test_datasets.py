"""Test core Datasets and DataLoaders."""

import numpy as np
import pytest
from conftest import parse_encoded_float
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
    steps = len(td)
    x_arrays = [td.get_input(s).numpy(force=True) for s in range(steps)]
    y_arrays = [td.get_label(s).numpy(force=True) for s in range(steps)]

    return np.stack(x_arrays, axis=0), np.stack(y_arrays, axis=0)


def test_test_util__parse_encoded_float():
    #       AAAAGGGG.TTTDD
    test1 = 27760145.03000
    assert parse_encoded_float(test1) == dict(
        lat=27.76,
        lng=14.5,
        days_since_start=30,
        data_var_index=0,
    )

    #       AAAAGGGG.TTTDD
    test2 = 27760145.03020
    assert parse_encoded_float(test2) == dict(
        lat=27.76,
        lng=14.5,
        days_since_start=30,
        data_var_index=20,
    )


def test_encode_decode_float():
    # import logging
    import xarray as xr

    summer_of_love = xr.cftime_range(
        "1969-08-05", "1969-12-31", freq="5D", calendar="noleap"
    )

    coords = {
        "lon": xr.DataArray(np.arange(0.5, 360, 1), dims=["lon"]),  # Float[360]
        "lat": xr.DataArray(np.arange(-89.24, 90, 1), dims=["lat"]),  # Float[180]
        "time": xr.DataArray(summer_of_love, dims=["time"]),  # CFTimeIndex[30]
    }

    # normal = np.random.normal(
    #     size=(len(coords["lat"]), len(coords["lon"]))
    # )  # Float[180, 360]

    # Create array of relative times (number of days since start).
    timedeltas = [date - summer_of_love[0] for date in summer_of_love]
    days_from_start = np.array(
        [delta.total_seconds() / (24 * 3600) for delta in timedeltas]
    )
    days_reshaped = days_from_start[:, np.newaxis, np.newaxis]  # Float[30, 1, 1]

    latlng_grid = np.stack(
        np.meshgrid(coords["lat"][::-1], coords["lon"], indexing="ij"),
        axis=0,
    )
    latlng_grid_3sf = np.around(latlng_grid, decimals=2)

    template_grid = latlng_grid_3sf[0, :, :] * 1_000_000 + latlng_grid_3sf[1, :, :] * 10
    rolled_out_grid = np.repeat(
        template_grid[np.newaxis, :, :], len(summer_of_love), axis=0
    )

    # A floating point digit-encoded grid.
    # ------------------------------------
    # Each number in this array is an interpretable float with the following scheme:
    # AAAAGGGG.TTTDD
    # - A := Latitude, which ranges from 90.00 <--> -90.00
    # - G := Longitude, which ranges from 000.0 <--> 360.0
    # - T := Time (the number of days since the start time).
    # - D := (optional) A int representing the index of the current data variable.
    interpretable_grid = rolled_out_grid + days_reshaped / 1000  # Float[30, 180, 360]

    vars_2d = {
        var: xr.DataArray(interpretable_grid, dims=["time", "lat", "lon"])
        + float(i) / 100_000
        for i, var in enumerate(["hfds", "tauuo", "tauvo", "zos"])
    }
    # vars_3d = {
    #     f"{var}_{lev}": xr.DataArray(interpretable_grid, dims=["time", "lat", "lon"])
    #     + float(i + j + len(vars_2d)) / 100_000
    #     for i, var in enumerate(["so", "thetao", "uo", "vo"])
    #     for j, lev in enumerate(c.DEPTH_I_LEVELS)
    # }
    # # Mask with a binary circle.
    # masks = {
    #     f"mask_{lev}": xr.DataArray(
    #         np.where(normal > 0.5**lev, 1, 0), dims=["lat", "lon"]
    #     )
    #     for lev in range(len(c.DEPTH_I_LEVELS))
    # }
    ds = xr.Dataset(vars_2d, coords=coords)
    np_vals = ds.to_array().to_numpy().flatten()
    s = set()
    for v in np_vals:
        if v in s:
            raise ValueError(f"Duplicate encoded float: {v}")
        s.add(v)

    s = set()
    from conftest import GridPoint, encode_float

    for days in range(0, 100, 5):
        for var in range(0, 10):
            for lat in np.arange(-89.24, 90, 1):
                for lng in np.arange(0.5, 360, 1):
                    org_dict = GridPoint(
                        lat=lat,
                        lng=lng,
                        days_since_start=days,
                        data_var_index=var,
                    )
                    encoded_float = encode_float(org_dict)
                    decoded_dict = parse_encoded_float(encoded_float)
                    # logging.info(f"Encoded float: {encoded_float}")
                    assert org_dict == decoded_dict
                    if encoded_float in s:
                        raise ValueError(f"Duplicate encoded float: {encoded_float}")
                    s.add(encoded_float)


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
