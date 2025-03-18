"""Test core Datasets and DataLoaders."""

import datetime
import itertools

import cftime
import numpy as np
import pytest
import xarray as xr
from conftest import DataSourceDims
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from config import TrainConfig
from constants import EXTRA_VARS, INPT_VARS, OUT_VARS
from datasets import OM4Dataset, TrainData
from train_3D import Trainer
from utils.train import collate_om4

# Note: Refactoring data loaders is planned for the near-term. Ideally,
# these fixtures allow us to isolate data loader tests from their setup.

LoaderPair = tuple[TrainConfig, DataLoader]
TrainPair = tuple[TrainConfig, Trainer]


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


def vector_of(max_vec_size: int, min_vec_size=1):
    """A hypothesis helper: generates vector array shapes."""
    return st.lists(
        st.integers(min_value=min_vec_size, max_value=max_vec_size),
        min_size=1,
        max_size=1,
    ).map(tuple)


@given(
    data_var_index=st.integers(min_value=0, max_value=255),
    lat=arrays(
        dtype=np.float64,
        shape=vector_of(50),
        elements=st.floats(-90.0, 90.0, allow_nan=False, allow_infinity=False),
    ),
    lng=arrays(
        dtype=np.float64,
        shape=vector_of(50),
        elements=st.floats(0, 360.0, allow_nan=False, allow_infinity=False),
    ),
    days_since_start=arrays(
        dtype=np.int32,
        shape=vector_of(50),
        elements=st.integers(min_value=0, max_value=999),
    ),
    start_day=st.datetimes(),
    calendar=st.sampled_from(["noleap", "standard"]),
)
@example(
    data_var_index=0,
    lat=np.array([-90.0, 0.0, 90.0]),
    lng=np.array([0.0, 180.0]),
    days_since_start=np.array([5, 10, 15, 20, 25]),
    start_day=datetime.datetime(2020, 1, 1),
    calendar="noleap",
)
@example(
    data_var_index=255,
    lat=np.array([90.00]),
    lng=np.array([360.0]),
    days_since_start=np.array([999]),
    start_day=datetime.datetime(2000, 5, 1, 12),
    calendar="noleap",
)
@example(
    data_var_index=7,
    lat=np.array([0.0]),
    lng=np.array([0.0]),
    days_since_start=np.array([0], dtype=np.int32),
    start_day=datetime.datetime(2000, 5, 1, 12),
    calendar="noleap",
)
@example(
    lat=np.array([32.87]),
    lng=np.array([0.0]),
    data_var_index=0,
    days_since_start=np.array([0], dtype=np.int32),
    start_day=datetime.datetime(2000, 5, 1, 12),
    calendar="noleap",
)
@example(
    data_var_index=0,
    lat=np.array([2.0]),
    lng=np.array([1.375]),
    days_since_start=np.array([0], dtype=np.int32),
    start_day=datetime.datetime(2000, 1, 1, 0, 0),
    calendar="noleap",
)
@settings(deadline=1000)
def test_test_util__data_source_roundtrip(
    data_var_index: int,
    lat: NDArray[np.floating],
    lng: NDArray[np.floating],
    days_since_start: NDArray[np.int32],
    start_day: datetime.datetime,
    calendar: str,
) -> None:
    start_day_cf = cftime.datetime.fromordinal(start_day.toordinal(), calendar=calendar)

    # start
    dims_uncoded = DataSourceDims(
        lat=lat,
        lng=lng,
        days_since_start=days_since_start,
        start_day=start_day_cf,
    )
    # intermediate representation: `xarray.DataArray`
    da = dims_uncoded.encode(data_var_index)

    # Additional property: If the inputs are unique, then the outputs should be unique.
    inputs = itertools.product(*[v.values for v in dims_uncoded.to_coords().values()])
    # The cross product of the input values maps on to one output coordinate.
    inputs_are_unique = len(set(inputs)) == len(list(inputs))
    if inputs_are_unique:
        unique, counts = np.unique(da.values.flatten(), return_counts=True)
        duplicates, num_dups = unique[counts > 1], counts[counts > 1]
        assert len(unique) == da.size, (
            f"All values are unique. frequency of duplicates: "
            f"{list(zip(duplicates, num_dups))}"
        )

    # end
    dims_decoded, decoded_var_index = DataSourceDims.decode(da)

    assert dims_decoded == dims_uncoded
    assert decoded_var_index == data_var_index


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

    num_samples = len(loader)
    for i, sample in enumerate(loader):
        X, y = extract_sample_arrays(sample)

        # Last validation batch may have fewer samples
        assert X.shape[0] == cfg.steps[0]
        assert X.shape[1] == batch_size or (
            i == num_samples - 1 and X.shape[1] < batch_size
        )
        assert X.shape[2:] == (input_var_dim, 180, 360)

        assert y.shape[0] == cfg.steps[0]
        assert y.shape[1] == batch_size or (
            i == num_samples - 1 and y.shape[1] < batch_size
        )
        assert y.shape[2:] == (output_var_dim, 180, 360)


def test_inference__loads_correct_number_of_samples(inference_loader_pair: LoaderPair):
    cfg, loader = inference_loader_pair
    n_samples = 1
    assert (
        len(list(loader)) == n_samples
    ), f"Current config {cfg} only supports {n_samples} examples; got {len(loader)}."


def test_inference__data_shape(inference_loader_pair: LoaderPair):
    cfg, loader = inference_loader_pair

    exp = cfg.experiment
    batch_size = 1  # Inference always uses batch size 1
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


def test_om4__is_equal_to_v1_data_loader(train_loader_pair: LoaderPair):
    cfg, loader = train_loader_pair

    ds = xr.open_dataset(cfg.data.data_path, chunks={})

    input_vars = INPT_VARS[cfg.experiment.exp_num_in]
    extra_vars = EXTRA_VARS[cfg.experiment.exp_num_extra]
    output_vars = OUT_VARS[cfg.experiment.exp_num_out]

    om4 = OM4Dataset(
        ds,
        input_vars,
        extra_vars,
        output_vars,
        cfg.data.hist,
        cfg.steps[0],
        cfg.data_stride[0],
    )

    om4_loader = DataLoader(
        om4,
        batch_size=cfg.batch_size,
        collate_fn=collate_om4,
    )

    def key(x):
        return np.sum(x[0].flat) + np.sum(x[1].flat)

    def as_numpy(x):
        return x[0].cpu().detach().numpy(), x[1].cpu().detach().numpy()

    original_samples = sorted(
        [extract_sample_arrays(sample) for sample in loader], key=key
    )
    om4_samples = sorted([as_numpy(sample) for sample in om4_loader], key=key)

    atol = 0.01
    for (x_orig, y_orig), (x_new, y_new) in zip(original_samples, om4_samples):
        x_not_close = not np.isclose(x_orig, x_new, atol=atol)
        y_not_close = not np.isclose(y_orig, y_new, atol=atol)

        x_not_close_index = list(zip(*np.where(x_not_close)))
        y_not_close_index = list(zip(*np.where(y_not_close)))

        assert np.allclose(x_orig, x_new, atol=atol), (
            f"{len(x_not_close_index)} values differ: "
            f"{x_orig[x_not_close]} != {x_new[x_not_close]}."
        )
        assert np.allclose(y_orig, y_new, atol=atol), (
            f"{len(y_not_close_index)} values differ: "
            f"{y_orig[y_not_close]} != {y_new[y_not_close]}."
        )


@pytest.mark.manual
def test_profile__loader__1gb(td_loader_pair: LoaderPair, benchmark):
    cfg, loader = td_loader_pair

    @benchmark
    def bench():
        for sample in loader:
            _ = sample


@pytest.mark.manual
def test_profile__inference_loader__1gb(inference_loader_pair: LoaderPair, benchmark):
    cfg, loader = inference_loader_pair

    @benchmark
    def bench():
        for sample in loader:
            dataset, n = sample
            for X, y in dataset:
                _, _ = X, y
