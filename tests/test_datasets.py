"""Test core Datasets and DataLoaders."""

import datetime
import os

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from ocean_emulators.config import TrainConfig
from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS
from ocean_emulators.datasets import OM4Dataset, TrainData, TrainDataset
from ocean_emulators.train import Trainer
from ocean_emulators.utils.data import Normalize, validate_data
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.train import collate_om4, collate_train_data
from tests.conftest import DataSourceDims

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


def calc_num_samples(cfg: TrainConfig, time_slice: slice) -> int:
    ds = xr.open_zarr(os.path.join(cfg.experiment.data_dir, cfg.data.data_path))

    data_size = ds.sel(time=time_slice).time.size
    steps = cfg.steps[0]
    hist = cfg.data.hist
    stride = cfg.data_stride[0]

    return data_size - (steps * (cfg.data.hist + 1) * stride) - hist * stride


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
        dtype=np.float16,
        shape=vector_of(50),
        elements=st.floats(
            -90.0, 90.0, allow_nan=False, allow_infinity=False, width=16
        ),
        unique=True,
    ),
    lng=arrays(
        dtype=np.float16,
        shape=vector_of(50),
        elements=st.floats(0, 360.0, allow_nan=False, allow_infinity=False, width=16),
        unique=True,
    ),
    days_since_start=arrays(
        dtype=np.int32,
        shape=vector_of(50),
        elements=st.integers(min_value=0, max_value=999),
        unique=True,
    ),
    start_day=st.dates(
        min_value=datetime.date(1900, 1, 1),  # to quiet cftime warning about year < 0
    ),
    calendar=st.sampled_from(["noleap", "standard"]),
)
@example(
    data_var_index=0,
    lat=np.array([-90.0, 0.0, 90.0]),
    lng=np.array([0.0, 180.0]),
    days_since_start=np.array([5, 10, 15, 20, 25]),
    start_day=datetime.date(2020, 1, 1),
    calendar="noleap",
)
@example(
    data_var_index=255,
    lat=np.array([90.00]),
    lng=np.array([360.0]),
    days_since_start=np.array([999]),
    start_day=datetime.date(2000, 5, 1),
    calendar="noleap",
)
@example(
    data_var_index=7,
    lat=np.array([0.0]),
    lng=np.array([0.0]),
    days_since_start=np.array([0], dtype=np.uint32),
    start_day=datetime.date(2000, 5, 1),
    calendar="noleap",
)
@example(
    lat=np.array([32.87]),
    lng=np.array([0.0]),
    data_var_index=0,
    days_since_start=np.array([0], dtype=np.uint32),
    start_day=datetime.date(2000, 5, 1),
    calendar="noleap",
)
@example(
    data_var_index=0,
    lat=np.array([2.0]),
    lng=np.array([1.375]),
    days_since_start=np.array([0], dtype=np.uint32),
    start_day=datetime.date(2000, 1, 1),
    calendar="noleap",
)
@settings(deadline=1000)
def test_test_util__data_source_roundtrip(
    data_var_index: int,
    lat: NDArray[np.floating],
    lng: NDArray[np.floating],
    days_since_start: NDArray[np.uint32],
    start_day: datetime.date,
    calendar: str,
) -> None:
    # We use hour=12 because that's what cftime uses when
    # converting from ordinals (in DataSourceDims)
    start_day_cf = cftime.datetime(
        start_day.year, start_day.month, start_day.day, hour=12, calendar=calendar
    )

    # start
    dims_uncoded = DataSourceDims(
        lat=lat,
        lng=lng,
        days_since_start=days_since_start,
        start_day=start_day_cf,
    )
    # intermediate representation: `xarray.DataArray`
    da = dims_uncoded.encode(data_var_index)

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


@pytest.mark.all_configs
def test_train__loads_correct_number_of_samples(train_loader_pair: LoaderPair):
    cfg, loader = train_loader_pair
    n_samples = calc_num_samples(cfg, cfg.train.time_slice)
    assert len(list(loader)) == n_samples, (
        f"Current config {cfg} only supports {n_samples} examples; got {len(loader)}."
    )


def test_train__data_shape(train_loader_pair: LoaderPair):
    cfg, loader = train_loader_pair

    exp = cfg.experiment
    batch_size = cfg.batch_size
    hist = cfg.data.hist + 1

    input_var_dim = len(PROGNOSTIC_VARS[exp.prognostic_vars_key]) * hist + len(
        BOUNDARY_VARS[exp.boundary_vars_key]
    )
    output_var_dim = len(PROGNOSTIC_VARS[exp.prognostic_vars_key]) * hist

    for sample in loader:
        X, y = extract_sample_arrays(sample)
        assert X.shape == (cfg.steps[0], batch_size, input_var_dim, 180, 360)
        assert y.shape == (cfg.steps[0], batch_size, output_var_dim, 180, 360)


def test_val__loads_correct_number_of_samples(val_loader_pair):
    cfg, loader = val_loader_pair
    n_samples = calc_num_samples(cfg, cfg.val.time_slice)
    assert len(list(loader)) == n_samples, (
        f"Current config {cfg} only supports {n_samples} examples; got {len(loader)}."
    )


def test_val__data_shape(val_loader_pair: LoaderPair):
    cfg, loader = val_loader_pair

    exp = cfg.experiment
    batch_size = cfg.batch_size
    hist = cfg.data.hist + 1

    input_var_dim = len(PROGNOSTIC_VARS[exp.prognostic_vars_key]) * hist + len(
        BOUNDARY_VARS[exp.boundary_vars_key]
    )
    output_var_dim = len(PROGNOSTIC_VARS[exp.prognostic_vars_key]) * hist

    num_samples = len(loader)
    for i, sample in enumerate(loader):
        X, y = extract_sample_arrays(sample)

        assert X.shape[0] == 1  # validation always has 1 step
        # Last validation batch may have fewer samples
        assert X.shape[1] == batch_size or (
            i == num_samples - 1 and X.shape[1] < batch_size
        )
        assert X.shape[2:] == (input_var_dim, 180, 360)

        assert y.shape[0] == 1
        assert y.shape[1] == batch_size or (
            i == num_samples - 1 and y.shape[1] < batch_size
        )
        assert y.shape[2:] == (output_var_dim, 180, 360)


def test_inference__loads_correct_number_of_samples(inference_loader_pair: LoaderPair):
    cfg, loader = inference_loader_pair
    n_samples = 1
    assert len(list(loader)) == n_samples, (
        f"Current config {cfg} only supports {n_samples} examples; got {len(loader)}."
    )


def test_inference__data_shape(inference_loader_pair: LoaderPair):
    cfg, loader = inference_loader_pair

    exp = cfg.experiment
    batch_size = 1  # Inference always uses batch size 1
    hist = cfg.data.hist + 1

    input_var_dim = len(PROGNOSTIC_VARS[exp.prognostic_vars_key]) * hist + len(
        BOUNDARY_VARS[exp.boundary_vars_key]
    )
    output_var_dim = len(PROGNOSTIC_VARS[exp.prognostic_vars_key]) * hist

    for sample in loader:
        inference_dataset, n = sample
        for X, y in inference_dataset:
            assert X.shape == (batch_size, input_var_dim, 180, 360)
            assert y.shape == (batch_size, output_var_dim, 180, 360)


@pytest.mark.all_configs
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
            assert np.count_nonzero(np.zeros(X.shape)) == 0, (
                "Sanity check: Zero is zero."
            )
            assert np.count_nonzero(X.numpy()) != 0, (
                "Input data should not be a zeros matrix!"
            )
            assert np.count_nonzero(y.numpy()) != 0, (
                "Label data should not be a zeros matrix!"
            )


@pytest.mark.all_configs
def test_om4__is_equal_to_v1_data_loader(train_loader_pair: LoaderPair):
    cfg, loader = train_loader_pair

    data_path = os.path.join(cfg.experiment.data_dir, cfg.data.data_path)
    ds = xr.open_dataset(data_path, chunks={})

    prognostic = PROGNOSTIC_VARS[cfg.experiment.prognostic_vars_key]
    boundary = BOUNDARY_VARS[cfg.experiment.boundary_vars_key]

    val_ds, _, _ = validate_data(ds, xr.Dataset(), xr.Dataset())

    om4 = OM4Dataset(
        val_ds.sel(time=cfg.train.time_slice),
        prognostic,
        boundary,
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

    # Why are we sorting here? Well, the default data loader uses a random sampler. So
    # we use sorting as a simple way to compare the two loaders (without having to
    # monkeypatch the train loader fixture).
    original_samples = sorted(
        [extract_sample_arrays(sample) for sample in loader], key=key
    )
    om4_samples = sorted(
        [extract_sample_arrays(sample) for sample in om4_loader], key=key
    )

    for (x_orig, y_orig), (x_new, y_new) in zip(original_samples, om4_samples):
        assert x_orig.dtype == x_new.dtype, "Input data types do not match."
        assert y_orig.dtype == y_new.dtype, "Output data types do not match."

        x_not_close = np.isclose(x_orig, x_new) == False  # noqa: E712
        y_not_close = np.isclose(y_orig, y_new) == False  # noqa: E712

        x_not_close_index = list(zip(*np.where(x_not_close)))
        y_not_close_index = list(zip(*np.where(y_not_close)))

        assert not np.any(x_not_close), (
            f"{len(x_not_close_index)} values differ: "
            f"{x_orig[x_not_close]} != {x_new[x_not_close]}."
        )
        assert not np.any(y_not_close), (
            f"{len(y_not_close_index)} values differ: "
            f"{y_orig[y_not_close]} != {y_new[y_not_close]}."
        )


@pytest.fixture
def traindataset_input():
    # Create data
    coords = {"time": range(10), "lat": range(2), "lon": range(2)}
    data_array = torch.ones(4, 10, 2, 2)  # [vars, time, lat, lon]

    data = xr.Dataset(
        {
            name: xr.DataArray(
                data_array[i], dims=["time", "lat", "lon"], coords=coords
            )
            for i, name in enumerate(
                ["prognostic1", "prognostic2", "boundary1", "boundary2"]
            )
        }
    )
    prognostic_var_names = ["prognostic1", "prognostic2"]
    boundary_var_names = ["boundary1", "boundary2"]

    # Create test data with mean and std
    data_mean = xr.Dataset(
        {
            "prognostic1": 0.0,
            "prognostic2": 0.0,
            "boundary1": 0.0,
            "boundary2": 0.0,
        },
        coords={"lat": [0], "lon": [0]},
    )
    data_std = xr.Dataset(
        {
            "prognostic1": 1.0,
            "prognostic2": 1.0,
            "boundary1": 1.0,
            "boundary2": 1.0,
        },
        coords={"lat": [0], "lon": [0]},
    )

    wet = torch.ones(2, 2, 2)
    wet_surface = torch.ones(2, 2)

    # Initialize and yield within the MultitonScope
    with MultitonScope():
        _ = Normalize.init_instance(
            data_mean=data_mean,
            data_std=data_std,
            prognostic_var_names=["prognostic1", "prognostic2"],
            boundary_var_names=["boundary1", "boundary2"],
            wet_mask=wet,
        )
        traindataset = TrainDataset(
            data=data,
            prognostic_var_names=prognostic_var_names,
            boundary_var_names=boundary_var_names,
            wet=wet,
            wet_surface=wet_surface,
            hist=0,
            steps=2,
            stride=1,
        )
        yield traindataset


def test_train_dataset(traindataset_input):
    traindataset = traindataset_input
    td = collate_train_data([traindataset[0], traindataset[1], traindataset[2]])
    pred = torch.randn_like(td.get_label(0)) * 0.1

    inp1 = td.get_input(1).clone()
    td.merge_prognostic_and_boundary(pred, 1)

    td_new = collate_train_data([traindataset[0], traindataset[1], traindataset[2]])
    assert torch.equal(td_new.get_input(1), inp1)


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
