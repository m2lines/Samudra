"""Test core Datasets and DataLoaders."""

import contextlib
import datetime
import os
from typing import Callable, Generator

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset, DataLoader

from ocean_emulators.config import TrainConfig
from ocean_emulators.constants import (
    BOUNDARY_VARS,
    PROGNOSTIC_VARS,
    LoaderVersion,
    TensorMap,
)
from ocean_emulators.datasets import OM4Dataset, TrainData, TrainDataset
from ocean_emulators.train import Trainer
from ocean_emulators.utils.data import Normalize, extract_wet_mask, validate_data
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.train import collate_om4, collate_train_data
from tests.conftest import DEFAULT_CONFIG, DataSourceDims

TrainPair = tuple[TrainConfig, Trainer]


@contextlib.contextmanager
def make_loader(
    cfg,
    time_slice: slice | None = None,
    drop_last: bool = True,
    version: LoaderVersion = LoaderVersion.OM4_EAGER,
) -> Generator[DataLoader, None, None]:
    if time_slice is None:
        time_slice = cfg.train.time_slice
    if isinstance(version, str):
        version = LoaderVersion(version)

    ds = xr.open_dataset(cfg.experiment.data_dir / cfg.data.data_path, chunks={})
    ds_means = xr.open_dataset(
        cfg.experiment.data_dir / cfg.data.data_means_path, chunks={}
    )
    ds_stds = xr.open_dataset(
        cfg.experiment.data_dir / cfg.data.data_stds_path, chunks={}
    )

    prognostic = PROGNOSTIC_VARS[cfg.experiment.prognostic_vars_key]
    boundary = BOUNDARY_VARS[cfg.experiment.boundary_vars_key]

    try:
        TensorMap.get_instance()
    except ValueError:
        TensorMap.init_instance(
            cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key
        )

    ds_, means_, stds_ = validate_data(ds, ds_means, ds_stds)
    wet, wet_surface = extract_wet_mask(ds_, prognostic, cfg.data.hist)

    with MultitonScope():
        Normalize.init_instance(means_, stds_, prognostic, boundary, wet)

        match version:
            case LoaderVersion.OM4_EAGER:
                data: ConcatDataset[TrainDataset] = ConcatDataset(
                    [
                        TrainDataset(
                            data=ds_.sel(time=time_slice),
                            prognostic_var_names=prognostic,
                            boundary_var_names=boundary,
                            wet=wet,
                            wet_surface=wet_surface,
                            hist=cfg.data.hist,
                            steps=cfg.steps[0],
                            stride=stride,
                        )
                        for stride in cfg.data_stride
                    ]
                )
                collate_fn: Callable = collate_train_data
            case LoaderVersion.OM4_LAZY:
                data = ConcatDataset(
                    [
                        OM4Dataset(
                            data=ds_.sel(time=time_slice),
                            prognostic_var_names=prognostic,
                            boundary_var_names=boundary,
                            hist=cfg.data.hist,
                            steps=cfg.steps[0],
                            stride=stride,
                        )
                        for stride in cfg.data_stride
                    ]
                )
                collate_fn = collate_om4
            case _:
                raise ValueError(f"Unknown loader version: {version}")

        loader = DataLoader(
            data,
            batch_size=cfg.batch_size,
            drop_last=drop_last,
            collate_fn=collate_fn,
        )

        yield loader


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


def test_loader__data_shape(train_config, history, loader_version):
    train_config.data.hist = history

    with make_loader(train_config, version=loader_version) as loader:
        exp = train_config.experiment
        batch_size = train_config.batch_size
        hist = train_config.data.hist + 1

        input_var_dim = len(PROGNOSTIC_VARS[exp.prognostic_vars_key]) * hist + len(
            BOUNDARY_VARS[exp.boundary_vars_key]
        )
        output_var_dim = len(PROGNOSTIC_VARS[exp.prognostic_vars_key]) * hist

        n_samples = calc_num_samples(train_config, train_config.train.time_slice)
        samples = list(loader)

        assert len(samples) == n_samples, (
            f"Current config {train_config} only supports {n_samples} examples; "
            f"got {len(loader)}."
        )

        for sample in samples:
            X, y = extract_sample_arrays(sample)
            assert X.shape == (
                train_config.steps[0],
                batch_size,
                input_var_dim,
                180,
                360,
            )
            assert y.shape == (
                train_config.steps[0],
                batch_size,
                output_var_dim,
                180,
                360,
            )


def test__data_is_not_zeros(train_config):
    cfg = train_config

    with make_loader(cfg) as loader:
        for sample in loader:
            X, y = extract_sample_arrays(sample)
            assert np.count_nonzero(np.zeros(X.shape)) == 0, (
                "Sanity check: Zero is zero."
            )
            assert np.count_nonzero(X) != 0, "Input data should not be a zeros matrix!"
            assert np.count_nonzero(y) != 0, "Label data should not be a zeros matrix!"


@pytest.mark.all_configs
def test_om4__is_equal_to_v1_data_loader(train_config):
    cfg = train_config

    with (
        make_loader(cfg) as loader,
        make_loader(cfg, version=LoaderVersion.OM4_LAZY) as om4_loader,
    ):

        def key(x):
            return np.sum(x[0].flat) + np.sum(x[1].flat)

        # Why are we sorting here? Well, the default data loader uses a random sampler.
        # So we use sorting as a simple way to compare the two loaders (without having
        # to monkeypatch the train loader fixture).
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
@pytest.mark.parametrize(
    "data_source,config_name", [("mock", DEFAULT_CONFIG)], indirect=True
)
def test_profile__loader__1gb(train_config, loader_version, benchmark):
    cfg = train_config

    with make_loader(cfg, version=loader_version) as loader:

        @benchmark
        def bench():
            for sample in loader:
                _ = sample
