"""Test core Datasets and DataLoaders."""

import contextlib
import dataclasses
import datetime
import json
from collections.abc import Generator

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
from einops import rearrange as einops_rearrange
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray
from torch.utils.data import ConcatDataset, DataLoader

from ocean_emulators.config import TimeConfig, TrainConfig
from ocean_emulators.constants import (
    BOUNDARY_VARS,
    DEPTH_I_LEVELS,
    DEPTH_LEVELS,
    PROGNOSTIC_VARS,
    LoaderVersion,
    TensorMap,
)
from ocean_emulators.datasets import (
    InferenceDataset,
    TorchTrainDataset,
    TrainData,
    TrainDataLoader,
    _dataset_to_numpy,
)
from ocean_emulators.utils.data import (
    PACKED_CACHE_FORMAT,
    DataSource,
    Masks,
    Normalize,
    _packed_spatial_features,
)
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.train import collate_raw_train_data
from tests.conftest import DEFAULT_CONFIG, DataSourceDims, TrainPair, cache_dir


@pytest.fixture
def inference_loader_pair(trainer_pair: TrainPair) -> tuple[TrainConfig, DataLoader]:
    cfg, trainer = trainer_pair
    return cfg, trainer.inference_loader


@contextlib.contextmanager
def make_loader(
    cfg: TrainConfig,
    time_config: TimeConfig | None = None,
    drop_last: bool = True,
    version: LoaderVersion | None = None,
) -> Generator[DataLoader | TrainDataLoader, None, None]:
    if time_config is None:
        time_config = cfg.train_time

    prognostic = PROGNOSTIC_VARS[cfg.experiment.prognostic_vars_key]
    boundary = BOUNDARY_VARS[cfg.experiment.boundary_vars_key]

    data_config = (
        cfg.data
        if version is None
        else cfg.data.model_copy(update={"loader_version": str(version.value)})
    )

    container = data_config.build(
        cfg.experiment.resolved_data_root, prognostic, boundary
    )
    version = container.loader_version
    src = container.source
    if src.is_compact and version != LoaderVersion.OM4_TORCH:
        pytest.skip(f"{version} does not support compact data.")

    with MultitonScope():
        TensorMap.init_instance(
            cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key
        )

        match version:
            case LoaderVersion.OM4_TORCH:
                dataset_list = [
                    TorchTrainDataset(
                        src=src.slice(time_config),
                        prognostic_var_names=prognostic,
                        boundary_var_names=boundary,
                        hist=cfg.data.hist,
                        steps=cfg.steps[0],
                        normalize_before_mask=cfg.data.normalize_before_mask,
                        masked_fill_value=cfg.data.masked_fill_value,
                        stride=stride,
                        temporal_stride=cfg.temporal_stride,
                    )
                    for stride in cfg.data_stride
                ]

                data: ConcatDataset = ConcatDataset(dataset_list)
                collate_fn = collate_raw_train_data

                raw_loader = DataLoader(
                    data,
                    batch_size=cfg.batch_size,
                    drop_last=drop_last,
                    collate_fn=collate_fn,
                )

                loader = TrainDataLoader(raw_loader, dataset_list, torch.device("cpu"))
                yield loader
            case _:
                raise ValueError(f"Unknown loader version: {version}")


def extract_sample_arrays(td: TrainData) -> tuple[np.ndarray, np.ndarray]:
    """Extract underlying X, y pairs from TrainData object."""
    steps = len(td)
    x_arrays = [td.get_input(s).numpy(force=True) for s in range(steps)]
    y_arrays = [td.get_label(s).numpy(force=True) for s in range(steps)]

    return np.stack(x_arrays, axis=0), np.stack(y_arrays, axis=0)


def calc_num_samples(cfg: TrainConfig, time_slice: slice) -> int:
    ds = cfg.experiment.resolved_data_root.resolve(cfg.data.data_location).open()

    data_size = ds.sel(time=time_slice).time.size
    steps = cfg.steps[0]
    hist = cfg.data.hist
    stride = cfg.data_stride[0]
    temporal_stride = cfg.temporal_stride

    base_size = data_size - (steps * (cfg.data.hist + 1) * stride) - hist * stride
    return max(0, (base_size + temporal_stride - 1) // temporal_stride)


def test_torch_train_dataset_temporal_stride_subsamples_window_starts() -> None:
    time = np.arange(20)
    lat = [0.0]
    lon = [0.0]
    base = np.arange(20, dtype=np.float32).reshape(20, 1, 1)

    data = xr.Dataset(
        {
            "prognostic1": (["time", "lat", "lon"], base),
            "boundary1": (["time", "lat", "lon"], base + 100.0),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    means = xr.Dataset(
        {
            "prognostic1": (["lat", "lon"], np.zeros((1, 1), dtype=np.float32)),
            "boundary1": (["lat", "lon"], np.zeros((1, 1), dtype=np.float32)),
        },
        coords={"lat": lat, "lon": lon},
    )
    stds = xr.Dataset(
        {
            "prognostic1": (["lat", "lon"], np.ones((1, 1), dtype=np.float32)),
            "boundary1": (["lat", "lon"], np.ones((1, 1), dtype=np.float32)),
        },
        coords={"lat": lat, "lon": lon},
    )
    masks = Masks(
        prognostic=torch.ones((1, 1, 1), dtype=torch.bool),
        boundary=torch.ones((1, 1), dtype=torch.bool),
    )
    src = DataSource("temporal_stride_test", data, means, stds, masks)

    with MultitonScope():
        _ = Normalize.init_instance(
            src,
            prognostic_var_names=["prognostic1"],
            boundary_var_names=["boundary1"],
        )
        every_point = TorchTrainDataset(
            src=src,
            prognostic_var_names=["prognostic1"],
            boundary_var_names=["boundary1"],
            hist=0,
            steps=1,
            normalize_before_mask=True,
            masked_fill_value=0.0,
            stride=1,
            temporal_stride=1,
        )
        every_other = TorchTrainDataset(
            src=src,
            prognostic_var_names=["prognostic1"],
            boundary_var_names=["boundary1"],
            hist=0,
            steps=1,
            normalize_before_mask=True,
            masked_fill_value=0.0,
            stride=1,
            temporal_stride=2,
        )

    assert len(every_point) == 19
    assert len(every_other) == 10

    first_start = int(every_other._get_x_index(0, step=0).values[0])
    second_start = int(every_other._get_x_index(1, step=0).values[0])
    third_start = int(every_other._get_x_index(2, step=0).values[0])
    assert (first_start, second_start, third_start) == (0, 2, 4)


def test_torch_train_dataset_compact_llc_like_loader_matches_flat() -> None:
    levels = len(DEPTH_I_LEVELS)
    time = np.arange(4)
    lat = [0.0]
    lon = [0.0]
    lev = np.asarray(DEPTH_LEVELS, dtype=np.float32)

    def make_3d(offset: float) -> np.ndarray:
        base = np.arange(time.size * levels, dtype=np.float32).reshape(
            time.size, levels, 1, 1
        )
        return base + offset

    compact_data = xr.Dataset(
        {
            "U": (["time", "lev", "lat", "lon"], make_3d(0.0)),
            "V": (["time", "lev", "lat", "lon"], make_3d(1000.0)),
            "Theta": (["time", "lev", "lat", "lon"], make_3d(2000.0)),
            "Salt": (["time", "lev", "lat", "lon"], make_3d(3000.0)),
            "Eta": (
                ["time", "lat", "lon"],
                np.arange(time.size, dtype=np.float32).reshape(time.size, 1, 1),
            ),
            "oceTAUX": (
                ["time", "lat", "lon"],
                (10.0 + np.arange(time.size, dtype=np.float32)).reshape(time.size, 1, 1),
            ),
            "oceTAUY": (
                ["time", "lat", "lon"],
                (20.0 + np.arange(time.size, dtype=np.float32)).reshape(time.size, 1, 1),
            ),
            "oceQnet": (
                ["time", "lat", "lon"],
                (30.0 + np.arange(time.size, dtype=np.float32)).reshape(time.size, 1, 1),
            ),
            "wetmask": (
                ["lev", "lat", "lon"],
                np.ones((levels, 1, 1), dtype=bool),
            ),
        },
        coords={"time": time, "lev": lev, "lat": lat, "lon": lon},
    )
    compact_means = compact_data.drop_vars("wetmask").mean("time", keep_attrs=True)
    compact_stds = xr.zeros_like(compact_means, dtype=np.float32) + 1.0

    flat_data = compact_data.drop_vars("wetmask").copy()
    for var_name in ["U", "V", "Theta", "Salt"]:
        data_array = flat_data[var_name]
        for level_index in range(levels):
            flat_data[f"{var_name}_{level_index}"] = data_array.isel(lev=level_index)
        flat_data = flat_data.drop_vars(var_name)
    flat_data["wetmask"] = compact_data["wetmask"]

    flat_means = compact_means.drop_vars(["U", "V", "Theta", "Salt"]).copy()
    flat_stds = compact_stds.drop_vars(["U", "V", "Theta", "Salt"]).copy()
    for var_name in ["U", "V", "Theta", "Salt"]:
        data_array = compact_means[var_name]
        std_array = compact_stds[var_name]
        for level_index in range(levels):
            flat_means[f"{var_name}_{level_index}"] = data_array.isel(lev=level_index)
            flat_stds[f"{var_name}_{level_index}"] = std_array.isel(lev=level_index)

    prognostic = PROGNOSTIC_VARS["all"]
    boundary = BOUNDARY_VARS["all"]

    compact_src = DataSource.from_datasets(
        compact_data,
        compact_means,
        compact_stds,
        name="compact_llc_like",
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
    )
    flat_src = DataSource.from_datasets(
        flat_data,
        flat_means,
        flat_stds,
        name="flat_llc_like",
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
    )

    compact_dataset = TorchTrainDataset(
        src=compact_src,
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
        hist=1,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=1,
        temporal_stride=1,
    )
    flat_dataset = TorchTrainDataset(
        src=flat_src,
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
        hist=1,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=1,
        temporal_stride=1,
    )

    compact_sample = compact_dataset.to_train_data(
        collate_raw_train_data([compact_dataset[0]])
    )
    flat_sample = flat_dataset.to_train_data(
        collate_raw_train_data([flat_dataset[0]])
    )

    compact_input, compact_label = extract_sample_arrays(compact_sample)
    flat_input, flat_label = extract_sample_arrays(flat_sample)

    np.testing.assert_allclose(compact_input, flat_input)
    np.testing.assert_allclose(compact_label, flat_label)


def test_packed_llc_spatial_features_use_sphere_coordinates_and_area() -> None:
    data = xr.Dataset(
        {
            "XC": (("lat", "lon"), np.array([[0.0, 90.0]], dtype=np.float32)),
            "YC": (("lat", "lon"), np.array([[0.0, 0.0]], dtype=np.float32)),
            "rA": (("lat", "lon"), np.array([[1.0, np.e]], dtype=np.float32)),
        }
    )

    features = _packed_spatial_features(data)

    assert features is not None
    assert features.shape == (4, 1, 2)
    torch.testing.assert_close(features[:3, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(features[:3, 0, 1], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(
        features[3, 0],
        torch.tensor(
            [-np.log(1_000_000.0), 1.0 - np.log(1_000_000.0)], dtype=torch.float32
        ),
    )


def test_torch_train_dataset_packed_llc_like_loader_matches_flat() -> None:
    levels = len(DEPTH_I_LEVELS)
    time = np.arange(4)
    lat = [0.0]
    lon = [0.0]
    lev = np.asarray(DEPTH_LEVELS, dtype=np.float32)

    def make_3d(offset: float) -> np.ndarray:
        base = np.arange(time.size * levels, dtype=np.float32).reshape(
            time.size, levels, 1, 1
        )
        return base + offset

    compact_data = xr.Dataset(
        {
            "U": (["time", "lev", "lat", "lon"], make_3d(0.0)),
            "V": (["time", "lev", "lat", "lon"], make_3d(1000.0)),
            "Theta": (["time", "lev", "lat", "lon"], make_3d(2000.0)),
            "Salt": (["time", "lev", "lat", "lon"], make_3d(3000.0)),
            "Eta": (
                ["time", "lat", "lon"],
                np.arange(time.size, dtype=np.float32).reshape(time.size, 1, 1),
            ),
            "oceTAUX": (
                ["time", "lat", "lon"],
                (10.0 + np.arange(time.size, dtype=np.float32)).reshape(time.size, 1, 1),
            ),
            "oceTAUY": (
                ["time", "lat", "lon"],
                (20.0 + np.arange(time.size, dtype=np.float32)).reshape(time.size, 1, 1),
            ),
            "oceQnet": (
                ["time", "lat", "lon"],
                (30.0 + np.arange(time.size, dtype=np.float32)).reshape(time.size, 1, 1),
            ),
            "wetmask": (
                ["lev", "lat", "lon"],
                np.ones((levels, 1, 1), dtype=bool),
            ),
        },
        coords={"time": time, "lev": lev, "lat": lat, "lon": lon},
    )
    compact_means = compact_data.drop_vars("wetmask").mean("time", keep_attrs=True)
    compact_stds = xr.zeros_like(compact_means, dtype=np.float32) + 1.0

    flat_data = compact_data.drop_vars("wetmask").copy()
    for var_name in ["U", "V", "Theta", "Salt"]:
        data_array = flat_data[var_name]
        for level_index in range(levels):
            flat_data[f"{var_name}_{level_index}"] = data_array.isel(lev=level_index)
        flat_data = flat_data.drop_vars(var_name)
    flat_data["wetmask"] = compact_data["wetmask"]

    flat_means = compact_means.drop_vars(["U", "V", "Theta", "Salt"]).copy()
    flat_stds = compact_stds.drop_vars(["U", "V", "Theta", "Salt"]).copy()
    for var_name in ["U", "V", "Theta", "Salt"]:
        data_array = compact_means[var_name]
        std_array = compact_stds[var_name]
        for level_index in range(levels):
            flat_means[f"{var_name}_{level_index}"] = data_array.isel(lev=level_index)
            flat_stds[f"{var_name}_{level_index}"] = std_array.isel(lev=level_index)

    prognostic = PROGNOSTIC_VARS["all"]
    boundary = BOUNDARY_VARS["all"]

    def channel_data(ds: xr.Dataset, channel_name: str) -> xr.DataArray:
        if channel_name in ds:
            return ds[channel_name]
        base_name, level_index = channel_name.rsplit("_", 1)
        return ds[base_name].isel(lev=int(level_index), drop=True)

    packed_prognostic = xr.concat(
        [channel_data(compact_data, name) for name in prognostic],
        dim="prognostic_channel",
    ).transpose("time", "prognostic_channel", "lat", "lon")
    packed_boundary = xr.concat(
        [channel_data(compact_data, name) for name in boundary],
        dim="boundary_channel",
    ).transpose("time", "boundary_channel", "lat", "lon")
    packed_prognostic_mean = xr.DataArray(
        np.asarray(
            [channel_data(compact_means, name).item() for name in prognostic],
            dtype=np.float32,
        ),
        dims=("prognostic_channel",),
    )
    packed_prognostic_std = xr.DataArray(
        np.asarray(
            [channel_data(compact_stds, name).item() for name in prognostic],
            dtype=np.float32,
        ),
        dims=("prognostic_channel",),
    )
    packed_boundary_mean = xr.DataArray(
        np.asarray(
            [channel_data(compact_means, name).item() for name in boundary],
            dtype=np.float32,
        ),
        dims=("boundary_channel",),
    )
    packed_boundary_std = xr.DataArray(
        np.asarray(
            [channel_data(compact_stds, name).item() for name in boundary],
            dtype=np.float32,
        ),
        dims=("boundary_channel",),
    )
    packed_prognostic_mask = xr.DataArray(
        np.stack(
            [
                compact_data["wetmask"].isel(lev=int(name.rsplit("_", 1)[1])).to_numpy()
                if "_" in name
                else compact_data["wetmask"].isel(lev=0).to_numpy()
                for name in prognostic
            ],
            axis=0,
        ),
        dims=("prognostic_channel", "lat", "lon"),
    )
    packed_boundary_mask = xr.DataArray(
        np.stack(
            [compact_data["wetmask"].isel(lev=0).to_numpy() for _ in boundary],
            axis=0,
        ),
        dims=("boundary_channel", "lat", "lon"),
    )
    packed_data = xr.Dataset(
        {
            "prognostic": packed_prognostic,
            "boundary": packed_boundary,
            "prognostic_mean": packed_prognostic_mean,
            "prognostic_std": packed_prognostic_std,
            "boundary_mean": packed_boundary_mean,
            "boundary_std": packed_boundary_std,
            "prognostic_mask": packed_prognostic_mask,
            "boundary_mask": packed_boundary_mask,
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "cache_format": PACKED_CACHE_FORMAT,
            "prognostic_channel_names_json": json.dumps(prognostic),
            "boundary_channel_names_json": json.dumps(boundary),
        },
    )

    packed_src = DataSource.from_packed_dataset(
        packed_data,
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
        name="packed_llc_like",
    )
    flat_src = DataSource.from_datasets(
        flat_data,
        flat_means,
        flat_stds,
        name="flat_llc_like",
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
    )

    packed_dataset = TorchTrainDataset(
        src=packed_src,
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
        hist=1,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=1,
        temporal_stride=1,
    )
    flat_dataset = TorchTrainDataset(
        src=flat_src,
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
        hist=1,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=1,
        temporal_stride=1,
    )

    packed_sample = packed_dataset.to_train_data(
        collate_raw_train_data([packed_dataset[0]])
    )
    flat_sample = flat_dataset.to_train_data(
        collate_raw_train_data([flat_dataset[0]])
    )

    packed_input, packed_label = extract_sample_arrays(packed_sample)
    flat_input, flat_label = extract_sample_arrays(flat_sample)

    np.testing.assert_allclose(packed_input, flat_input)
    np.testing.assert_allclose(packed_label, flat_label)


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
)
@example(
    data_var_index=0,
    lat=np.array([-90.0, 0.0, 90.0]),
    lng=np.array([0.0, 180.0]),
    days_since_start=np.array([5, 10, 15, 20, 25]),
    start_day=datetime.date(2020, 1, 1),
)
@example(
    data_var_index=255,
    lat=np.array([90.00]),
    lng=np.array([360.0]),
    days_since_start=np.array([999]),
    start_day=datetime.date(2000, 5, 1),
)
@example(
    data_var_index=7,
    lat=np.array([0.0]),
    lng=np.array([0.0]),
    days_since_start=np.array([0], dtype=np.uint32),
    start_day=datetime.date(2000, 5, 1),
)
@example(
    lat=np.array([32.87]),
    lng=np.array([0.0]),
    data_var_index=0,
    days_since_start=np.array([0], dtype=np.uint32),
    start_day=datetime.date(2000, 5, 1),
)
@example(
    data_var_index=0,
    lat=np.array([2.0]),
    lng=np.array([1.375]),
    days_since_start=np.array([0], dtype=np.uint32),
    start_day=datetime.date(2000, 1, 1),
)
@settings(deadline=1000)
def test_test_util__data_source_roundtrip(
    data_var_index: int,
    lat: NDArray[np.floating],
    lng: NDArray[np.floating],
    days_since_start: NDArray[np.uint32],
    start_day: datetime.date,
) -> None:
    # We use hour=12 because that's what cftime uses when
    # converting from ordinals (in DataSourceDims)
    start_day_cf = cftime.datetime(
        start_day.year, start_day.month, start_day.day, hour=12, calendar="julian"
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


def test_loader__data_shape(
    train_config: TrainConfig, history: int, loader_version: LoaderVersion
):
    train_config.data.hist = history

    with make_loader(train_config, version=loader_version) as loader:
        exp = train_config.experiment
        batch_size = train_config.batch_size
        num_input_timesteps = history + 1

        input_var_dim = (
            len(PROGNOSTIC_VARS[exp.prognostic_vars_key])
            + len(BOUNDARY_VARS[exp.boundary_vars_key])
        ) * num_input_timesteps
        output_var_dim = (
            len(PROGNOSTIC_VARS[exp.prognostic_vars_key]) * num_input_timesteps
        )

        n_samples = calc_num_samples(train_config, train_config.train_time.time_slice)
        samples = list(loader)

        assert len(samples) == n_samples, (
            f"Current config {train_config} only supports {n_samples} examples; "
            f"got {len(samples)}."
        )

        # Only check the first 2 samples; this should be proof enough that everything is
        # the right shape.
        for sample in samples[:2]:
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


def test_inference__data_shape(inference_loader_pair):
    cfg, loader = inference_loader_pair

    exp = cfg.experiment
    batch_size = 1  # Inference always uses batch size 1
    hist = cfg.data.hist + 1

    input_var_dim = (
        len(PROGNOSTIC_VARS[exp.prognostic_vars_key])
        + len(BOUNDARY_VARS[exp.boundary_vars_key])
    ) * hist
    output_var_dim = len(PROGNOSTIC_VARS[exp.prognostic_vars_key]) * hist

    samples = list(loader)
    assert len(samples) == 1, (
        f"Current config {cfg.inference!r} only supports 1 examples for inference; "
        f"got {len(samples)}."
    )

    for sample in samples:
        inference_dataset, n = sample
        for X, y in inference_dataset:
            assert X.shape == (batch_size, input_var_dim, 180, 360)
            assert y.shape == (batch_size, output_var_dim, 180, 360)


def test__data_is_not_zeros(train_config):
    with make_loader(train_config) as loader:
        for sample in loader:
            X, y = extract_sample_arrays(sample)
            assert np.count_nonzero(np.zeros(X.shape)) == 0, (
                "Sanity check: Zero is zero."
            )
            assert np.count_nonzero(X) != 0, "Input data should not be a zeros matrix!"
            assert np.count_nonzero(y) != 0, "Label data should not be a zeros matrix!"


def test_inference__data_is_not_zero(inference_loader_pair):
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


def assert_equal_samples(original_samples, new_samples):
    for (x_orig, y_orig), (x_new, y_new) in zip(original_samples, new_samples):
        assert x_orig.dtype == x_new.dtype, "Input data types do not match."
        assert y_orig.dtype == y_new.dtype, "Output data types do not match."

        x_not_equal = np.equal(x_orig, x_new) == False  # noqa: E712
        y_not_equal = np.equal(y_orig, y_new) == False  # noqa: E712

        x_not_equal_index = np.where(x_not_equal)
        y_not_equal_index = np.where(y_not_equal)

        assert not np.any(x_not_equal), (
            f"{len(x_not_equal_index[0])} values differ: "
            f"{x_orig[x_not_equal_index]} != {x_new[x_not_equal_index]}."
        )
        assert not np.any(y_not_equal), (
            f"{len(y_not_equal_index[0])} values differ: "
            f"{y_orig[y_not_equal_index]} != {y_new[y_not_equal_index]}."
        )


# Warning: the names/constants used in this test are catered to the implementation
# details of the caches used in `data_source`. For example, this only works for the
# constants "remote-om4" and "compact", which this tests uses to create specific paths
# to a local directory of cached data.
@pytest.mark.parametrize("data_source", ["remote-om4"], indirect=True)
def test_compact_loader__equals_flat_loader(
    data_source: DataSource, pytestconfig: pytest.Config
):
    cache = cache_dir(pytestconfig)
    default_config = str(pytestconfig.rootpath / "configs" / DEFAULT_CONFIG)

    def make_config(src: DataSource):
        return TrainConfig.from_yaml_and_cli(
            [
                default_config,
                "--experiment.data_root",
                str(cache / src.name),
            ]
        )

    flat_config = make_config(data_source)

    # Now, we get the compact data from its local data cache! We can do this just by
    # passing in the correct name. The cache will already have been set up by the test
    # fixture.
    compact_source = dataclasses.replace(data_source, name="compact")
    compact_config = make_config(compact_source)

    with make_loader(flat_config, version=LoaderVersion.OM4_TORCH) as flat_loader:
        original_samples = [extract_sample_arrays(sample) for sample in flat_loader]
    with make_loader(compact_config, version=LoaderVersion.OM4_TORCH) as compact_loader:
        new_samples = [extract_sample_arrays(sample) for sample in compact_loader]

    assert_equal_samples(original_samples, new_samples)


@pytest.fixture
def tiny_dataset_input(normalize_before_mask: bool, masked_fill_value: float):
    # Create data
    coords = {"time": range(10), "lat": range(2), "lon": range(2)}
    times = torch.arange(10)
    data_array = (
        torch.repeat_interleave(times, torch.tensor([2 * 2 * 4]))
        .reshape(10, 4, 2, 2)
        .permute(1, 0, 2, 3)
    )

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
            "prognostic1": 0.5,
            "prognostic2": 0.5,
            "boundary1": 0.5,
            "boundary2": 0.5,
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

    wet_surface = torch.ones(2, 2)
    wet_surface[0, 0] = 0.0
    wet_surface[1, 1] = 0.0
    wet = wet_surface.expand(2, 2, 2)
    masks = Masks(
        prognostic=wet,
        boundary=wet_surface,
    )
    test = DataSource("test", data, data_mean, data_std, masks=masks)

    # Initialize and yield within the MultitonScope
    with MultitonScope():
        _ = Normalize.init_instance(
            test,
            prognostic_var_names=["prognostic1", "prognostic2"],
            boundary_var_names=["boundary1", "boundary2"],
        )
        torch_train_dataset = TorchTrainDataset(
            src=test,
            prognostic_var_names=prognostic_var_names,
            boundary_var_names=boundary_var_names,
            hist=1,
            steps=2,
            normalize_before_mask=normalize_before_mask,
            masked_fill_value=masked_fill_value,
            stride=1,
        )
        inference_dataset = InferenceDataset(
            src=test,
            prognostic_var_names=prognostic_var_names,
            boundary_var_names=boundary_var_names,
            hist=1,
            normalize_before_mask=normalize_before_mask,
            masked_fill_value=masked_fill_value,
            long_rollout=True,
        )

        # Create a TrainDataLoader wrapper
        raw_loader = DataLoader(
            torch_train_dataset,
            batch_size=1,
            collate_fn=collate_raw_train_data,
        )
        train_loader = TrainDataLoader(
            raw_loader, [torch_train_dataset], torch.device("cpu")
        )

        yield train_loader, inference_dataset


@pytest.mark.parametrize("normalize_before_mask", [True, False])
@pytest.mark.parametrize("masked_fill_value", [0.0, -1.0])
def test_train_dataset_no_input_change(
    tiny_dataset_input, normalize_before_mask, masked_fill_value
):
    train_loader, _ = tiny_dataset_input
    td = train_loader[0]
    pred = torch.randn_like(td.get_label(0)) * 0.1

    inp1 = td.get_input(1).clone()
    td.merge_prognostic_and_boundary(pred, 1)

    # Get a fresh copy from the loader
    td_new = train_loader[0]
    assert torch.equal(td_new.get_input(1), inp1)


@pytest.mark.parametrize("normalize_before_mask", [True, False])
@pytest.mark.parametrize("masked_fill_value", [0.0, -1.0])
def test_train_dataset_normalize_pre_fill(
    tiny_dataset_input, normalize_before_mask, masked_fill_value
):
    train_loader, inference_dataset = tiny_dataset_input
    td0 = train_loader[0]
    data = masked_fill_value

    td0_step0_input = td0.get_input(0)
    td0_step0_label = td0.get_label(0)
    inf_step0_input, inf_step0_label = inference_dataset[0]

    assert td0_step0_input.shape == (1, 8, 2, 2)
    assert td0_step0_label.shape == (1, 4, 2, 2)
    assert inf_step0_input.shape == (1, 8, 2, 2)
    assert inf_step0_label.shape == (1, 4, 2, 2)

    # We expect [0,0,0] to be masked
    if normalize_before_mask:
        assert td0.get_input(0)[0, 0, 0, 0] == data
        assert inference_dataset[0][0][0][0, 0, 0] == data
    else:
        mean = 0.5
        std = 1.0
        data = (data - mean) / std
        assert td0.get_input(0)[0, 0, 0, 0] == data
        assert inference_dataset[0][0][0][0, 0, 0] == data


@pytest.mark.manual
@pytest.mark.parametrize(
    "data_source,config_name", [("mock", DEFAULT_CONFIG)], indirect=True
)
def test_profile__loader__1gb(train_config, loader_version, benchmark):
    cfg = train_config

    with make_loader(cfg, version=loader_version) as loader:

        @benchmark
        def bench():
            indices = np.random.randint(0, len(loader), size=len(loader))
            for idx in indices:
                _ = loader.dataset[int(idx)]


@pytest.mark.manual
@pytest.mark.parametrize(
    "data_source,config_name", [("mock", DEFAULT_CONFIG)], indirect=True
)
def test_profile__inference_loader__1gb(inference_loader_pair, benchmark):
    cfg, loader = inference_loader_pair

    @benchmark
    def bench():
        for sample in loader:
            dataset, n = sample
            for X, y in dataset:
                _, _ = X, y


def _spatial_feature_source(features: torch.Tensor | None) -> tuple[DataSource, list, list]:
    """A 2x2, 6-timestep source, optionally carrying fixed geographic features."""
    coords = {"time": range(6), "lat": range(2), "lon": range(2)}
    # Encode the time index in the values so channel identity is checkable.
    values = torch.arange(6 * 4 * 2 * 2, dtype=torch.float32).reshape(4, 6, 2, 2)
    names = ["prognostic1", "prognostic2", "boundary1", "boundary2"]
    data = xr.Dataset(
        {
            name: xr.DataArray(values[i], dims=["time", "lat", "lon"], coords=coords)
            for i, name in enumerate(names)
        }
    )
    stats_coords = {"lat": [0], "lon": [0]}
    data_mean = xr.Dataset({name: 0.0 for name in names}, coords=stats_coords)
    data_std = xr.Dataset({name: 1.0 for name in names}, coords=stats_coords)
    masks = Masks(prognostic=torch.ones(2, 2, 2), boundary=torch.ones(2, 2))
    src = DataSource(
        "spatial-test",
        data,
        data_mean,
        data_std,
        masks=masks,
        spatial_features=features,
    )
    return src, ["prognostic1", "prognostic2"], ["boundary1", "boundary2"]


def _make_inference_dataset(src, prognostic, boundary, *, append: bool):
    return InferenceDataset(
        src=src,
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
        hist=0,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        long_rollout=False,
        append_spatial_features_to_inputs=append,
    )


def test_inference_dataset_appends_spatial_features_to_both_input_paths() -> None:
    features = torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2)
    src, prognostic, boundary = _spatial_feature_source(features)

    with MultitonScope():
        Normalize.init_instance(
            src, prognostic_var_names=prognostic, boundary_var_names=boundary
        )
        plain = _make_inference_dataset(src, prognostic, boundary, append=False)
        augmented = _make_inference_dataset(src, prognostic, boundary, append=True)

        # __getitem__ is the path behind get_initial_input().
        plain_input, _ = plain[0]
        augmented_input, _ = augmented[0]
        assert plain_input.shape[1] == 4
        assert augmented_input.shape[1] == 4 + 4
        torch.testing.assert_close(augmented_input[:, :4], plain_input)
        torch.testing.assert_close(augmented_input[0, 4:], features)

        # merge_prognostic_and_boundary is the path used every rollout step.
        prognostic_state = torch.zeros(1, 2, 2, 2)
        plain_merged = plain.merge_prognostic_and_boundary(prognostic_state, step=0)
        augmented_merged = augmented.merge_prognostic_and_boundary(
            prognostic_state, step=0
        )
        assert augmented_merged.shape[1] == plain_merged.shape[1] + 4
        torch.testing.assert_close(
            augmented_merged[:, : plain_merged.shape[1]], plain_merged
        )
        torch.testing.assert_close(augmented_merged[0, plain_merged.shape[1] :], features)


def test_inference_and_train_datasets_append_spatial_features_identically() -> None:
    features = torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2)
    src, prognostic, boundary = _spatial_feature_source(features)

    with MultitonScope():
        Normalize.init_instance(
            src, prognostic_var_names=prognostic, boundary_var_names=boundary
        )
        inference = _make_inference_dataset(src, prognostic, boundary, append=True)
        train = TorchTrainDataset(
            src=src,
            prognostic_var_names=prognostic,
            boundary_var_names=boundary,
            hist=0,
            steps=1,
            normalize_before_mask=True,
            masked_fill_value=0.0,
            stride=1,
            temporal_stride=1,
            append_spatial_features_to_inputs=True,
        )

        base = torch.zeros(1, 3, 2, 2)
        torch.testing.assert_close(
            inference.append_static_channels(base),
            train.append_static_channels(base),
        )


def test_inference_dataset_without_spatial_features_leaves_inputs_untouched() -> None:
    src, prognostic, boundary = _spatial_feature_source(None)

    with MultitonScope():
        Normalize.init_instance(
            src, prognostic_var_names=prognostic, boundary_var_names=boundary
        )
        dataset = _make_inference_dataset(src, prognostic, boundary, append=False)
        assert dataset[0][0].shape[1] == 4


def test_inference_dataset_rejects_append_without_spatial_features() -> None:
    src, prognostic, boundary = _spatial_feature_source(None)

    with MultitonScope():
        Normalize.init_instance(
            src, prognostic_var_names=prognostic, boundary_var_names=boundary
        )
        with pytest.raises(ValueError, match="XC, YC, and rA"):
            _make_inference_dataset(src, prognostic, boundary, append=True)


def test_append_spatial_features_rejects_shape_mismatch() -> None:
    features = torch.zeros(4, 3, 3)
    src, prognostic, boundary = _spatial_feature_source(features)

    with MultitonScope():
        Normalize.init_instance(
            src, prognostic_var_names=prognostic, boundary_var_names=boundary
        )
        dataset = _make_inference_dataset(src, prognostic, boundary, append=True)
        with pytest.raises(ValueError, match="Static channel shape"):
            dataset.append_static_channels(torch.zeros(1, 4, 2, 2))


# ---------------------------------------------------------------------------
# InferenceDataset read path
#
# The rollout reads through this class once per tile per step, so how it
# addresses the time axis dominates autoregressive validation and inference. A
# 2-D indexer over time costs seconds per call: xarray broadcasts it against the
# whole (channel, lat, lon) shape and zarr falls back to point selection. These
# pin the read down to basic slices covering exactly the requested steps.
# ---------------------------------------------------------------------------

PACKED_PROGNOSTIC = ["U_0", "U_1", "Theta_0"]
PACKED_BOUNDARY = ["oceTAUX", "oceQnet"]


def _packed_float16_source(*, timesteps: int = 24, lats: int = 2, lons: int = 3):
    """A packed train-ready source shaped like the real caches: float16 throughout.

    The caches store data, means, and stds as float16, which is what makes the
    normalization dtype observable.
    """
    rng = np.random.default_rng(0)
    shape = (timesteps, len(PACKED_PROGNOSTIC), lats, lons)
    prognostic = (rng.normal(3.0, 7.0, shape)).astype(np.float16)
    boundary = (
        rng.normal(-2.0, 5.0, (timesteps, len(PACKED_BOUNDARY), lats, lons))
    ).astype(np.float16)
    packed = xr.Dataset(
        {
            "prognostic": (
                ["time", "prognostic_channel", "lat", "lon"],
                prognostic,
            ),
            "boundary": (["time", "boundary_channel", "lat", "lon"], boundary),
            "prognostic_mean": (
                ["prognostic_channel"],
                np.array([0.3, -1.7, 2.9], dtype=np.float16),
            ),
            "prognostic_std": (
                ["prognostic_channel"],
                np.array([1.3, 0.7, 3.1], dtype=np.float16),
            ),
            "boundary_mean": (
                ["boundary_channel"],
                np.array([0.9, -0.4], dtype=np.float16),
            ),
            "boundary_std": (
                ["boundary_channel"],
                np.array([2.1, 0.6], dtype=np.float16),
            ),
            "prognostic_mask": (
                ["prognostic_channel", "lat", "lon"],
                np.ones((len(PACKED_PROGNOSTIC), lats, lons), dtype=bool),
            ),
            "boundary_mask": (
                ["boundary_channel", "lat", "lon"],
                np.ones((len(PACKED_BOUNDARY), lats, lons), dtype=bool),
            ),
        },
        coords={
            "time": np.arange(timesteps),
            "lat": np.arange(lats, dtype=np.float32),
            "lon": np.arange(lons, dtype=np.float32),
        },
        attrs={
            "cache_format": PACKED_CACHE_FORMAT,
            "prognostic_channel_names_json": json.dumps(PACKED_PROGNOSTIC),
            "boundary_channel_names_json": json.dumps(PACKED_BOUNDARY),
        },
    )
    return DataSource.from_packed_dataset(
        packed,
        prognostic_var_names=PACKED_PROGNOSTIC,
        boundary_var_names=PACKED_BOUNDARY,
        name="packed_f16",
    )


@contextlib.contextmanager
def _packed_inference_dataset(*, hist: int = 0, normalize_before_mask: bool = True):
    src = _packed_float16_source()
    with MultitonScope():
        Normalize.init_instance(
            src,
            prognostic_var_names=PACKED_PROGNOSTIC,
            boundary_var_names=PACKED_BOUNDARY,
        )
        yield InferenceDataset(
            src=src,
            prognostic_var_names=PACKED_PROGNOSTIC,
            boundary_var_names=PACKED_BOUNDARY,
            hist=hist,
            normalize_before_mask=normalize_before_mask,
            masked_fill_value=0.0,
            long_rollout=False,
        )


@contextlib.contextmanager
def _recorded_time_indexers(monkeypatch):
    """Capture every `time` indexer handed to `Dataset.isel`."""
    seen: list[object] = []
    original = xr.Dataset.isel

    def spy(self, indexers=None, **kwargs):
        merged = dict(indexers or {})
        merged.update(
            {
                key: value
                for key, value in kwargs.items()
                if key not in {"drop", "missing_dims"}
            }
        )
        if "time" in merged:
            seen.append(merged["time"])
        return original(self, indexers, **kwargs)

    monkeypatch.setattr(xr.Dataset, "isel", spy)
    yield seen
    monkeypatch.undo()


def _vectorized_reference(dataset: InferenceDataset, idx: int, field: str):
    """The pre-rewrite read: a 2-D time indexer, normalized inside xarray.

    Kept here rather than in the library so the rewrite has something to be
    equivalent to. It is also what makes the float16 rounding visible: the
    arithmetic below runs in the dtype of the stored data.
    """
    rolling = dataset.rolling_indices.isel(window_dim=slice(idx, idx + 1))
    x_index = xr.Variable(["window_dim", "time"], rolling)
    hist = dataset.hist
    if field == "boundary":
        source, mask, steps = dataset._boundary_src, dataset.wet_surface, slice(
            None, hist + 1
        )
    else:
        source, mask = dataset._prognostic_src, dataset.wet
        steps = slice(hist + 1, None) if field == "label" else slice(None, hist + 1)

    selected = source.map_data(
        lambda ds: ds.isel(time=x_index).isel(time=steps),
        suffix=f"reference_{field}",
    )
    array = _dataset_to_numpy(selected.normalize(), ("window_dim", "time"))
    tensor = torch.from_numpy(array).float()
    tensor = torch.where(mask, tensor, dataset.masked_fill_value)
    return einops_rearrange(
        tensor,
        "window_dim time variable lat lon -> window_dim (time variable) lat lon",
    )


def _float64_reference(dataset: InferenceDataset, idx: int, field: str):
    """The same read in float64, to score both paths against."""
    hist = dataset.hist
    window = dataset._windows[idx]
    if field == "boundary":
        source, times = dataset._boundary_src, window[: hist + 1]
        mean_var, std_var = "boundary_mean", "boundary_std"
    else:
        source = dataset._prognostic_src
        times = window[hist + 1 :] if field == "label" else window[: hist + 1]
        mean_var, std_var = "prognostic_mean", "prognostic_std"

    raw = source.data.isel(time=slice(int(times[0]), int(times[-1]) + 1))
    values = np.asarray(next(iter(raw.data_vars.values())).to_numpy(), dtype=np.float64)
    mean = np.asarray(source.means[mean_var].to_numpy(), dtype=np.float64)
    std = np.asarray(source.stds[std_var].to_numpy(), dtype=np.float64)
    normalized = (values - mean[None, :, None, None]) / std[None, :, None, None]
    return torch.from_numpy(normalized.reshape(1, -1, *values.shape[-2:]))


@pytest.mark.parametrize("hist", [0, 1, 2])
def test_inference_dataset_reads_only_basic_time_slices(monkeypatch, hist) -> None:
    """The guard on the rewrite: no indexer over time may exceed one dimension.

    A `[window, step]` table handed to `isel` is a 2-D indexer, which is what
    made a single rollout step cost seconds per tile.
    """
    with _packed_inference_dataset(hist=hist) as dataset:
        with _recorded_time_indexers(monkeypatch) as seen:
            dataset.rollout_boundary_and_target(3)
            dataset[3]
            dataset.initial_prognostic
            dataset.inference_target(slice(2, 5))

        assert seen, "expected the read path to index the time axis"
        for indexer in seen:
            assert isinstance(indexer, slice) or np.ndim(indexer) == 0, (
                f"time was indexed with {indexer!r}; basic slices only"
            )


@pytest.mark.parametrize("hist", [0, 1, 2])
def test_inference_dataset_reads_each_timestep_once(monkeypatch, hist) -> None:
    """One rollout step must not read the steps it is about to discard.

    The old path selected the whole `2 * (hist + 1)` window and sliced it down
    afterwards, so every rollout step paid for twice the timesteps it used.
    """
    with _packed_inference_dataset(hist=hist) as dataset:
        with _recorded_time_indexers(monkeypatch) as seen:
            dataset.rollout_boundary_and_target(3)

        spans = [indexer for indexer in seen if isinstance(indexer, slice)]
        assert len(spans) == 2, "expected one read for boundary and one for truth"
        for span in spans:
            assert span.stop - span.start == hist + 1


@pytest.mark.parametrize("hist", [0, 1, 2])
@pytest.mark.parametrize("field", ["prognostic", "boundary", "label"])
def test_inference_dataset_matches_vectorized_reference(hist, field) -> None:
    """Same values as the path it replaces, to float16 tolerance.

    The residual difference is the old path's own rounding: it normalized inside
    xarray, so the arithmetic ran in the stored float16 rather than float32.
    """
    with _packed_inference_dataset(hist=hist) as dataset:
        readers = {
            "prognostic": dataset._get_prognostic,
            "boundary": dataset._get_boundary,
            "label": dataset._get_label,
        }
        actual = readers[field](4)
        reference = _vectorized_reference(dataset, 4, field)

        assert actual.shape == reference.shape
        torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("field", ["prognostic", "boundary", "label"])
def test_inference_dataset_normalizes_in_float32(field) -> None:
    """Normalizing in torch is not just faster, it is the more accurate path.

    Both are scored against the same read done in float64. The old path cannot
    beat float16 spacing; this one should be exact to float32.
    """
    with _packed_inference_dataset(hist=0) as dataset:
        readers = {
            "prognostic": dataset._get_prognostic,
            "boundary": dataset._get_boundary,
            "label": dataset._get_label,
        }
        exact = _float64_reference(dataset, 4, field)
        new_error = (readers[field](4).double() - exact).abs().max().item()
        old_error = (
            (_vectorized_reference(dataset, 4, field).double() - exact)
            .abs()
            .max()
            .item()
        )

        assert new_error < 1e-6, f"float32 normalization should be exact, got {new_error}"
        assert old_error > new_error, (
            "expected the float16 xarray path to be the less accurate one "
            f"(new={new_error}, old={old_error})"
        )


@pytest.mark.parametrize("hist", [0, 1, 2])
def test_inference_dataset_windows_agree_with_rolling_indices(hist) -> None:
    """`_windows` is the integer view of `rolling_indices`, and rows are contiguous."""
    with _packed_inference_dataset(hist=hist) as dataset:
        np.testing.assert_array_equal(
            dataset._windows, dataset.rolling_indices.to_numpy()
        )
        assert dataset._windows.shape[1] == 2 * (hist + 1)
        for row in dataset._windows:
            np.testing.assert_array_equal(row, np.arange(row[0], row[0] + row.size))

        # Consecutive windows abut, which is what lets a window slice stay one read.
        inputs = dataset._windows_for(slice(0, 3))[:, : hist + 1]
        assert dataset._contiguous_span(inputs) == (
            int(inputs[0][0]),
            int(inputs[0][0]) + inputs.size,
        )


def test_inference_target_slice_matches_per_step_labels() -> None:
    """`BaseModel.inference` asks for a whole chunk of targets in one call."""
    with _packed_inference_dataset(hist=1) as dataset:
        batched = dataset.inference_target(slice(2, 6))
        per_step = torch.cat(
            [dataset.inference_target(step) for step in range(2, 6)], dim=0
        )
        torch.testing.assert_close(batched, per_step)


def test_inference_dataset_rejects_strided_window_slices() -> None:
    """A strided request is not contiguous, so it cannot be one basic slice."""
    with _packed_inference_dataset(hist=0) as dataset:
        with pytest.raises(ValueError, match="not contiguous"):
            dataset.inference_target(slice(0, 6, 2))


@pytest.mark.parametrize("normalize_before_mask", [True, False])
def test_inference_dataset_preserves_mask_and_normalize_order(
    normalize_before_mask,
) -> None:
    """The two orderings fill land differently, and the rewrite keeps both.

    Masking after normalizing writes the fill value straight into normalized
    space; masking first fills in physical units and then normalizes that, so
    land ends up at `(fill - mean) / std`. Either way land is one constant per
    channel and the live cell is untouched.
    """
    src = _packed_float16_source()
    land = torch.zeros_like(src.masks.prognostic, dtype=torch.bool)
    land[:, 0, 0] = True  # exactly one live cell
    src = dataclasses.replace(
        src, masks=dataclasses.replace(src.masks, prognostic=land)
    )
    with MultitonScope():
        Normalize.init_instance(
            src,
            prognostic_var_names=PACKED_PROGNOSTIC,
            boundary_var_names=PACKED_BOUNDARY,
        )
        dataset = InferenceDataset(
            src=src,
            prognostic_var_names=PACKED_PROGNOSTIC,
            boundary_var_names=PACKED_BOUNDARY,
            hist=0,
            normalize_before_mask=normalize_before_mask,
            masked_fill_value=0.0,
            long_rollout=False,
        )
        label = dataset._get_label(4)
        mean = torch.from_numpy(
            src.means["prognostic_mean"].to_numpy().astype(np.float32)
        )
        std = torch.from_numpy(src.stds["prognostic_std"].to_numpy().astype(np.float32))
        expected_land = (
            torch.zeros_like(mean) if normalize_before_mask else (0.0 - mean) / std
        )

        land_cells = label[0, :, 0, 1:]  # every cell but the one live one
        torch.testing.assert_close(
            land_cells, expected_land.unsqueeze(1).expand_as(land_cells)
        )
        live = label[0, :, 0, 0]
        assert not torch.allclose(live, expected_land), "the live cell was masked"
