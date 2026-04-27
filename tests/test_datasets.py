"""Test core Datasets and DataLoaders."""

import contextlib
import dataclasses
import datetime
import itertools
from collections.abc import Generator, Iterable
from typing import assert_never

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

from ocean_emulators.config import TimeConfig, TrainConfig, TrainSchedule
from ocean_emulators.constants import (
    BOUNDARY_VARS,
    PROGNOSTIC_VARS,
    LoaderVersion,
    TensorMap,
)
from ocean_emulators.datasets import (
    InferenceDataset,
    TorchTrainDataset,
    TrainData,
    TrainDataLoader,
)
from ocean_emulators.utils.data import DataSource, Masks, Normalize
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.samplers import EquivalenceGroupBatchSampler
from ocean_emulators.utils.train import collate_raw_train_data
from tests.conftest import DEFAULT_CONFIG, DataSourceDims, TrainPair, cache_dir


@pytest.fixture
def inference_loader_pair(trainer_pair: TrainPair) -> tuple[TrainConfig, DataLoader]:
    cfg, trainer = trainer_pair
    return cfg, trainer.inference_loader


def coarsen_data(ds: xr.Dataset) -> xr.Dataset:
    return ds.coarsen(lat=2, lon=2).mean()  # type: ignore


def coarsen_masks(masks: Masks) -> Masks:
    """Coarsen masks to half resolution using max pooling (any True -> True)."""
    # For masks, we use max pooling: if any cell in the 2x2 block is True (valid),
    # the coarsened cell is True
    import torch.nn.functional as F

    # Coarsen prognostic mask (3D: channels x lat x lon)
    prog_mask = masks.prognostic.float().unsqueeze(0)  # Add batch dim
    prog_coarsened = F.max_pool2d(prog_mask, kernel_size=2, stride=2)
    prog_coarsened = prog_coarsened.squeeze(0).bool()  # Remove batch dim, back to bool

    # Coarsen boundary mask (2D: lat x lon)
    bound_mask = (
        masks.boundary.float().unsqueeze(0).unsqueeze(0)
    )  # Add batch and channel dims
    bound_coarsened = F.max_pool2d(bound_mask, kernel_size=2, stride=2)
    bound_coarsened = bound_coarsened.squeeze(0).squeeze(0).bool()  # Remove extra dims

    return Masks(prognostic=prog_coarsened, boundary=bound_coarsened)


@contextlib.contextmanager
def make_loader(
    cfg: TrainConfig,
    time_config: TimeConfig | None = None,
    drop_last: bool = True,
    version: LoaderVersion | None = None,
    schedule: TrainSchedule = "standard",
    shuffle: bool = True,
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
        cfg.experiment.resolved_data_root,
        prognostic,
        boundary,
    )
    version = container.loader_version
    src = container.primary_source
    if src.is_compact and version != LoaderVersion.OM4_TORCH:
        pytest.skip(f"{version} does not support compact data.")

    with MultitonScope():
        TensorMap.init_instance(
            cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key
        )

        match schedule:
            case "standard":
                srcs: Iterable[tuple[DataSource, DataSource | None]] = [(src, None)]
            case "match":
                coarsened_src = src.map_data(coarsen_data, suffix="half-size")
                coarsened_src = dataclasses.replace(
                    coarsened_src, masks=coarsen_masks(src.masks)
                )
                scales = [src, coarsened_src]
                srcs = [(s, s) for s in scales]
            case "mix":
                coarsened_src = src.map_data(coarsen_data, suffix="half-size")
                coarsened_src = dataclasses.replace(
                    coarsened_src, masks=coarsen_masks(src.masks)
                )
                scales = [src, coarsened_src]
                srcs = list(itertools.product(scales, repeat=2))  # type: ignore
            case _:
                assert_never(schedule)

        match version:
            case LoaderVersion.OM4_TORCH:
                dataset_list = [
                    TorchTrainDataset(
                        src=src.slice(time_config),
                        dst=dst.slice(time_config) if dst else None,
                        prognostic_var_names=prognostic,
                        boundary_var_names=boundary,
                        hist=cfg.data.hist,
                        steps=cfg.steps[0],
                        normalize_before_mask=cfg.data.normalize_before_mask,
                        masked_fill_value=cfg.data.masked_fill_value,
                        stride=stride,
                    )
                    for src, dst in srcs
                    for stride in cfg.data_stride
                ]

                data: ConcatDataset = ConcatDataset(dataset_list)
                collate_fn = collate_raw_train_data

                # Group datasets by input AND label resolution, allowing different strides to batch together
                # This ensures datasets with same (src, dst) resolution pair but different strides can batch
                batch_sampler = EquivalenceGroupBatchSampler.from_datasets(
                    datasets=dataset_list,
                    group_key=lambda ds: tuple(
                        prog.grid_size for prog in ds.prognostic_srcs
                    ),
                    batch_size=cfg.batch_size,
                    drop_last=drop_last,
                    shuffle=shuffle,
                )

                raw_loader = DataLoader(
                    data,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_fn,
                )

                loader = TrainDataLoader(raw_loader, dataset_list, torch.device("cpu"))
                yield loader
            case _:
                raise ValueError(f"Unknown loader version: {version}")


def extract_sample_arrays(td: TrainData) -> tuple[np.ndarray, np.ndarray]:
    """Extract underlying X, y pairs from TrainData object.

    X is the channel-concatenated (prognostic + boundary) tensor for parity
    with the pre-split-API shape checks these tests do.
    """
    steps = len(td)
    x_arrays = []
    for s in range(steps):
        prog, boundary = td.get_input(s)
        x_arrays.append(torch.cat((prog, boundary), dim=1).numpy(force=True))
    y_arrays = [td.get_label(s).numpy(force=True) for s in range(steps)]

    return np.stack(x_arrays, axis=0), np.stack(y_arrays, axis=0)


def calc_num_samples(
    cfg: TrainConfig, time_slice: slice, schedule: TrainSchedule
) -> int:
    primary = cfg.data.sources[0]
    ds = cfg.experiment.resolved_data_root.resolve(primary.data_location).open()

    data_size = ds.sel(time=time_slice).time.size
    steps = cfg.steps[0]
    hist = cfg.data.hist
    stride = cfg.data_stride[0]

    n_samples = data_size - (steps * (cfg.data.hist + 1) * stride) - hist * stride
    if schedule == "match":
        n_samples *= 2
    if schedule == "mix":
        n_samples *= 4

    return n_samples


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

        n_samples = calc_num_samples(
            train_config, train_config.train_time.time_slice, "standard"
        )
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


@pytest.mark.parametrize(
    "data_source,config_name", [("mock", DEFAULT_CONFIG)], indirect=True
)
def test_loader__data_shape__across_schedules(
    train_config: TrainConfig, schedule: TrainSchedule
):
    history = train_config.data.hist

    with make_loader(
        train_config, version=LoaderVersion.OM4_TORCH, schedule=schedule
    ) as loader:
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

        n_samples = calc_num_samples(
            train_config, train_config.train_time.time_slice, schedule
        )
        samples = list(loader)

        assert len(samples) == n_samples, (
            f"Current config {train_config} only supports {n_samples} examples; "
            f"got {len(samples)}."
        )

        example_resolutions = []
        # Subsample the examples; this should be proof enough that everything is the right shape.
        for sample in samples[::3]:
            X, y = extract_sample_arrays(sample)
            # Exclude the coordinate shape information for now; we'll test that separately.
            assert X.shape[:-2] == (
                train_config.steps[0],
                batch_size,
                input_var_dim,
            )
            assert y.shape[:-2] == (
                train_config.steps[0],
                batch_size,
                output_var_dim,
            )
            example_resolutions.append((X.shape[-2:], y.shape[-2:]))

        match schedule:
            case "standard":
                assert example_resolutions[0][0] == example_resolutions[0][1], (
                    "The input and output should be equal"
                )
                assert all(
                    example_resolutions[0] == eg for eg in example_resolutions[1:]
                ), "All resolutions should be equal"
            case "match":
                for x_res, y_res in example_resolutions:
                    assert x_res == y_res, (
                        f"Resolutions must match across batches for 'match' schedule multiscale loader. {example_resolutions=}"
                    )
            case "mix":
                # In mix mode with 2 scales, multiplex creates pattern: (0,0), (0,1), (1,0), (1,1)
                # With grouped batch sampler, order may vary due to shuffling within groups.
                # With drop_last=True and small sample counts, some groups might not produce any batches
                valid_patterns = {
                    ((180, 360), (180, 360)),  # (0,0): full-res input, full-res label
                    ((180, 360), (90, 180)),  # (0,1): full-res input, half-res label
                    ((90, 180), (180, 360)),  # (1,0): half-res input, full-res label
                    ((90, 180), (90, 180)),  # (1,1): half-res input, half-res label
                }
                observed_patterns = set(example_resolutions)
                # All observed patterns must be valid
                assert observed_patterns == valid_patterns, (
                    f"All resolutions must be valid members of the cartesian product for 'mix' schedule. "
                    f"Valid patterns: {valid_patterns}, got {observed_patterns}, "
                    f"invalid patterns: {observed_patterns - valid_patterns}"
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
        for prog, boundary, y in inference_dataset:
            X = torch.cat((prog, boundary), dim=1)
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
        for prog, boundary, y in dataset:
            X = torch.cat((prog, boundary), dim=1)
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
# constants "mock-om4" and "compact", which this tests uses to create specific paths
# to a local directory of cached data.
@pytest.mark.parametrize("data_source", ["mock-om4"], indirect=True)
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

    with make_loader(
        flat_config, version=LoaderVersion.OM4_TORCH, shuffle=False
    ) as flat_loader:
        original_samples = [extract_sample_arrays(sample) for sample in flat_loader]
    with make_loader(
        compact_config, version=LoaderVersion.OM4_TORCH, shuffle=False
    ) as compact_loader:
        new_samples = [extract_sample_arrays(sample) for sample in compact_loader]

    assert_equal_samples(original_samples, new_samples)


@pytest.mark.parametrize("data_source", ["mock-om4"], indirect=True)
def test_mixed_schedule__has_consistent_collated_batches(
    train_config: TrainConfig, schedule: TrainSchedule
):
    # Exposes underling consistency issue
    train_config.batch_size = 4

    with make_loader(train_config, schedule=schedule) as loader:
        for _ in itertools.islice(loader, 2):
            pass


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
            dst=None,
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

    prog1, bnd1 = td.get_input(1)
    prog1_orig, bnd1_orig = prog1.clone(), bnd1.clone()

    # Get a fresh copy from the loader
    td_new = train_loader[0]
    prog1_new, bnd1_new = td_new.get_input(1)
    assert torch.equal(prog1_new, prog1_orig)
    assert torch.equal(bnd1_new, bnd1_orig)


@pytest.mark.parametrize("normalize_before_mask", [True, False])
@pytest.mark.parametrize("masked_fill_value", [0.0, -1.0])
def test_train_dataset_normalize_pre_fill(
    tiny_dataset_input, normalize_before_mask, masked_fill_value
):
    train_loader, inference_dataset = tiny_dataset_input
    td0 = train_loader[0]
    data = masked_fill_value

    td0_prog, td0_boundary = td0.get_input(0)
    td0_step0_label = td0.get_label(0)
    inf_prog, inf_boundary, inf_step0_label = inference_dataset[0]

    # Prog and boundary each carry (hist+1)*2 channels over a 2x2 grid.
    assert td0_prog.shape == (1, 4, 2, 2)
    assert td0_boundary.shape == (1, 4, 2, 2)
    assert td0_step0_label.shape == (1, 4, 2, 2)
    assert inf_prog.shape == (1, 4, 2, 2)
    assert inf_boundary.shape == (1, 4, 2, 2)
    assert inf_step0_label.shape == (1, 4, 2, 2)

    # We expect [0,0,0] to be masked in the prognostic stream.
    if normalize_before_mask:
        assert td0_prog[0, 0, 0, 0] == data
        assert inf_prog[0, 0, 0, 0] == data
    else:
        mean = 0.5
        std = 1.0
        data = (data - mean) / std
        assert td0_prog[0, 0, 0, 0] == data
        assert inf_prog[0, 0, 0, 0] == data


@pytest.mark.manual
@pytest.mark.parametrize(
    "data_source,config_name", [("mock", DEFAULT_CONFIG)], indirect=True
)
def test_profile__loader__1gb(train_config, loader_version, benchmark):
    cfg = train_config

    with make_loader(cfg, version=loader_version) as loader:
        indices = np.random.default_rng(0).integers(0, len(loader), size=len(loader))

        def bench():
            for idx in indices:
                _ = loader.dataset[int(idx)]

        # Warm the restored Zarr cache before timing steady-state loader throughput.
        bench()
        benchmark(bench)


@pytest.mark.manual
@pytest.mark.parametrize(
    "data_source,config_name", [("mock", DEFAULT_CONFIG)], indirect=True
)
def test_profile__inference_loader__1gb(inference_loader_pair, benchmark):
    cfg, loader = inference_loader_pair

    def bench():
        for sample in loader:
            dataset, n = sample
            for X, y in dataset:
                _, _ = X, y

    # Warm the restored Zarr cache before timing steady-state loader throughput.
    bench()
    benchmark(bench)
