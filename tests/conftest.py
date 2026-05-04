# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import pathlib
from collections.abc import Generator, Iterable
from typing import ClassVar, Self

import cftime
import filelock
import numpy as np
import pytest
import torch
import xarray as xr
from numpy.typing import ArrayLike, NDArray

import ocean_emulators.constants as c
from ocean_emulators.config import (
    JulianDate,
    TrainBackendConfig,
    TrainConfig,
    TrainSchedule,
)
from ocean_emulators.train import Trainer
from ocean_emulators.utils.data import DataSource, Masks, _is_compact, compact_dataset
from ocean_emulators.utils.multiton import MultitonScope

REMOTE_DATA = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/"
DEFAULT_CONFIG = "test/train_default.yaml"
FOMO_CONFIG = "test/train_fomo.yaml"
ALL_CONFIGS = [DEFAULT_CONFIG, "test/train_default_2step.yaml", FOMO_CONFIG]
TEST_DATASET_SPEC = c.build_om4_spec(
    prognostic_vars_key="thetao_1",
    boundary_vars_key="hfds",
)
TEST_FULL_DATASET_SPEC = c.build_om4_spec(
    prognostic_vars_key="thermo_dynamic_all",
    boundary_vars_key="tau_hfds_hfds_anom",
)


def _om4_canonical_var_names(var_names: Iterable[str]) -> list[str]:
    out = []
    for var_name in var_names:
        if "_lev_" not in var_name:
            out.append(var_name)
            continue

        base_name, lev_depth = var_name.split("_lev_", maxsplit=1)
        depth_level = float(lev_depth.replace("_", "."))
        out.append(f"{base_name}_{TEST_DATASET_SPEC.depth_levels.index(depth_level)}")
    return out


TrainPair = tuple[TrainConfig, Trainer]


@dataclasses.dataclass
class BitField:
    """Represents a bit field in a uint64."""

    offset: np.uint64
    dtype: np.dtype

    # numpy implicitly converts scalar types like np.float16 to a dtype when needed,
    # so we do, too
    def __init__(self, offset: int, dtype: np.dtype | type) -> None:
        """Create a description of a bit field.

        Arguments:
            offset: int - The offset of the field within the uint64, in bits.
            dtype: np.dtype | type - The type of the field. The underlying bytes
            of this type are what is stored in the uint64 at this offset.
        """
        self.offset = np.uint64(offset)
        self.dtype = np.dtype(dtype)

    def ensure_value_fits(self, value: ArrayLike) -> None:
        # There's no type annotation for "integer array like" so we let it throw here if
        # this is not an integer
        if np.any(value < np.iinfo(self.dtype).min):  # type: ignore[operator]
            raise ValueError(
                f"Value {value!r} is less than the minimum value for {self.dtype}"
            )
        if np.any(value > np.iinfo(self.dtype).max):  # type: ignore[operator]
            raise ValueError(
                f"Value {value!r} is greater than the maximum value for {self.dtype}"
            )

    def mask(self) -> np.uint64:
        """Returns an all-1s bitmask for the field."""
        return np.uint64((1 << (self.size_in_bytes() * 8)) - 1)

    def size_in_bytes(self) -> int:
        """How many bytes does this field take up?"""
        return self.dtype.itemsize

    def uint_type(self) -> np.dtype:
        """A uint type that is the same size as the field's value."""
        # Yes, this is in bytes, so u4 == uint32
        return np.dtype(f"u{self.size_in_bytes()}")

    def decode_from(self, container: NDArray[np.uint64] | np.uint64) -> NDArray:
        """Extracts the field from a uint64 container."""
        return (
            ((container >> self.offset) & self.mask())
            .astype(self.uint_type())
            .view(self.dtype)
        )

    def encode(self, value) -> NDArray[np.uint64]:
        """Encodes & places the field; caller should + the field into the container."""
        # Note that just `view(self.uint_type())` is not sufficient -- we need to
        # extend the value to fill the full uint64 size so when we shift it doesn't
        # just shift off the end of the (smaller) uint_type.
        return (
            value.astype(self.dtype).view(self.uint_type()).astype(np.uint64)
            << self.offset
        )


@dataclasses.dataclass
class DataSourceDims:
    """Dimension metadata to produce interpretable `xarray.DataArray`s.

    Each float in the encoded `xarray.DataArray` is interpreted as a uint64 broken
    into the following fields, MSB first:
      * 8 bits fixed as 0100 0000
        * which overlaps with float64 exponent to make it non-NaN and non-subnormal
      * lat encoded as a float16
      * lng encoded as a float16
      * days_since_start encoded as a uint16
      * data_var_index encoded as a uint8

    For example, given lat=90, lng=180, days_since_start=8, data_var_index=7:

        * header: 01000000, hex 0x40
        * lat: 01010101 10100000, hex 0x55A0 (the float16 encoding of 90.0)
        * lng: 01011001 10100000, hex 0x59A0 (the float16 encoding of 180.0)
        * days_since_start: 00000000 00001000, hex 0x0008 (the uint16 encoding of 8)
        * data_var_index: 00000111, hex 0x07 (the uint8 encoding of 7)

    This produces this uint64: 0x4055A059A0000807
    Which when interpreted as a float: https://float.exposed/0x4055A059A0000807
    gives about 86.50.

    """

    _header_value: ClassVar[np.uint64] = np.uint64(0b01000000)

    _header_field: ClassVar[BitField] = BitField(offset=56, dtype=np.uint8)
    _lat_field: ClassVar[BitField] = BitField(offset=40, dtype=np.float16)
    _lng_field: ClassVar[BitField] = BitField(offset=24, dtype=np.float16)
    _days_since_start_field: ClassVar[BitField] = BitField(offset=8, dtype=np.uint16)
    _data_var_index_field: ClassVar[BitField] = BitField(offset=0, dtype=np.uint8)

    lat: NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.arange(-89.24, 90, 1, dtype=np.float64)
    )
    lng: NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.arange(0.5, 360, 1, dtype=np.float64)
    )
    days_since_start: NDArray[np.uint32] = dataclasses.field(
        default_factory=lambda: np.array([0, 5, 10], dtype=np.uint32)
    )
    start_day: cftime.datetime = JulianDate("1969-08-05").datetime

    def __post_init__(self):
        if np.any(self.lat < -90.0) or np.any(self.lat > 90.0):
            raise ValueError("lat values are expected to be between -90 and 90.")

        if np.any(self.lng < 0.0) or np.any(self.lng > 360.0):
            raise ValueError("lng values are expected to be between 0 and 360 degrees.")

        self._days_since_start_field.ensure_value_fits(self.days_since_start)

    def __eq__(self, other) -> bool:
        # lat and lng values round-trip via float16s, so declare them equal if they
        # would encode to the same float16.
        # We can use exact equality with days_since_start since they are ints.
        return (
            np.array_equal(self.lat.astype(np.float16), other.lat.astype(np.float16))
            and np.array_equal(
                self.lng.astype(np.float16), other.lng.astype(np.float16)
            )
            and np.array_equal(self.days_since_start, other.days_since_start)
            and self.start_day == other.start_day
        )

    def set_time_range(self, time_range: xr.CFTimeIndex) -> None:
        self.start_day = time_range[0]
        units = f"days since {self.start_day}"
        self.days_since_start = np.array(
            [
                cftime.date2num(date, units, calendar=self.start_day.calendar)
                for date in time_range
            ]
        )

    def to_coords(self) -> dict[str, xr.DataArray]:
        units = f"days since {self.start_day}"
        time = np.array(
            [
                cftime.num2date(num, units, calendar=self.start_day.calendar)
                for num in self.days_since_start
            ]
        )

        coords = {
            "lon": xr.DataArray(self.lng, dims=["lon"]),
            "lat": xr.DataArray(self.lat, dims=["lat"]),
            "lev": xr.DataArray(np.array(TEST_DATASET_SPEC.depth_levels), dims=["lev"]),
            "time": xr.DataArray(time, dims=["time"]),
        }
        return coords

    def encode(self, data_var_index: np.uint | int = 0) -> xr.DataArray:
        """Encodes source data dimensions into an array of interpretable `np.float64`s.

        Arguments:
            data_var_index: int - The index of the data variable. Default is 0.

        Returns:
            An xarray.DataArray of np.float64 numbers with the above encoding scheme.
        """
        self._data_var_index_field.ensure_value_fits(data_var_index)

        days_reshaped = self.days_since_start[:, np.newaxis, np.newaxis]
        latlng_grid = np.stack(
            np.meshgrid(
                self.lat[::-1],
                self.lng,
                indexing="ij",
            ),
            axis=0,
        )

        template_grid = self._lat_field.encode(
            latlng_grid[0, :, :]
        ) + self._lng_field.encode(latlng_grid[1, :, :])
        rolled_out_grid = np.repeat(
            template_grid[np.newaxis, :, :], days_reshaped.shape[0], axis=0
        )

        interpretable_grid = (
            rolled_out_grid
            + self._days_since_start_field.encode(days_reshaped)
            + self._data_var_index_field.encode(np.array(data_var_index))
            + self._header_field.encode(self._header_value)
        )
        return xr.DataArray(
            interpretable_grid.view(np.float64),
            dims=["time", "lat", "lon"],
            attrs={
                "start_day": self.start_day.toordinal(),
                "start_day_cal": self.start_day.calendar,
            },
        )

    @classmethod
    def decode(cls, da: xr.DataArray) -> tuple[Self, NDArray[np.uint]]:
        """Parse array of encoded floats into its constituent parts.

        Arguments:
            da: DataArray with encoded floats. See `encode`.

        Returns:
            (DataSourceDims, int) - Parsed dims and data_var index.
        """
        encoded = da.to_numpy().view(np.uint64)
        assert len(encoded.shape) == 3, (
            "DataArray must have (time, lat, lng) dimensions."
        )

        assert np.all(cls._header_field.decode_from(encoded) == cls._header_value), (
            "Data did not come from `encode`. "
        )

        scalar = encoded.flat[0].view(np.uint64)
        tim_dim = encoded[:, 0, 0]
        lat_dim = encoded[0, :, 0][::-1]
        lng_dim = encoded[0, 0, :]

        days_since_start = cls._days_since_start_field.decode_from(tim_dim)
        lat = cls._lat_field.decode_from(lat_dim)
        lng = cls._lng_field.decode_from(lng_dim)
        data_var_index = cls._data_var_index_field.decode_from(scalar)

        data_source = cls(
            lat=lat,
            lng=lng,
            days_since_start=days_since_start,
            start_day=cftime.datetime.fromordinal(
                da.attrs.get("start_day"), da.attrs.get("start_day_cal")
            ),
        )
        return data_source, data_var_index


def cache_dir(pytestconfig: pytest.Config) -> pathlib.Path:
    dir = pytestconfig.rootpath / ".data_cache"
    dir.mkdir(parents=True, exist_ok=True)
    return dir


@pytest.fixture(scope="session", params=ALL_CONFIGS)
def config_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="session", params=[e for e in c.LoaderVersion])
def loader_version(request: pytest.FixtureRequest) -> c.LoaderVersion:
    return request.param


@pytest.fixture(scope="session", params=[0, 1], ids=lambda x: f"hist{x}")
def history(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(scope="session", params=["standard", "match", "mix"])
def schedule(request: pytest.FixtureRequest) -> TrainSchedule:
    return request.param


# Run a test for both CPU and GPU, and allows selecting or skipping CUDA tests.
@pytest.fixture(
    params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)], scope="session"
)
def backend(request) -> TrainBackendConfig:
    return request.param


def _uncached_data_source(name: str) -> DataSource:
    match name:
        case "mock":
            time_range = xr.cftime_range(
                "1975-08-05", "1975-12-31", freq="5D", calendar="julian"
            )
            dims = DataSourceDims()
            dims.set_time_range(time_range)

            coords = dims.to_coords()
            normal = np.random.normal(size=(len(coords["lat"]), len(coords["lon"])))

            vars_2d = {
                var: dims.encode(i)
                for i, var in enumerate(["hfds", "tauuo", "tauvo", "zos"])
            }
            vars_3d = {
                f"{var}_{lev}": dims.encode(len(vars_2d) + i + j * 10)
                for i, var in enumerate(["so", "thetao", "uo", "vo"])
                for j, lev in enumerate(TEST_DATASET_SPEC.depth_i_levels)
            }
            # Mask with a binary circle.
            masks = {
                f"mask_{lev}": xr.DataArray(
                    np.where(normal > 0.5**lev, 1, 0), dims=["lat", "lon"]
                )
                for lev in range(len(TEST_DATASET_SPEC.depth_i_levels))
            }
            ds = xr.Dataset(vars_2d | vars_3d | masks, coords=coords)

            return DataSource.from_datasets(
                ds,
                ds.mean(),
                ds.std(),
                dataset_spec=TEST_DATASET_SPEC,
                name=name,
                prognostic_var_names=list(vars_3d.keys()),
                boundary_var_names=list(vars_2d.keys()),
            )

        case "mock-om4" | "compact":
            # Create an accurate mock of the om4 data (and compact data).

            # 48 time slices.
            time_range = xr.cftime_range(
                "1975-08-05", "1976-03-31", freq="5D", calendar="julian"
            )
            dims = DataSourceDims()
            dims.set_time_range(time_range)

            # The original remote OM4 data uses y/x, not lat/lon. This should be ok for our tests.
            coords = dims.to_coords()
            lev, lat, lon = coords["lev"], coords["lat"], coords["lon"]

            # Use a fixed seed so "mock-om4" and "compact" produce identical wetmasks
            # (they are cached independently).
            rng = np.random.RandomState(42)
            norm = rng.normal(size=(len(lev), len(lat), len(lon)))
            coords.update(
                wetmask=xr.DataArray(
                    np.where(norm > 0.5, 1, 0),
                    coords=[lev, lat, lon],
                )
            )

            var_names_2d = ["hfds", "hfds_anomalies", "tauuo", "tauvo"]
            var_names_3d = ["so", "thetao", "uo", "vo"]

            def _fmtl(lev: float) -> str:
                return str(lev).replace(".", "_")

            vars_2d = {var: dims.encode(i) for i, var in enumerate(var_names_2d)}
            vars_3d = {
                f"{var}_lev_{_fmtl(lev)}": dims.encode(len(vars_2d) + i + j * 10)
                for i, var in enumerate(var_names_3d)
                for j, lev in enumerate(TEST_DATASET_SPEC.depth_levels)
            }

            # zos is an edge case 3d var.
            vars_3d.update(zos=dims.encode(len(vars_2d) + len(vars_3d) + 21))
            var_names_3d += ["zos"]

            data = xr.Dataset(vars_2d | vars_3d, coords=coords)

            # This should be equivalent to what the remote om4 means and stds are.
            means = data.mean(dim=["time", "lat", "lon"])
            stds = data.std(dim=["time", "lat", "lon"])

            prognostic_var_names = _om4_canonical_var_names(vars_3d.keys())
            boundary_var_names = list(vars_2d.keys())
            if name == "compact":
                data = compact_dataset(data)
                means = compact_dataset(means)
                stds = compact_dataset(stds)
                prognostic_var_names = var_names_3d
                boundary_var_names = var_names_2d

            return DataSource.from_datasets(
                data=data,
                means=means,
                stds=stds,
                dataset_spec=TEST_DATASET_SPEC,
                name=name,
                prognostic_var_names=prognostic_var_names,
                boundary_var_names=boundary_var_names,
            )
        case _:
            raise ValueError(f"Unknown data source: {name}.")


def _maybe_read_cache(cache_root: pathlib.Path, cache_name: str) -> DataSource | None:
    """Open a cached DataSource from a cache directory if it exists.

    The caller must ensure concurrent processes/threads do not change this cache.
    """
    cache = cache_root / cache_name
    try:
        data = xr.open_zarr(cache / "data.zarr")
        means = xr.open_dataset(cache / "means.nc")
        stds = xr.open_dataset(cache / "stds.nc")

        boundary_vars = [
            str(v)
            for v in data.data_vars
            if v in TEST_FULL_DATASET_SPEC.boundary_var_names
        ]

        if _is_compact(data, means, stds):
            prognostic_var_names: list[str] = []
            for var in data.data_vars:
                if var in boundary_vars or "mask" in var:
                    continue
                if "lev" in data[var].dims:
                    prognostic_var_names.extend(
                        f"{var}_{i}" for i in range(len(data.lev))
                    )
                else:
                    prognostic_var_names.append(str(var))
        else:
            prognostic_var_names = _om4_canonical_var_names(
                str(v)
                for v in data.data_vars
                if v not in boundary_vars and "mask" not in v
            )

        return DataSource.from_datasets(
            name=cache_name,
            data=data,
            means=means,
            stds=stds,
            dataset_spec=TEST_DATASET_SPEC,
            prognostic_var_names=prognostic_var_names,
            boundary_var_names=boundary_vars,
        )
    except (FileNotFoundError, PermissionError):
        return None


def _write_cache(cache_root: pathlib.Path, data_source: DataSource) -> None:
    """Write a DataSource to a new cache directory.

    The caller must ensure concurrent processes/threads do not read or write to
    this cache while this function is running.
    """
    cache = cache_root / data_source.name

    # Turn off compression! Our training datasets currently have compression turned off.
    #  See https://github.com/m2lines/ocean_emulators/blob/main/ocean_emulators/__main__.py#L240
    assert not (dz := cache / "data.zarr").exists(), "Data already exists in cache"
    data_source.data.to_zarr(
        dz, encoding={dv: {"compressor": None} for dv in data_source.data.data_vars}
    )
    assert not (dm := cache / "means.nc").exists(), "Means already exists in cache"
    data_source.means.to_netcdf(dm)
    assert not (ds := cache / "stds.nc").exists(), "Stds already exists in cache"
    data_source.stds.to_netcdf(ds)


@pytest.fixture(scope="session", params=["mock", "mock-om4", "compact"])
def data_source(request, pytestconfig) -> DataSource:
    """Returns remote and in-memory `xarray.Dataset`s for tests."""
    our_cache_dir = cache_dir(pytestconfig)
    data_type = request.param
    with filelock.FileLock(our_cache_dir / f"{data_type}.lock"):
        # Use cached data if available.
        if cached_data := _maybe_read_cache(our_cache_dir, data_type):
            return cached_data

        new_data = _uncached_data_source(data_type)
        _write_cache(our_cache_dir, new_data)
        return new_data


@pytest.fixture(scope="session", params=[[]])
def extra_config_args(request) -> list[str]:
    return request.param


_NEXT_TEST_ID = 0


def unique_test_name(config_name: str) -> str:
    global _NEXT_TEST_ID
    _NEXT_TEST_ID += 1
    return f"test_{config_name}_{_NEXT_TEST_ID}"


@pytest.fixture(scope="function")
def train_config(
    data_source: DataSource,
    pytestconfig: pytest.Config,
    config_name: str,
    backend: TrainBackendConfig,
    extra_config_args: list[str],
) -> TrainConfig:
    """
    This fixture is used to create a config/trainer pair for each possible
    configuration.
    """
    cache = cache_dir(pytestconfig)
    assert (cache / data_source.name).exists(), (
        "Expected cache to be created by data_source fixture"
    )

    # Open default training script; modify it as necessary.
    train_config = TrainConfig.from_yaml_and_cli(
        [
            # file to read
            str(pytestconfig.rootpath / "configs" / config_name),
            "--experiment.data_root",
            str(cache / data_source.name),
            "--backend",
            backend,
            "--experiment.name",
            # we make a unique name to avoid collisions on disk for output files
            unique_test_name(config_name),
        ]
        + extra_config_args
    )

    return train_config


@pytest.fixture(scope="function")
def trainer_pair(
    train_config,
    request,
    config_name: str,
) -> Generator[tuple[TrainConfig, Trainer], None, None]:
    """Provides a state-scoped config and trainer for tests."""
    with MultitonScope():
        trainer = Trainer(train_config)

        # cur_step will set the number of pairs in the input/output sample
        trainer.init_data_loaders(cur_step=train_config.steps[0])

        yield train_config, trainer


@pytest.fixture
def dummy_src():
    h, w = 4, 8
    coords = {"lev": [0], "lat": np.arange(h), "lon": np.arange(w)}
    data = xr.Dataset(
        {
            "thetao": (("lev", "lat", "lon"), np.zeros((1, h, w))),
            "hfds": (("lat", "lon"), np.zeros((h, w))),
        },
        coords=coords,
    )
    masks = Masks(torch.ones(1, h, w), torch.ones(h, w))
    src = DataSource(
        name="dummy",
        data=data,
        means=data.mean(dim=["lat", "lon"]),
        stds=data.std(dim=["lat", "lon"]),
        masks=masks,
        dataset_spec=TEST_DATASET_SPEC,
    )
    yield src
