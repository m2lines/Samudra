import dataclasses
import pathlib
import random
import time
from typing import ClassVar

import cftime
import numpy as np
import pytest
import xarray as xr
from aiohttp import ServerDisconnectedError
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

import ocean_emulators.constants as c
from ocean_emulators.config import TrainBackendConfig, TrainConfig
from ocean_emulators.utils.multiton import MultitonScope

REMOTE_DATA = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/"
DEFAULT_CONFIG = "train_default.test.yaml"
ALL_CONFIGS = [DEFAULT_CONFIG, "train_default_2step.test.yaml"]


@dataclasses.dataclass
class DataSource:
    """In-memory `xarray.Dataset`s needed for tests."""

    name: str
    data: xr.Dataset
    means: xr.Dataset
    stds: xr.Dataset


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
    start_day: cftime.datetime = cftime.DatetimeNoLeap.strptime(
        "1969-08-05", "%Y-%m-%d", "noleap"
    )

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
            "lev": xr.DataArray(np.array(c.DEPTH_LEVELS), dims=["lev"]),
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


@pytest.fixture(scope="session", params=ALL_CONFIGS)
def config_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(autouse=True)
def check_skip_configs(request, config_name):
    """Decide if we run a test given only_configs and all_configs marks.

    * If there is an only_configs mark, that overrides all_configs.
      We run all configs in only_configs in that case.
    * If there is no only_configs mark, and there is an all_configs mark,
      we run all configs.
    * If there is no only_configs or all_configs mark, we run the test for the
      default config.

    """
    only_configs = request.node.get_closest_marker("only_configs")
    if only_configs:
        if config_name not in only_configs.args:
            pytest.skip(
                f"Skipping test for {config_name} because"
                f" it is not in {only_configs.args}"
            )
        if config_name not in ALL_CONFIGS:
            raise ValueError(
                f"Test config {config_name} is not"
                f" in the list of all configs: {ALL_CONFIGS}"
            )
    else:
        all_configs = request.node.get_closest_marker("all_configs")
        if all_configs is None:
            # If the test is not marked with all_configs, skip all configs except
            # the default config.
            if config_name != DEFAULT_CONFIG:
                pytest.skip(
                    f"Skipping test for {config_name}"
                    " because it is not marked with all_configs"
                )


def pytest_addoption(parser):
    parser.addoption(
        "--model1", action="store", help="Path to the first model .pt file"
    )
    parser.addoption(
        "--model2", action="store", help="Path to the second model .pt file"
    )


@pytest.fixture
def model1_path(request):
    return request.config.getoption("--model1")


@pytest.fixture
def model2_path(request):
    return request.config.getoption("--model2")


# Run a test for both CPU and GPU, and allows selecting or skipping CUDA tests.
# TODO(jder): Note that due to singletons, we can't use both cuda and non-cuda
# tests in the same test run. You should run the tests separately.
# See https://github.com/suryadheeshjith/Ocean_Emulator/issues/87
@pytest.fixture(
    params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)], scope="session"
)
def backend(request) -> TrainBackendConfig:
    return request.param


def maybe_read_cache(cache_root: pathlib.Path, cache_name: str) -> DataSource | None:
    """Open a cached DataSource from a cache directory if it exists."""
    cache = cache_root / cache_name
    try:
        data = xr.open_zarr(cache / "data.zarr")
        means = xr.open_dataset(cache / "means.nc")
        stds = xr.open_dataset(cache / "stds.nc")
        return DataSource(name=cache_name, data=data, means=means, stds=stds)
    except FileNotFoundError:
        return None


def maybe_write_cache(cache_root: pathlib.Path, data_source: DataSource) -> None:
    """Write a DataSource to a cache directory if it doesn't exist."""
    cache = cache_root / data_source.name
    if not (dz := cache / "data.zarr").exists():
        data_source.data.to_zarr(str(dz))
    if not (dm := cache / "means.nc").exists():
        data_source.means.to_netcdf(str(dm))
    if not (ds := cache / "stds.nc").exists():
        data_source.stds.to_netcdf(str(ds))


def cache_dir(pytestconfig: pytest.Config) -> pathlib.Path:
    dir = pytestconfig.rootpath / ".data_cache"
    dir.mkdir(parents=True, exist_ok=True)
    return dir


def retry_with_backoff(
    func, max_retries: int = 5, catch: type[Exception] = ServerDisconnectedError
):
    """Retry a function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except catch:
            if attempt == max_retries - 1:
                raise
            wait_time = (2**attempt) + random.uniform(0, 1)
            print(
                f"{catch}. Retrying in {wait_time:.2f}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_time)


@pytest.fixture(scope="session", params=["simulated", "remote-om4"])
def data_source(request, pytestconfig) -> DataSource:
    """Returns remote and in-memory `xarray.Dataset`s for tests."""
    # Use cached data if available.
    if cached_data := maybe_read_cache(cache_dir(pytestconfig), request.param):
        return cached_data

    match request.param:
        case "simulated":
            time_range = xr.cftime_range(
                "1975-08-05", "1975-12-31", freq="5D", calendar="noleap"
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
                for j, lev in enumerate(c.DEPTH_I_LEVELS)
            }
            # Mask with a binary circle.
            masks = {
                f"mask_{lev}": xr.DataArray(
                    np.where(normal > 0.5**lev, 1, 0), dims=["lat", "lon"]
                )
                for lev in range(len(c.DEPTH_I_LEVELS))
            }
            ds = xr.Dataset(vars_2d | vars_3d | masks, coords=coords)

            return DataSource(
                name=request.param, data=ds, means=ds.mean(), stds=ds.std()
            )
        case "remote-om4":
            # The chunk-size should be about the same as the size of the time slice
            # for optimal download time. In local experiments, this time range (which
            # matches the simulated data) is about 30 items.

            data = retry_with_backoff(
                (
                    lambda: xr.open_zarr(REMOTE_DATA + "OM4", chunks=dict(time=50))
                    .sel(time=slice("1975-08-05", "1976-03-31"))
                    .compute()
                )
            )
            means = retry_with_backoff(
                lambda: xr.open_dataset(
                    REMOTE_DATA + "OM4_means", engine="zarr", chunks={}
                ).compute()
            )
            stds = retry_with_backoff(
                lambda: xr.open_dataset(
                    REMOTE_DATA + "OM4_stds", engine="zarr", chunks={}
                ).compute()
            )

            return DataSource(
                name=request.param,
                data=data,
                means=means,
                stds=stds,
            )
        case _:
            raise ValueError(f"Unknown data source: {request.param}.")


@pytest.fixture(scope="session")
def train_config(
    data_source: DataSource,
    pytestconfig: pytest.Config,
    config_name: str,
    backend: TrainBackendConfig,
) -> TrainConfig:
    # Write test data to the cache directory if they aren't already there.
    cache = cache_dir(pytestconfig)
    maybe_write_cache(cache, data_source)

    # Open default training script; modify it as necessary.
    trainer = TrainConfig.from_yaml(pytestconfig.rootpath / "configs" / config_name)
    experiment_config = dataclasses.replace(
        trainer.experiment,
        cluster_data_dir=str(cache / data_source.name),
    )
    test_data_config = dataclasses.replace(
        trainer,
        experiment=experiment_config,
        backend=backend,
    )

    # Create a fresh scope to use in any tests using this config
    # (see set_scope below)
    scope = MultitonScope()
    setattr(test_data_config, "_multiton_scope", scope)

    return test_data_config


@pytest.fixture(scope="session")
def trainer_pair(train_config: TrainConfig):
    # Import needs to be here in order to prevent a gnarly jaxtyping bug:
    # See https://github.com/patrick-kidger/jaxtyping/issues/306
    from ocean_emulators.train import Trainer

    # NB fixtures still need to do this "by hand" since set_scope
    # doesn't run at session-scope time
    with getattr(train_config, "_multiton_scope"):
        trainer = Trainer(train_config)

        # cur_step will set the number of pairs in the input/output sample
        trainer.init_data_loaders(cur_step=train_config.steps[0])

    return train_config, trainer


@pytest.fixture(autouse=True, scope="function")
def set_scope(train_config: TrainConfig):
    """Automatically sets up the correct Multiton scope for each test.

    NB you must still do this manually for session-scoped fixtures.
    """
    with getattr(train_config, "_multiton_scope"):
        yield
