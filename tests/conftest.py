import dataclasses
import os
import tempfile
from typing import Any, ClassVar, Generator

import cftime
import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import Self

import constants as c
from config import TrainBackendConfig, TrainConfig


@dataclasses.dataclass
class DataSource:
    """In-memory `xarray.Dataset`s needed for tests."""

    data: xr.Dataset
    means: xr.Dataset
    stds: xr.Dataset


@dataclasses.dataclass
class BitField:
    offset: int
    dtype: np.dtype

    # numpy implicitly converts scalar types like np.float16 to a dtype when needed,
    # so we do, too
    def __init__(self, offset: int, dtype: np.dtype | type) -> None:
        self.offset = offset
        self.dtype = np.dtype(dtype)

    def ensure_value_fits(self, value: np.array) -> None:
        if np.any(value < np.iinfo(self.dtype).min):
            raise ValueError(
                f"Value {value} is less than the minimum value for {self.dtype}"
            )
        if np.any(value > np.iinfo(self.dtype).max):
            raise ValueError(
                f"Value {value} is greater than the maximum value for {self.dtype}"
            )

    def mask(self) -> np.uint64:
        return np.uint64((1 << (self.size_in_bytes() * 8)) - 1)

    def size_in_bytes(self) -> int:
        return np.dtype(self.dtype).itemsize

    def uint_type(self) -> np.dtype:
        # Yes, this is in bytes for some reason so u4 == uint32
        return np.dtype(f"u{self.size_in_bytes()}")

    def decode_from(self, container: NDArray[np.uint64]) -> NDArray[np.uint64]:
        return (
            ((container >> np.uint64(self.offset)) & self.mask())
            .astype(self.uint_type())
            .view(self.dtype)
        )

    def encode(self, value: NDArray) -> NDArray[np.uint64]:
        return value.astype(self.dtype).view(self.uint_type()).astype(
            np.uint64
        ) << np.uint64(self.offset)


@dataclasses.dataclass
class DataSourceDims:
    """Dimension metadata to produce interpretable `xarray.DataArray`s.

    Each float in the encoded `xarray.DataArray` is interpreted as a uint64 broken
    into the following fields, MSB first:
      * 8 bits fixed as 0100 0000 (a non-NaN exponent)
      * lat encoded as a float16
      * lng encoded as a float16
      * days_since_start encoded as a uint16
      * data_var_index encoded as a uint8
    """

    _header_value: ClassVar[np.uint64] = np.uint64(0b01000000)

    _header: ClassVar[BitField] = BitField(offset=56, dtype=np.uint8)
    _lat: ClassVar[BitField] = BitField(offset=40, dtype=np.float16)
    _lng: ClassVar[BitField] = BitField(offset=24, dtype=np.float16)
    _days_since_start: ClassVar[BitField] = BitField(offset=8, dtype=np.uint16)
    _data_var_index: ClassVar[BitField] = BitField(offset=0, dtype=np.uint8)

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

        self._days_since_start.ensure_value_fits(self.days_since_start)

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
        self._data_var_index.ensure_value_fits(data_var_index)

        days_reshaped = self.days_since_start[:, np.newaxis, np.newaxis]
        latlng_grid = np.stack(
            np.meshgrid(
                self.lat[::-1],
                self.lng,
                indexing="ij",
            ),
            axis=0,
        )

        template_grid = self._lat.encode(latlng_grid[0, :, :]) + self._lng.encode(
            latlng_grid[1, :, :]
        )
        rolled_out_grid = np.repeat(
            template_grid[np.newaxis, :, :], days_reshaped.shape[0], axis=0
        )

        interpretable_grid = (
            rolled_out_grid
            + self._days_since_start.encode(days_reshaped)
            + self._data_var_index.encode(np.array(data_var_index))
            + self._header.encode(self._header_value)
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
    def decode(cls, da: xr.DataArray) -> tuple[Self, int]:
        """Parse array of encoded floats into its constituent parts.

        Arguments:
            da: DataArray with encoded floats. See `encode_data_source`.

        Returns:
            (DataSourceDims, int) - Parsed dims and data_var index.
        """
        encoded = da.to_numpy().view(np.uint64)
        assert len(encoded.shape) == 3, (
            "DataArray must have (time, lat, lng) dimensions."
        )

        assert np.all(cls._header.decode_from(encoded) == cls._header_value), (
            "Data did not come from `encode_data_source`. "
        )

        scalar = encoded.flat[0].view(np.uint64)
        tim_dim = encoded[:, 0, 0]
        lat_dim = encoded[0, :, 0][::-1]
        lng_dim = encoded[0, 0, :]

        days_since_start = cls._days_since_start.decode_from(tim_dim)
        lat = cls._lat.decode_from(lat_dim)
        lng = cls._lng.decode_from(lng_dim)
        data_var_index = cls._data_var_index.decode_from(scalar)

        data_source = cls(
            lat=lat,
            lng=lng,
            days_since_start=days_since_start,
            start_day=cftime.datetime.fromordinal(
                da.attrs.get("start_day"), da.attrs.get("start_day_cal")
            ),
        )
        return data_source, data_var_index


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


@pytest.fixture(scope="session")
def data_source() -> DataSource:
    """Returns in-memory `xarray.Dataset`s for tests."""
    summer_of_love = xr.cftime_range(
        "1969-08-05", "1969-12-31", freq="5D", calendar="noleap"
    )
    dims = DataSourceDims()
    dims.set_time_range(summer_of_love)

    coords = dims.to_coords()
    normal = np.random.normal(size=(len(coords["lat"]), len(coords["lon"])))

    vars_2d = {
        var: dims.encode(i) for i, var in enumerate(["hfds", "tauuo", "tauvo", "zos"])
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

    return DataSource(data=ds, means=ds.mean(), stds=ds.std())


@pytest.fixture(scope="session")
def train_config(
    data_source: DataSource, pytestconfig: pytest.Config, backend: TrainBackendConfig
) -> Generator[TrainConfig, Any, None]:
    with tempfile.TemporaryDirectory() as tmpdir:

        def _make_path(name_with_ext: str) -> str:
            return os.path.join(tmpdir, name_with_ext)

        # Write test data to a temporary directory.
        data_source.data.to_zarr(_make_path("data.zarr"))
        data_source.means.to_netcdf(_make_path("means.netcdf"))
        data_source.stds.to_netcdf(_make_path("stds.netcdf"))

        # Open default training script; modify it so it uses the temporary directory.
        default_config = pytestconfig.rootpath / "configs" / "train_cm4.test.yaml"
        trainer = TrainConfig.from_yaml(default_config)
        data_config = dataclasses.replace(
            trainer.data,
            data_path=_make_path("data.zarr"),
            data_means_path=_make_path("means.netcdf"),
            data_stds_path=_make_path("stds.netcdf"),
        )
        experiment_config = dataclasses.replace(
            trainer.experiment,
            cluster_data_dir=os.path.join(tmpdir, "cluster_data"),
        )
        test_data_trainer = dataclasses.replace(
            trainer,
            data=data_config,
            experiment=experiment_config,
            backend=backend,
        )

        # After contextmanager closes, all test data will be automatically cleaned up.
        yield test_data_trainer


# This micro-fixture is cached by pytest. Thus, we don't have to change
# the factory methods that throw errors during double initialization.
@pytest.fixture(scope="session")
def trainer_pair(train_config: TrainConfig):
    # Import needs to be here in order to prevent a gnarly jaxtyping bug:
    # See https://github.com/patrick-kidger/jaxtyping/issues/306
    from train_3D import Trainer

    trainer = Trainer(train_config)

    # cur_step will set the number of pairs in the input/output sample
    trainer.init_data_loaders(cur_step=train_config.steps[0])

    return train_config, trainer
