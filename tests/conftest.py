import dataclasses
import os
import tempfile
from typing import Any, Generator

import cftime
import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray
from typing_extensions import Self

import constants as c
from config import TrainBackendConfig, TrainConfig
from train_3D import Trainer


@dataclasses.dataclass
class DataSource:
    """In-memory `xarray.Dataset`s needed for tests."""

    data: xr.Dataset
    means: xr.Dataset
    stds: xr.Dataset


def _trunc(arr: NDArray[np.floating], decimals: int) -> NDArray[np.floating]:
    factor = 10**decimals
    return np.trunc(arr * factor) / factor


@dataclasses.dataclass
class DataSourceDims:
    """Dimension metadata to produce interpretable `xarray.DataArray`s.

    Each float in the encoded `xarray.DataArray` has the following scheme:
    (s)AAAAGGGG.TTTDDD
    - s := Sign (+/-) of the Latitude. All other values are non-negative.
    - A := Latitude, which ranges from 90.00 <--> -90.00 (only decimals=2).
    - G := Longitude, which ranges from 000.0 <--> 360.0 (only decimals=1).
    - T := Time, the number of days since the start time.
    - D := A int representing the index of the current data variable.
    """

    lat: NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.arange(-89.24, 90, 1, dtype=np.float64)
    )
    lng: NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.arange(0.5, 360, 1, dtype=np.float64)
    )
    days_since_start: NDArray[np.int32] = dataclasses.field(
        default_factory=lambda: np.array([0, 5, 10])
    )
    start_day: cftime.datetime = cftime.DatetimeNoLeap.strptime(
        "1969-08-05", "%Y-%m-%d", "noleap"
    )

    def __post_init__(self):
        # In Python, modulus operations preserve the sign _based on the denominator_.
        # So, to ensure that we can have negative latitudes, we parse the sign value
        # of the lat array.
        lat_sgn = np.sign(self.lat)
        lat_sgn = np.where(lat_sgn == 0, 1, lat_sgn)  # change np.sign(0) to +1, not 0.
        self.lat = _trunc(self.lat.astype(np.float64), 2)
        if np.any(self.lat < -90.0) or np.any(self.lat > 90.0):
            raise ValueError("lat values must be between -90 and 90.")
        self.lat %= lat_sgn * 100

        # All other values in the encoding must be non-negative.
        self.lng = _trunc(self.lng.astype(np.float64), 1)
        if np.any(self.lng < 0.0) or np.any(self.lng > 360.0):
            raise ValueError("lng values must be between 0 and 360 degrees.")
        self.lng %= 1000

        self.days_since_start = self.days_since_start.astype(np.int32)
        if np.any(self.days_since_start < 0) or np.any(self.days_since_start > 999):
            raise ValueError("days_since_start must be between 0 and 999.")

    def __eq__(self, other) -> bool:
        # lat and lng values will often have floating point rounding errors.
        # So, we declare arrays as equal within a generous tolerance.
        # We can use exact equality with days_since_start since they are ints.
        return (
            np.allclose(self.lat, other.lat, atol=0.1)
            and np.allclose(self.lng, other.lng, atol=0.01)
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

    def encode(self, data_var_index: int = 0) -> xr.DataArray:
        """Encodes source data dimensions into an array of interpretable `np.float64`s.

        Arguments:
            data_var_index: int - The index of the data variable. Default is 0. This
              must be a value between 0 and 999.

        Returns:
            An xarray.DataArray of np.float64 numbers with the above encoding scheme.
        """
        if data_var_index < 0 or data_var_index > 999:
            raise ValueError("data_var_index must be between 0 and 999.")

        days_reshaped = self.days_since_start[:, np.newaxis, np.newaxis].astype(
            np.int32
        )  # Float[D, 1, 1]

        latlng_grid = np.stack(
            np.meshgrid(self.lat[::-1], self.lng, indexing="ij"),
            axis=0,
        )
        latlng_grid_3sf = np.around(latlng_grid, decimals=2)

        template_grid = (
            latlng_grid_3sf[0, :, :] * 1_000_000.0 + latlng_grid_3sf[1, :, :] * 10.0
        )
        rolled_out_grid = np.repeat(
            template_grid[np.newaxis, :, :], days_reshaped.shape[0], axis=0
        )

        interpretable_grid = rolled_out_grid + (days_reshaped / 1000)
        data_index_digits = float(data_var_index) / 1_000_000
        return xr.DataArray(
            interpretable_grid + data_index_digits,
            dims=["time", "lat", "lon"],
            attrs={
                "start_day": self.start_day.toordinal(),
                "start_day_cal": self.start_day.calendar,
            },
        )

    @classmethod
    def decode(cls, da: xr.DataArray) -> tuple[Self, int]:
        """Parse array of encoded floats into its constituent parts.

        AAAAGGGG.TTTDDD -->
         (
            DataSourceDims(lat=AA.AA, lng=GGG.G, days_since_start=TTT),
            DDD,  # data_source_index
        )

        Arguments:
            da: DataArray with encoded floats. See `encode_data_source`.

        Returns:
            (DataSourceDims, int) - Parsed dims and data_var index.
        """
        encoded = da.to_numpy()
        assert len(encoded.shape) == 3, (
            "DataArray must have (time, lat, lng) dimensions."
        )

        scalar = encoded.flat[0]
        tim_dim = encoded[:, 0, 0]
        lat_dim = encoded[0, :, 0][::-1]
        lng_dim = encoded[0, 0, :]

        # Signs are either +1 or -1. If the value is 0, default to +1.
        lat_sign = np.sign(lat_dim)
        lat_sign = np.where(lat_sign == 0, 1, lat_sign)

        # In python, the modulus operator preserves the of the _denominator only_.
        # Thus, we need to parse the sign values for latitude above and include it
        # in our float arithemetic array processing.
        # We only need to do this for the lat values because we do not allow any
        # other values encoded in the float to be negative.
        days_since_start = ((tim_dim * 1000) % 1000).astype(np.int32)
        lat = (lat_dim // (lat_sign * 10_000)) / (lat_sign * 100)
        lng = lng_dim / 10 % 1000
        # we need both np.round and _trunc because changing magnitude
        # of very small floating point values (especially, odd numbers)
        # leads to repeated decimal values.
        data_var_index = int(np.round(_trunc(scalar, decimals=7) * 1_000_000) % 1000)

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


TrainPair = tuple[TrainConfig, Trainer]


# This micro-fixture is cached by pytest. Thus, we don't have to change
# the factory methods that throw errors during double initialization.
@pytest.fixture(scope="session")
def trainer_pair(train_config: TrainConfig) -> TrainPair:
    trainer = Trainer(train_config)

    # cur_step will set the number of pairs in the input/output sample
    trainer.init_data_loaders(cur_step=train_config.steps[0])

    return train_config, trainer
