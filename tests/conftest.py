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
    AAAAGGGG.TTTDD
    - A := Latitude, which ranges from 90.00 <--> -90.00
    - G := Longitude, which ranges from 000.0 <--> 360.0
    - T := Time (the number of days since the start time).
    - D := (optional) A int representing the index of the current data variable.
    """

    lat: NDArray[np.float64] = np.arange(-89.24, 90, 1, dtype=np.float64)
    lng: NDArray[np.float64] = np.arange(0.5, 360, 1, dtype=np.float64)
    days_since_start: NDArray[np.int32] = np.array([0, 5, 10])
    start_day: cftime.datetime = cftime.DatetimeNoLeap.strptime(
        "1969-08-05", "%Y-%m-%d", "noleap"
    )

    def __post_init__(self):
        self.lat = _trunc(self.lat.astype(np.float64), 2) % 100
        self.lng = _trunc(self.lng.astype(np.float64), 1) % 1000
        self.days_since_start = self.days_since_start.astype(np.int32)
        if np.any(self.days_since_start < 0):
            raise ValueError("days_since_start must be non-negative.")
        if np.any(self.days_since_start > 999):
            raise ValueError("days_since_start must be less than three digits.")

    def __eq__(self, other) -> bool:
        return (
            np.array_equal(self.lat, other.lat)
            and np.array_equal(self.lng, other.lng)
            and np.array_equal(self.days_since_start, other.days_since_start)
            and self.start_day == other.start_day
        )

    def parse_cftime_range(self, time_range: xr.CFTimeIndex) -> None:
        self.start_day = time_range[0]
        timedeltas = [date - self.start_day for date in time_range]
        self.days_since_start = np.array(
            [delta.total_seconds() / (24 * 3600) for delta in timedeltas],
            dtype=np.int32,
        )

    def to_coords(self) -> dict[str, xr.DataArray]:
        time = np.array(
            [
                self.start_day + cftime._cftime.timedelta(days=int(d))
                for d in self.days_since_start
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
            data_var_index: int - The index of the data variable. Default is 0.

        Returns:
            An xarray.DataArray of np.float64 numbers with the above encoding scheme.
        """
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
        data_index_digits = float(data_var_index) / 100_000
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

        AAAAGGGG.TTTDD --> (
            DataSourceDims(
                lat=AA.AA,
                lng=GGG.G,
                days_since_start=TTT
            ),
            data_var_index=DD
        )

        Arguments:
            da: DataArray with encoded floats. See `encode_data_source`.

        Returns:
            (DataSourceDims, int) - Parsed coordinates and data_var index.
        """
        encoded = da.to_numpy()
        assert len(encoded.shape) == 3

        scalar = encoded.flat[0]
        tim_dim = encoded[:, 0, 0]
        lat_dim = encoded[0, :, 0][::-1]
        lng_dim = encoded[0, 0, :]

        # Signs are either +1 or -1. If the value is 0, default to +1.
        tim_sign = np.sign(tim_dim)
        tim_sign = np.where(tim_sign == 0, 1, tim_sign)
        lat_sign = np.sign(lat_dim)
        lat_sign = np.where(lat_sign == 0, 1, lat_sign)
        lng_sign = np.sign(lng_dim)
        lng_sign = np.where(lng_sign == 0, 1, lng_sign)

        days_since_start = ((tim_dim * 1000) % (tim_sign * 1000)).astype(np.int32)
        lat = (lat_dim // (lat_sign * 10_000)) / 100
        lng = lng_dim / 10 % (lng_sign * 1000)
        # we need both np.round and _trunc because changing magnitude
        # of very small floating point values (especially, odd numbers)
        # leads to repeated decimal values.
        data_var_index = int(np.round(_trunc(scalar, decimals=6) * 100_000) % 100)

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
    dims.parse_cftime_range(summer_of_love)

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

    return DataSource(data=ds, means=ds.mean("time"), stds=ds.std("time"))


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
