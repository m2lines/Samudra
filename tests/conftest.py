import dataclasses
import os
import tempfile
from typing import Any, Generator, TypedDict

import numpy as np
import pytest
import torch
import xarray as xr

import constants as c
from config import TrainConfig


class DataSource(TypedDict):
    """In-memory `xarray.Dataset`s needed for tests."""

    data: xr.Dataset
    means: xr.Dataset
    stds: xr.Dataset


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
@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def device(request):
    return torch.device(request.param)


@pytest.fixture(scope="session")
def data_source() -> DataSource:
    """Returns in-memory `xarray.Dataset`s for tests."""
    summer_of_love = xr.cftime_range(
        "1969-08-05", "1969-12-31", freq="5D", calendar="noleap"
    )

    coords = {
        "lon": xr.DataArray(np.arange(0.5, 360, 1), dims=["lon"]),  # Float[360]
        "lat": xr.DataArray(np.arange(-89.24, 90, 1), dims=["lat"]),  # Float[180]
        "time": xr.DataArray(summer_of_love, dims=["time"]),  # CFTimeIndex[30]
    }

    normal = np.random.normal(
        size=(len(coords["lat"]), len(coords["lon"]))
    )  # Float[180, 360]

    # Create array of relative times (number of days since start).
    timedeltas = [date - summer_of_love[0] for date in summer_of_love]
    days_from_start = np.array(
        [delta.total_seconds() / (24 * 3600) for delta in timedeltas]
    )
    days_reshaped = days_from_start[:, np.newaxis, np.newaxis]  # Float[30, 1, 1]

    latlng_grid = np.stack(
        np.meshgrid(coords["lat"][::-1], coords["lon"], indexing="ij"),
        axis=0,
    )
    latlng_grid_3sf = np.around(latlng_grid, decimals=2)

    template_grid = latlng_grid_3sf[0, :, :] * 1_000_000 + latlng_grid_3sf[1, :, :] * 10
    rolled_out_grid = np.repeat(
        template_grid[np.newaxis, :, :], len(summer_of_love), axis=0
    )

    # A floating point digit-encoded grid.
    # ------------------------------------
    # Each number in this array is an interpretable float with the following scheme:
    # AAAAGGGG.TTT[DD]
    # - A := Latitude (originally a float with 2 decimal digits).
    # - G := Longitude ""
    # - T := Time (the number of days since the start time).
    # - D := (optional) A int representing the index of the current data variable.
    interpretable_grid = rolled_out_grid + days_reshaped / 1000  # Float[30, 180, 360]

    vars_2d = {
        var: xr.DataArray(interpretable_grid, dims=["time", "lat", "lon"])
        + float(i) / 100_000
        for i, var in enumerate(["hfds", "tauuo", "tauvo", "zos"])
    }
    vars_3d = {
        f"{var}_{lev}": xr.DataArray(interpretable_grid, dims=["time", "lat", "lon"])
        + float(i + j + len(vars_2d)) / 100_000
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

    return {"data": ds, "means": ds.mean("time"), "stds": ds.std("time")}


# TODO(alxmrs): Consider yielding multiple test configs.
@pytest.fixture(scope="session")
def train_config(
    data_source: DataSource, pytestconfig: pytest.Config
) -> Generator[TrainConfig, Any, None]:
    with tempfile.TemporaryDirectory() as tmpdir:

        def _make_path(name_with_ext: str) -> str:
            return os.path.join(tmpdir, name_with_ext)

        # Write test data to a temporary directory.
        data_source["data"].to_zarr(_make_path("data.zarr"))
        data_source["means"].to_netcdf(_make_path("means.netcdf"))
        data_source["stds"].to_netcdf(_make_path("stds.netcdf"))

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
        )

        # After contextmanager closes, all test data will be automatically cleaned up.
        yield test_data_trainer
