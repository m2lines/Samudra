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
    out = {}

    summer_of_love = xr.cftime_range(
        "1969-08-10", "1969-10-01", freq="D", calendar="noleap"
    )

    coords = {
        "lon": xr.DataArray(np.arange(0.5, 360, 1), dims=["lon"]),
        "lat": xr.DataArray(np.arange(-89.24, 90, 1), dims=["lat"]),
        "time": xr.DataArray(summer_of_love, dims=["time"]),
    }

    normal = np.random.normal(
        size=(lat := len(coords["lat"]), lon := len(coords["lon"]))
    )

    ds = xr.Dataset(
        {
            # 2D variables
            var: xr.DataArray(
                np.random.random([len(summer_of_love), lat, lon]),
                dims=["time", "lat", "lon"],
            )
            for var in ["hfds", "tauuo", "tauvo", "zos"]
        }
        | {
            # 3D variables
            f"{var}_{lev}": xr.DataArray(
                np.random.random([len(summer_of_love), lat, lon]),
                dims=["time", "lat", "lon"],
            )
            for var in ["so", "thetao", "uo", "vo"]
            for lev in c.DEPTH_I_LEVELS
        }
        | {
            # Masks -- make a binary circle mask.
            f"mask_{lev}": xr.DataArray(
                np.where(normal > 0.5**lev, 1, 0),
                dims=["lat", "lon"],
            )
            for lev in range(len(c.DEPTH_I_LEVELS))
        },
        coords=coords,
    )

    out["data"] = ds
    out["means"] = ds.mean("time")
    out["stds"] = ds.std("time")

    return out


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
