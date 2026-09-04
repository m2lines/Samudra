# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import multiprocessing
import subprocess
import sys
from pathlib import Path

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from samudra.config import Om4DataSourceConfig
from samudra.datasets import TorchTrainDataset
from samudra.utils.data import CanonicalSource, Masks
from samudra.utils.location import LocalLocation
from samudra.utils.xql import with_xql_reader
from tests.conftest import TEST_DATA_LAYOUT

pytest.importorskip("xarray_sql")


def _source() -> CanonicalSource:
    time = xr.CFTimeIndex(
        [cftime.datetime(2000, 1, day, 12, calendar="julian") for day in range(1, 7)]
    )
    lat = np.array([-1.0, 1.0])
    lon = np.array([0.5, 1.5, 2.5])
    base = np.arange(len(time) * len(lat) * len(lon), dtype=np.float32).reshape(
        len(time), len(lat), len(lon)
    )
    data = xr.Dataset(
        {
            "so_0": (("time", "lat", "lon"), base),
            "so_2": (("time", "lat", "lon"), base + 100),
            "zos": (("time", "lat", "lon"), base + 200),
            "hfds": (("time", "lat", "lon"), base + 300),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    means = data.mean(("time", "lat", "lon"))
    stds = data.std(("time", "lat", "lon"))
    return CanonicalSource.from_canonical_datasets(
        "xql-test",
        data,
        means,
        stds,
        Masks(
            prognostic=torch.ones((3, len(lat), len(lon)), dtype=torch.bool),
            boundary=torch.ones((len(lat), len(lon)), dtype=torch.bool),
        ),
        TEST_DATA_LAYOUT,
    )


def test_xql_reader_matches_xarray_for_shaped_repeated_indices() -> None:
    source = _source()
    xql_source = with_xql_reader(source, time_chunk_size=2)
    indices = np.array([[3, 1], [-1, 3]])
    channels = ("hfds", "so_2")

    np.testing.assert_allclose(
        xql_source.read(indices, channels), source.read(indices, channels)
    )


def test_xql_reader_loads_torch_training_samples() -> None:
    source = _source()

    def training_dataset(input_source: CanonicalSource) -> TorchTrainDataset:
        return TorchTrainDataset(
            input_source=input_source,
            label_source=None,
            prognostic_var_names=("so_0", "so_2", "zos"),
            boundary_var_names=("hfds",),
            hist=1,
            steps=1,
            normalize_before_mask=True,
            masked_fill_value=0.0,
            concurrent_compute_=True,
        )

    expected = training_dataset(source)[0]
    actual = training_dataset(with_xql_reader(source, time_chunk_size=2))[0]
    for expected_step, actual_step in zip(expected.steps, actual.steps, strict=True):
        for expected_tensor, actual_tensor in zip(
            expected_step, actual_step, strict=True
        ):
            torch.testing.assert_close(actual_tensor, expected_tensor)


def test_xql_context_is_created_after_fork() -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is unavailable on this platform")

    # A fresh interpreter ensures this specifically exercises the supported
    # lifecycle: construct in the parent, initialize DataFusion in the worker.
    code = """
import multiprocessing
import numpy as np
from tests.test_xql_data import _source
from samudra.utils.xql import with_xql_reader

source = with_xql_reader(_source())

def read(queue):
    queue.put(source.read(np.array([0, 2]), (\"so_0\",)).shape)

ctx = multiprocessing.get_context(\"fork\")
queue = ctx.Queue()
process = ctx.Process(target=read, args=(queue,))
process.start()
process.join(30)
assert process.exitcode == 0, process.exitcode
assert queue.get(timeout=1) == (2, 1, 2, 3)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=45)


@pytest.mark.manual
def test_public_om4_xql_read_matches_xarray() -> None:
    import yaml

    config_path = Path("src/samudra/configs/data/om4_demo.yaml")
    source_config = Om4DataSourceConfig.model_validate(
        yaml.safe_load(config_path.read_text())["sources"][0]
    )
    source = source_config._build_source(
        LocalLocation(path=Path("/tmp")), turn_on_dask=False
    )
    indices = np.array([[0, 2], [1, 3]])
    channels = ("zos", "hfds")

    np.testing.assert_allclose(
        with_xql_reader(source).read(indices, channels),
        source.read(indices, channels),
        equal_nan=True,
    )
