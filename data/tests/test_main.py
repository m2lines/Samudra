# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import sentinel

import numpy as np
import ocean_preprocessing.__main__ as main
import xarray as xr
from ocean_preprocessing.__main__ import CLI


def test_clusterless_dry_run_computes_without_a_client(tmp_path):
    pipeline = CLI(output_path=str(tmp_path / "unused.zarr"), dry_run=True)
    ds = xr.Dataset({"value": ("time", np.arange(3))}).chunk({"time": 1})

    pipeline._collect(ds)

    assert not (tmp_path / "unused.zarr").exists()


def test_collect_explicitly_writes_zarr_v2(tmp_path):
    output = tmp_path / "output.zarr"
    pipeline = CLI(output_path=str(output))
    ds = xr.Dataset({"value": ("time", np.arange(3))}).chunk({"time": 1})

    pipeline._collect(ds)

    metadata = json.loads((output / ".zgroup").read_text(encoding="utf-8"))
    assert metadata["zarr_format"] == 2


def test_collect_turns_compression_off_by_default(tmp_path):
    """The training stores are read without a decompression pass."""
    output = tmp_path / "output.zarr"
    pipeline = CLI(output_path=str(output))
    ds = xr.Dataset({"value": ("time", np.arange(3))}).chunk({"time": 1})

    pipeline._collect(ds)

    metadata = json.loads((output / "value" / ".zarray").read_text(encoding="utf-8"))
    assert metadata["compressor"] is None


def test_collect_can_keep_zarrs_default_compressor(tmp_path):
    """The basin masks keep the compression the published masks use."""
    output = tmp_path / "output.zarr"
    pipeline = CLI(output_path=str(output))
    ds = xr.Dataset({"value": ("time", np.arange(3))}).chunk({"time": 1})

    pipeline._collect(ds, compress=True)

    metadata = json.loads((output / "value" / ".zarray").read_text(encoding="utf-8"))
    assert metadata["compressor"]["id"] == "blosc"


def test_small_run_leaves_a_dataset_without_time_alone(tmp_path):
    """Not everything written here is a time series; basin masks are not."""
    output = tmp_path / "output.zarr"
    pipeline = CLI(output_path=str(output), small_run=True)
    ds = xr.Dataset({"mask": (("lat", "lon"), np.ones((4, 6)))})

    pipeline._collect(ds)

    assert xr.open_zarr(output)["mask"].shape == (4, 6)


def test_small_run_still_truncates_a_time_series(tmp_path):
    output = tmp_path / "output.zarr"
    pipeline = CLI(output_path=str(output), small_run=True)
    ds = xr.Dataset({"value": ("time", np.arange(50))}).chunk({"time": 1})

    pipeline._collect(ds)

    assert xr.open_zarr(output).sizes["time"] == 10


def test_wfo_source_path_is_not_forwarded_to_cluster(monkeypatch, tmp_path):
    cluster_calls = []

    def fake_init_cluster(cluster, **cluster_opts):
        cluster_calls.append((cluster, cluster_opts))
        return sentinel.client

    monkeypatch.setattr(main, "init_cluster", fake_init_cluster)

    pipeline = CLI(
        output_path=str(tmp_path / "unused.zarr"),
        cluster="local",
        wfo_source_path="s3://example/wfo.zarr",
        n_workers=4,
    )

    assert pipeline.wfo_source_path == "s3://example/wfo.zarr"
    assert pipeline.dask_client is sentinel.client
    assert cluster_calls == [("local", {"n_workers": 4})]
