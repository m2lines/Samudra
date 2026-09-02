# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
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
