# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


def load_script():
    path = Path(__file__).parents[1] / "scripts" / "audit_quarter_compact.py"
    spec = importlib.util.spec_from_file_location("quarter_compact_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_reference_packed_time_level_channel_order(tmp_path):
    # Physical dimension order is deliberately lev,time,y,x. Encoding each
    # coordinate in decimal digits makes expected values independent of xarray.
    lev, time, y, x = np.indices((3, 4, 2, 5))
    packed = (1000 * lev + 100 * time + 10 * y + x).astype(np.float32)
    packed[2, 3, 1, 4] = np.nan
    surface = packed[0] + 10000
    dataset = xr.Dataset(
        {
            "thetao": (("lev", "time", "y", "x"), packed),
            "so": (("time", "lev", "y", "x"), (packed + 20000).transpose(1, 0, 2, 3)),
            "zos": (("time", "y", "x"), surface),
        }
    )
    path = tmp_path / "compact.zarr"
    dataset.to_zarr(path, consolidated=True)
    with xr.open_zarr(path, chunks=None, decode_times=False) as opened:
        result = load_script().reference_read(
            opened,
            [3, 0, 3],
            [("thetao", 2), ("zos", None), ("so", 1), ("thetao", 0), ("thetao", 2)],
        )
    expected = np.empty((3, 5, 2, 5), np.float32)
    for ti, physical_time in enumerate([3, 0, 3]):
        for ci, (offset, level) in enumerate(
            [(0, 2), (10000, 0), (20000, 1), (0, 0), (0, 2)]
        ):
            for yi in range(2):
                for xi in range(5):
                    expected[ti, ci, yi, xi] = (
                        offset + level * 1000 + physical_time * 100 + yi * 10 + xi
                    )
    expected[[0, 2], 0, 1, 4] = np.nan
    expected[[0, 2], 4, 1, 4] = np.nan
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 5, 2, 5)
    assert result.dtype == np.float32
