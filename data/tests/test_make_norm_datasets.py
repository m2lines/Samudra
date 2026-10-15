# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr
from numcodecs import Blosc
from ocean_preprocessing.make_norm_datasets import write_zarr_with_retries


def test_write_zarr_uses_dask_task_retries():
    ds = Mock()
    client = Mock()
    delayed = object()
    ds.to_zarr.return_value = delayed

    write_zarr_with_retries(ds, "s3://test/output.zarr", client, write_retries=7)

    ds.to_zarr.assert_called_once_with(
        "s3://test/output.zarr",
        mode="w",
        zarr_format=2,
        consolidated=True,
        encoding=None,
        compute=False,
    )
    client.compute.assert_called_once_with(delayed, retries=7, sync=True)


@pytest.mark.parametrize("statistic", ["mean", "std"])
def test_compressed_normalization_round_trip(tmp_path: Path, statistic: str) -> None:
    output = tmp_path / f"CM4_{statistic}.zarr"
    source = xr.Dataset(
        {
            "thetao": (("time", "lev"), np.array([[1, 10], [3, 20]], dtype=np.float32)),
            "wfo": ("time", np.array([2, 4], dtype=np.float32)),
        },
        coords={"lev": [5.0, 20.0], "dz": ("lev", [10.0, 20.0])},
    )
    ds = getattr(source, statistic)("time")
    ds.attrs["normalization_steps"] = 2
    for field in ds.variables.values():
        field.encoding["compressor"] = Blosc(cname="lz4")
    client = Mock()
    client.compute.side_effect = lambda task, **kwargs: task.compute()

    write_zarr_with_retries(ds, str(output), client, 5, compress=True)

    for name, field in ds.variables.items():
        metadata = json.loads((output / name / ".zarray").read_text())
        if field.ndim:
            assert metadata["compressor"]["id"] == "blosc"
            assert metadata["compressor"]["cname"] == "zstd"
            assert metadata["compressor"]["clevel"] == 5
            assert metadata["compressor"]["shuffle"] == Blosc.SHUFFLE
        else:
            # Zarr v2 always leaves zero-dimensional scalars uncompressed.
            assert metadata["compressor"] is None
    with xr.open_zarr(output) as actual:
        xr.testing.assert_identical(actual.load(), ds)


def test_compact_normalization_keeps_depth_and_excludes_land() -> None:
    import numpy as np
    import xarray as xr
    from ocean_preprocessing.make_norm_datasets import normalization_fields

    source = xr.Dataset(
        {
            "thetao": (
                ("time", "lev", "y", "x"),
                [[[[1.0, 999.0]], [[10.0, 999.0]]], [[[3.0, 999.0]], [[20.0, 999.0]]]],
            ),
            "hfds": (("time", "y", "x"), [[[4.0, 999.0]], [[6.0, 999.0]]]),
        },
        coords={"wetmask": (("lev", "y", "x"), [[[True, False]], [[True, False]]])},
    )
    masked = normalization_fields(source)
    mean = masked.mean(["time", "y", "x"])
    std = masked.std(["time", "y", "x"])
    np.testing.assert_allclose(mean.thetao, [2.0, 15.0])
    np.testing.assert_allclose(std.thetao, [1.0, 5.0])
    assert mean.hfds.item() == 5.0
    assert std.hfds.item() == 1.0


def test_flat_normalization_uses_each_depth_mask() -> None:
    import xarray as xr
    from ocean_preprocessing.make_norm_datasets import normalization_fields

    source = xr.Dataset(
        {
            "thetao_1": (("time", "y", "x"), [[[999.0, 10.0]], [[999.0, 20.0]]]),
            "mask_0": (("y", "x"), [[True, True]]),
            "mask_1": (("y", "x"), [[False, True]]),
        }
    )
    masked = normalization_fields(source)
    assert set(masked.data_vars) == {"thetao_1"}
    assert masked.mean().thetao_1.item() == 15.0


@pytest.mark.parametrize("flattened", [False, True])
def test_normalization_with_depth_dependent_grid_coordinates(flattened: bool) -> None:
    import numpy as np
    import xarray as xr
    from ocean_preprocessing.make_norm_datasets import normalization_fields

    source = xr.Dataset(
        {
            "thetao": (
                ("time", "lev", "y", "x"),
                [[[[1.0, 999.0]], [[10.0, 999.0]]], [[[3.0, 999.0]], [[20.0, 999.0]]]],
                {"units": "degC"},
            ),
            "hfds": (("time", "y", "x"), [[[4.0, 999.0]], [[6.0, 999.0]]]),
        },
        coords={
            "lev": [5.0, 20.0],
            "wetmask": (("lev", "y", "x"), [[[True, False]], [[True, False]]]),
            "ocean_fraction": (("lev", "y", "x"), [[[1.0, 0.0]], [[0.5, 0.0]]]),
        },
        attrs={"grid_type": "gaussian"},
    )
    if flattened:
        for level in range(2):
            source[f"thetao_{level}"] = source.thetao.isel(lev=level, drop=True)
        source = source.drop_vars("thetao")

    masked = normalization_fields(source)
    means = masked.mean(["time", "y", "x"])
    stds = masked.std(["time", "y", "x"])
    assert masked.attrs == source.attrs
    assert means.hfds.item() == 5.0
    assert stds.hfds.item() == 1.0
    if flattened:
        assert means.thetao_0.item() == 2.0
        assert means.thetao_1.item() == 15.0
        assert stds.thetao_0.item() == 1.0
        assert stds.thetao_1.item() == 5.0
        assert masked.thetao_1.attrs["units"] == "degC"
    else:
        np.testing.assert_allclose(means.thetao, [2.0, 15.0])
        np.testing.assert_allclose(stds.thetao, [1.0, 5.0])
        xr.testing.assert_identical(masked.lev, source.lev)
        xr.testing.assert_identical(masked.ocean_fraction, source.ocean_fraction)
        assert masked.thetao.attrs["units"] == "degC"
