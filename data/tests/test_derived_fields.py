# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
import xarray as xr
from ocean_preprocessing.derived_fields import add_derived_fields, main
from ocean_preprocessing.make_norm_datasets import normalization_fields


@pytest.fixture(params=["noleap", "julian"])
def source(request):
    time = xr.date_range(
        "2001-01-01", periods=3 * 73, freq="5D", calendar=request.param, use_cftime=True
    )
    shape = (len(time), 2, 2, 3)
    values = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    mask = np.ones(shape[1:], dtype=bool)
    mask[0, 0, 0] = False
    mask[1, 1, 0] = False
    ds = xr.Dataset(
        {
            "thetao": (("time", "lev", "y", "x"), values),
            "hfds": (("time", "y", "x"), values[:, 0], {"units": "W m-2"}),
        },
        coords={
            "time": time,
            "lev": (
                "lev",
                [2.5, 10.0],
                {
                    "units": "meters",
                    "long_name": "Depth at cell center",
                    "positive": "down",
                },
            ),
            "y": [-1.0, 1.0],
            "x": [0.0, 1.0, 2.0],
            "wetmask": (("lev", "y", "x"), mask),
            "areacello": (("y", "x"), [[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]]),
            "dz": ("lev", [5.0, 10.0]),
        },
        attrs={"source": "test"},
    )
    return ds.chunk({"time": 1})


@pytest.mark.parametrize("flat", [False, True])
def test_derived_fields_match_previous_computations(source, flat, tmp_path):
    if flat:
        for level in range(2):
            source[f"thetao_{level}"] = source.thetao.isel(lev=level, drop=True)
            source[f"mask_{level}"] = source.wetmask.isel(lev=level, drop=True)
        source = source.drop_vars(["thetao", "wetmask"])
    actual = add_derived_fields(source)
    for name in source.coords:
        assert actual[name].attrs == source[name].attrs
    assert actual.hfds_anomalies.dtype == np.float32
    assert actual.hfds_anomalies.chunks[0] == (1,) * source.sizes["time"]
    expected = source.hfds.groupby("time.dayofyear") - source.hfds.groupby(
        "time.dayofyear"
    ).mean("time")
    xr.testing.assert_allclose(
        actual.hfds_anomalies.reset_coords(drop=True), expected.reset_coords(drop=True)
    )
    assert "dayofyear" not in actual.coords
    assert actual.hfds_anomalies.attrs["units"] == "W m-2"
    masked = normalization_fields(source)
    expected_profiles = masked.weighted(source.areacello.fillna(0)).mean(("y", "x"))
    for name in masked.data_vars:
        xr.testing.assert_allclose(
            actual[f"profile_{name}"].reset_coords(drop=True),
            expected_profiles[name].reset_coords(drop=True),
        )
    norm = normalization_fields(actual)
    assert "hfds_anomalies" in norm
    assert not any(str(name).startswith("profile_") for name in norm.data_vars)
    expected_std = expected.where(
        source.get("wetmask", source.get("mask_0")).isel(lev=0, missing_dims="ignore")
        > 0
    ).std()
    xr.testing.assert_allclose(
        norm.hfds_anomalies.std(), expected_std.reset_coords(drop=True)
    )
    path = tmp_path / "output.zarr"
    actual.to_zarr(path, zarr_format=2, consolidated=True)
    assert json.loads((path / "hfds_anomalies" / ".zarray").read_text())["chunks"] == [
        1,
        2,
        3,
    ]


def test_backfill_is_repeatable_and_preserves_source(source, tmp_path):
    path = tmp_path / "source.zarr"
    source.to_zarr(path, zarr_format=2, consolidated=True)
    original = xr.open_zarr(path).load()
    main(str(path))
    first = xr.open_zarr(path).load()
    main(str(path))
    with xr.open_zarr(path) as actual:
        xr.testing.assert_identical(actual.load(), first)
        xr.testing.assert_identical(
            actual[list(original.data_vars)].drop_vars("areacello_weights"), original
        )
        assert actual.attrs == original.attrs


def test_prediction_profiles_use_existing_nan_mask(source):
    prediction = source[["thetao"]].where(source.wetmask).drop_vars("wetmask")
    out = add_derived_fields(prediction)
    assert "hfds_anomalies" not in out
    expected = prediction.thetao.weighted(prediction.areacello.fillna(0)).mean(
        ("y", "x")
    )
    xr.testing.assert_allclose(
        out.profile_thetao.reset_coords(drop=True), expected.reset_coords(drop=True)
    )


def test_backfill_refreshes_existing_buffer_chunked_profiles(source, tmp_path):
    path = tmp_path / "predictions.zarr"
    prediction = source[["thetao"]].where(source.wetmask).drop_vars("wetmask")
    enriched = add_derived_fields(prediction).chunk({"time": 10})
    enriched["profile_thetao"] = xr.zeros_like(enriched.profile_thetao)
    enriched.to_zarr(path, zarr_format=2, consolidated=True)
    main(str(path))
    with xr.open_zarr(path) as actual:
        expected = prediction.thetao.weighted(prediction.areacello.fillna(0)).mean(
            ("y", "x")
        )
        xr.testing.assert_allclose(
            actual.profile_thetao.reset_coords(drop=True),
            expected.reset_coords(drop=True),
        )
