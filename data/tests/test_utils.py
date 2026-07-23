# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr
from ocean_preprocessing.utils import (
    apply_mask,
    assert_mask_match,
    ensure_nan_consistency,
)

from tests.data import (
    input_data,  # noqa # Might want to put these in conftest.py (see https://stackoverflow.com/questions/73191533/using-conftest-py-vs-importing-fixtures-from-dedicate-modules)
)


@pytest.mark.parametrize("mask_dtype", [None, int])
def test_assert_mask_match(mask_dtype):
    x = np.arange(0, 2)
    y = np.arange(0, 3)
    z = np.arange(0, 4)
    data_2d = xr.DataArray(
        np.arange(6).reshape(2, 3), dims=["x", "y"], coords={"x": x, "y": y}
    )
    data_3d = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=["x", "y", "z"],
        coords={"x": x, "y": y, "z": z},
    )
    mask_3d = data_3d % 3 == 0
    if mask_dtype is not None:
        mask_3d = mask_3d.astype(mask_dtype)
    ds = xr.Dataset({"3d": data_3d, "2d": data_2d})
    data_3d_masked = data_3d.where(mask_3d)
    data_2d_masked = data_2d.where(mask_3d.isel(z=0))
    ds_masked = xr.Dataset({"3d": data_3d_masked, "2d": data_2d_masked})

    assert_mask_match(ds_masked, mask_3d)

    # should raise on the unmasked dataset
    with pytest.raises(ValueError):
        assert_mask_match(ds, mask_3d)


def test_apply_mask(input_data):
    input_data_masked = apply_mask(input_data, input_data.wetmask)
    # assure that every variable in the dataset has the same shape as before
    for var in input_data.data_vars:
        assert input_data[var].shape == input_data_masked[var].shape
        assert input_data[var].dims == input_data_masked[var].dims
        assert input_data[var].coords.keys() == input_data_masked[var].coords.keys()
        assert input_data[var].attrs.keys() == input_data_masked[var].attrs.keys()


@pytest.mark.parametrize("dask", [True, False])
def test_ensure_nan_consistency(dask):
    ds = xr.Dataset(
        {
            "a": xr.DataArray(np.random.rand(4, 5, 6), dims=["x", "y", "time"]),
            "b": xr.DataArray(np.random.rand(4, 5, 6), dims=["x", "y", "time"]),
        }
    )

    if dask:
        ds = ds.chunk({"time": 1})

    # masking the same thing in each variable should pass
    ensure_nan_consistency(ds.where(ds.x > 1))

    # create a mismatch between variables
    ds_mismatch_variables = xr.Dataset(
        {"a": ds.a.where(ds.x > 2), "b": ds.b.where(ds.x <= 2)}
    )

    msg = "Found non-matching nan values between variables on the first time step."
    with pytest.raises(ValueError, match=msg):
        ensure_nan_consistency(ds_mismatch_variables)

    # create a nan mismatch in only one of the variables in time
    ds_mismatch_time_single = xr.Dataset({"a": ds.a, "b": ds.b.where(ds.time > 1)})
    msg = "None:Found nonmatching nans compared to first time step in the following indexes*"
    with pytest.raises(ValueError, match=msg):
        ensure_nan_consistency(ds_mismatch_time_single)
