import xarray as xr
import numpy as np
from tests.data import input_data  # noqa # Might want to put these in conftest.py (see https://stackoverflow.com/questions/73191533/using-conftest-py-vs-importing-fixtures-from-dedicate-modules)
from ocean_emulators.utils import assert_mask_match, apply_mask
import pytest


@pytest.mark.parametrize("mask_dtype", [None, int])
def test_assert_mask_match(mask_dtype):
    x = np.arange(0, 2)
    y = np.arange(0, 3)
    z = np.arange(0, 4)
    data_2d = xr.DataArray(
        np.random.random([2, 3]), dims=["x", "y"], coords={"x": x, "y": y}
    )
    data_3d = xr.DataArray(
        np.random.random([2, 3, 4]),
        dims=["x", "y", "z"],
        coords={"x": x, "y": y, "z": z},
    )
    mask_3d = data_3d > 0.25
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
