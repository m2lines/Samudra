import xarray as xr


def _pick_first_element_of_missing_dims(mask: xr.DataArray, data: xr.DataArray):
    missing_dims = [di for di in mask.dims if di not in data.dims]
    if len(missing_dims) == 0:
        return mask
    else:
        return mask.isel({di: 0 for di in missing_dims})


def apply_mask(ds: xr.Dataset, mask: xr.DataArray):
    """applies mask to same and lower dimensional data"""
    ds_out = xr.Dataset(attrs=ds.attrs)
    for var in ds.data_vars:
        data = ds[var]
        mask_pruned = _pick_first_element_of_missing_dims(mask, data)
        ds_out[var] = data.where(mask_pruned)
    return ds_out


def assert_mask_match(ds: xr.Dataset, mask: xr.DataArray):
    """Assert that nans at a sample time step are consistent with a mask (mask True or 1 indicates not nan)"""
    for var in ds.data_vars:
        data_test = ds[var]
        # make sure that 2d variables are only tested agains 2d wetmask
        mask_test = _pick_first_element_of_missing_dims(mask, data_test)
        print("data_test", data_test)
        print("mask_test", mask_test)
        if not (data_test.notnull() == mask_test).all():
            raise ValueError(
                f"Wetmask does not match between `ds` and `wetmask` for variable {var}!"
            )
