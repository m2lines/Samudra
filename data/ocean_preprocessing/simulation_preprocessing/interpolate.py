# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import xarray as xr
from xgcm import Grid


def interpolate_to_cell_centers(
    ds: xr.Dataset,
    like: xr.DataArray,
    grid: Grid,
):
    """Interplate variables defined on cell boundaries to cell centers.

    Args:
        ds: Input dataset.
        like: The data array to use as a template for the interpolation.
        grid: The grid object which will perform the interplation.

    """
    xh, yh = grid.axes["X"].coords["center"], grid.axes["Y"].coords["center"]
    xq, yq = grid.axes["X"].coords["right"], grid.axes["Y"].coords["right"]
    ds_interpolated = xr.Dataset()
    for var in ds.data_vars:
        da = ds[var]
        if set([xh, yh]).issubset(da.dims):
            ds_interpolated[var] = da
        elif xq in da.dims or yq in da.dims:
            # fill the velocities with 0 before interpolation to avoid mismatches in nans
            ds_interpolated[var] = grid.interp_like(da.fillna(0), like)
        if var in ds_interpolated:
            ds_interpolated[var].attrs = da.attrs

    return ds_interpolated
