import xarray as xr
from xgcm import Grid


def interpolate_to_cell_centers(
    ds: xr.Dataset,
    like: xr.DataArray,
    center_dim_names=("xh", "yh"),
    boundary_dim_names=("xq", "yq"),
):
    """Interplate variables defined on cell boundaries to cell centers.

    Args:
        ds: Input dataset.
        like:
        center_dim_names: Tuple of dim names corresponding to cell centers.
        boundary_dim_names: Tuple of dim names corresponding to cell boundaries.

    """

    xh, yh = center_dim_names
    xq, yq = boundary_dim_names
    if ds[xh].size == ds[xq].size:
        # outputs written in "non-symmetric" mode
        # see https://xgcm.readthedocs.io/en/latest/xgcm-examples/03_MOM6.html#xgcm-grid-definition
        grid_coords = {
            "X": {"center": xh, "right": xq},
            "Y": {"center": yh, "right": yq},
        }
    else:
        # outputs written in "symmetric" mode
        # periodicity is already 'built in with the outer coords'.
        # NOTE: This would not be sufficient to interpolate tracer points back!
        # For the velocity we need to extend, not pad otherwise the QC plots in the rotation will not work!
        grid_coords = {
            "X": {"center": xh, "outer": xq},
            "Y": {"center": yh, "outer": yq},
        }

    grid = Grid(
        ds,
        coords=grid_coords,
        boundary={"X": None, "Y": "extend"},
    )
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
