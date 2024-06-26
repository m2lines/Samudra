"""Preprocess arbitrary datasets to standardized naming, grids"""

from xgcm import Grid
import xarray as xr
import numpy as np
from xmip.preprocessing import combined_preprocessing
import cf_xarray
import xesmf as xe

def input_data_test(ds_input: xr.Dataset):
    """Test function to assert the format of the input dataset"""
    
    expected_data_vars = ['thetao', 'so', 'uo', 'vo', 'zos', 'hfds', 'tauvo', 'tauuo', 'sithick', 'siconc']
    # add the derived mean/std variables
    expected_data_vars_full = []
    for v in expected_data_vars:
        expected_data_vars_full.extend([f"{v}_mean", f"{v}_std"])
    
    expected_coords = ['areacello', 'dz', 'x', 'y', 'time', 'lev', 'lon', 'lat']
    if not set(ds_input.coords.keys()) == set(expected_coords):
        raise ValueError(f"Expected coords {set(expected_coords)} but found {list(set(ds_input.coords.keys()))}")

    expected_sizes = {'x':360, 'y':180, 'lev':19}
    for di,s in expected_sizes.items():
        assert ds_input.sizes[di] == s
    assert 'm2lines/ocean-emulators_git_hash' in ds_input.attrs.keys()
    


def rename(ds: xr.Dataset) -> xr.Dataset:
    """Rename variables and dimensions to CMOR standard names"""
    # TODO: how to detect non-CMIP datasets?
    return combined_preprocessing(ds)


def standardize_dataset(ds_ocean: xr.Dataset, ds_atmos: xr.Dataset) -> xr.Dataset:
    """Full wrapper that does
    1. Rename variables and dimensions to CMOR standard names
    2. Combine varibles if necessary
    3. Interpolate velocity to tracer cells
    4. Optional Filter the data
    5. Horizontal regridding
    6. Add metadata and provenance info
    """
    # Rename variables and dimensions to CMOR standard names
    ds_renamed = rename(ds)
    # Combine varibles if necessary
    # Interpolate velocity to tracer cells
    # Optional Filter the data
    # Horizontal regridding
    # Add metadata and provenance info
    return ds_renamed


#################### CMIP specific Code ###########################
def infer_vertical_cell_extent(ds: xr.Dataset, dz_name: str = "dz") -> xr.Dataset:
    """
    recomputes z* vertical cell extent according to

    thkcello is the nominal cell thickness in z* coordinates. The model actual thkcello is time-dependent and can be calculated as thkcello * ( deptho + zos ) / deptho
    """
    required_vars = ["thkcello", "deptho", "zos"]

    if not all(v in ds.variables for v in required_vars):
        raise ValueError(
            f"Could not find {set(required_vars)-set(ds.variables)} in datasset coords. Found {list(ds.coords)}"
        )

    ds = ds.assign_coords({dz_name: ds.thkcello * (ds.deptho + ds.zos) / ds.deptho})
    return ds


def cmip_vertical_outer_grid(ds: xr.Dataset) -> xr.Dataset:
    # TODO: Check if an outer grid position is already available (e.g. from combining tracer and vertical velocities in xmip.grids.something_staggered_grid

    # TODO: Ask alistair if it is ok to just use the nominal depth levels + extensive quantities?
    lev_outer = cf_xarray.bounds_to_vertices(ds["lev_bounds"], "bnds").rename(
        {"lev_vertices": "lev_outer"}
    )
    ds = ds.assign_coords(
        {
            "lev_outer": cf_xarray.bounds_to_vertices(ds["lev_bounds"], "bnds").rename(
                {"lev_vertices": "lev_outer"}
            )
        }
    )
    # set up an xgcm grid
    # FIXME: This should work with metadata!
    grid = Grid(
        ds,
        coords={"Z": {"center": "lev", "outer": "lev_outer"}},
        boundary="fill",
        autoparse_metadata=False,
    )
    return grid, ds


##################### General Code #################


def vertical_regrid(ds_raw: xr.Dataset, target_depth_bounds: np.ndarray) -> xr.Dataset:
    # reconstruct vertical bounds
    # TODO (this should be done outside to make this function more general)
    grid, ds = cmip_vertical_outer_grid(ds_raw)
    # split out the 2d variables
    ds_2d = xr.Dataset({var:ds[var] for var in ds.data_vars if not 'lev' in ds[var].dims})
    ds = ds.drop_vars(list(ds_2d.data_vars))
    
    dz = ds["dz"]
    ds_extensive = ds * dz

    ds_extensive_regridded = xr.Dataset()
    for var in ds_extensive.data_vars:
        # TODO: assert that lev is actually on this variable, otherwise what?
        ds_extensive_regridded[var] = grid.transform(
            ds_extensive[var],
            "Z",
            target_depth_bounds,
            target_data=ds.lev_outer,
            method="conservative",
        )

    # by default this is named after the 'target_data', but for the purpose of simplicity, lets rename this here
    ds_extensive_regridded = ds_extensive_regridded.rename({"lev_outer": "lev"})

    # Calculate the cell thickness of the target grid.
    dz_regridded = xr.DataArray(
        np.diff(target_depth_bounds),
        dims=["lev"],
        coords={"lev": ds_extensive_regridded.thetao.lev},
    )

    ds_regridded = ds_extensive_regridded / dz_regridded
    ds_regridded = ds_regridded.assign_coords(dz=dz_regridded)
    for co_name, co in ds.coords.items():
        if "lev" not in co.dims:
            ds_regridded = ds_regridded.assign_coords({co_name: co})
    ds_regridded = ds_regridded.drop("lev_outer")
    # merge the 2d variables back in
    ds_regridded = xr.merge([ds_regridded, ds_2d])
    ds_regridded.attrs = ds_raw.attrs
    return ds_regridded


# test:
# - What about the coordinates after? Are the non-depth ones the same (not weirdly scaled?).


def cmip_bounds_to_xesmf(ds: xr.Dataset, order=None):
    # the order is specific to the way I reorganized vertex order in xmip (if not passed we get the stripes in the regridded output!

    
    
    if not all(var in ds.variables for var in ["lon_b", "lat_b"]):
        ds = ds.assign_coords(
            lon_b=cf_xarray.bounds_to_vertices(
                ds.lon_verticies.load(), bounds_dim="vertex", order=order 
            ),
            lat_b=cf_xarray.bounds_to_vertices(
                ds.lat_verticies.load(), bounds_dim="vertex", order=order
            ),
        )
    return ds

def test_vertex_order(ds):
    # pick a point in the southern hemisphere to avoid curving nonsense
    p = {'x':slice(20,22), 'y':slice(20, 22)}
    ds_p = ds.isel(**p).squeeze()
    # get rid of all the unneccesary variables
    for var in ds_p.variables:
        if ('lev' in ds_p[var].dims) or ('time' in ds_p[var].dims) or (var in ['sub_experiment_label', 'variant_label']):
            ds_p = ds_p.drop_vars(var)
    ds_p = cmip_bounds_to_xesmf(ds_p, order=None) # woudld be nice if this could automatically get the settings provided to `cmip_bounds_to_xesmf`
    ds_p = ds_p.load().transpose(..., 'x', 'y','vertex')
    assert (ds_p.lon_b.diff('x_vertices')>0).all()
    assert (ds_p.lat_b.diff('y_vertices')>0).all()
    
def spatially_regrid(ds_source: xr.Dataset, ds_target: xr.Dataset, method:str="conservative", check=False) -> xr.Dataset:

    if check:
        for test_ds, name in [(ds_source, 'source dataset'), (ds_target, 'target dataset')]:
            try:
                test_vertex_order(test_ds)
            except:
                raise ValueError(f'something is wrong with the vertex order of the {name}')
    
    regridder = xe.Regridder(
        cmip_bounds_to_xesmf(ds_source),
        cmip_bounds_to_xesmf(ds_target),
        method,
        ignore_degenerate=True,
        unmapped_to_nan=True,
        periodic=True
    )
    return regridder(ds_source)
