"""Preprocess arbitrary datasets to standardized naming, grids"""

from xgcm import Grid
import xarray as xr
import numpy as np
from xmip.preprocessing import combined_preprocessing
import cf_xarray
import xesmf as xe

def rename(ds: xr.Dataset) -> xr.Dataset:
    """Rename variables and dimensions to CMOR standard names"""
    # TODO: how to detect non-CMIP datasets?
    return combined_preprocessing(ds)


def standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
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
def infer_vertical_cell_extent(ds:xr.Dataset, dz_name:str='dz') -> xr.Dataset:
    """
    recomputes z* vertical cell extent according to
    
    thkcello is the nominal cell thickness in z* coordinates. The model actual thkcello is time-dependent and can be calculated as thkcello * ( deptho + zos ) / deptho
    """
    required_vars = ['thkcello', 'deptho', 'zos']
    
    if not all(v in ds.variables for v in required_vars):
        raise ValueError(f"Could not find {set(required_vars)-set(ds.variables)} in datasset coords. Found {list(ds.coords)}")

    ds = ds.assign_coords({dz_name:ds.thkcello * (ds.deptho + ds.zos ) / ds.deptho})
    return ds

def vertical_staggered_grid():
    #Should contain all the stuff that is in `vertical_regrid` below
    pass

##################### General Code #################


def vertical_regrid(ds:xr.Dataset, target_depth_bounds: np.ndarray) -> xr.Dataset: 
    
    # reconstruct vertical bounds
    # TODO: Ask alistair if it is ok to just use the nominal depth levels + extensive quantities?
    lev_outer = cf_xarray.bounds_to_vertices(ds['lev_bounds'], 'bnds').rename({'lev_vertices':'lev_outer'})
    ds = ds.assign_coords({'lev_outer':cf_xarray.bounds_to_vertices(ds['lev_bounds'], 'bnds').rename({'lev_vertices':'lev_outer'})})
    # set up an xgcm grid
    # FIXME: This should work with metadata!
    grid = Grid(
        ds,
        coords={"Z": {"center": "lev", "outer": "lev_outer"}},
        boundary="fill",
        autoparse_metadata=False,
    )
    dz = ds['dz']
    ds_extensive = ds * dz

    ds_extensive_regridded = xr.Dataset()
    for var in ds_extensive.data_vars:
        # TODO: assert that lev is actually on this variable, otherwise what?
        ds_extensive_regridded[var] = grid.transform(ds_extensive[var], 'Z', target_depth_bounds, target_data=ds.lev_outer, method='conservative')
    
    # by default this is named after the 'target_data', but for the purpose of simplicity, lets rename this here
    ds_extensive_regridded = ds_extensive_regridded.rename({'lev_outer':'lev'})
    
    # Calculate the cell thickness of the target grid. 
    dz_regridded = xr.DataArray(np.diff(target_depth_bounds), dims=['lev'], coords={'lev':ds_extensive_regridded.thetao.lev})
    
    ds_regridded = ds_extensive_regridded / dz_regridded
    ds_regridded = ds_regridded.assign_coords(dz=dz_regridded)
    for co_name, co in ds.coords.items():
        if 'lev' not in co.dims:
            ds_regridded = ds_regridded.assign_coords({co_name:co})
    ds_regridded = ds_regridded.drop('lev_outer')
    return ds_regridded

# test:
# - What about the coordinates after? Are the non-depth ones the same (not weirdly scaled?).

def cmip_bounds_to_xesmf(ds:xr.Dataset):
    if not all(var in ds.variables for var in ['lon_b', 'lat_b']):
        ds = ds.assign_coords(
            lon_b=cf_xarray.bounds_to_vertices(ds.lon_verticies.load(), bounds_dim='vertex'),
            lat_b=cf_xarray.bounds_to_vertices(ds.lat_verticies.load(), bounds_dim='vertex')
        )
    return ds
    

def spatially_regrid(ds_source:xr.Dataset, ds_target:xr.Dataset) -> xr.Dataset:
    regridder = xe.Regridder(
        cmip_bounds_to_xesmf(ds_source),
        cmip_bounds_to_xesmf(ds_target),
        'conservative',
        ignore_degenerate=True, 
    )
    return regridder(ds_source)