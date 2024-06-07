"""Preprocess arbitrary datasets to standardized naming, grids"""

import xarray as xr
from xmip.preprocessing import combined_preprocessing


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
