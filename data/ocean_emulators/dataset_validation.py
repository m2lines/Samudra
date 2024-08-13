from ocean_emulators.schema import ds_processed_coords_schema, ds_processed_schema, ds_input_coords_schema, ds_input_schema
from ocean_emulators.utils import split_2d_3d, ensure_nan_consistency
import xarray as xr

def nan_test_deep(ds_input: xr.Dataset):
    """Expensive tests that compute on the entire dataset"""
    ds_nan_test_2d, ds_nan_test_3d = split_2d_3d(ds_input)
    print("2D consistency check")
    ensure_nan_consistency(ds_nan_test_2d, "2D nan consistency check")

    print("3D consistency check")
    ensure_nan_consistency(ds_nan_test_3d, "3D nan consistency check")


#### For processed (model specific) datasets
def ds_processed_validate(ds_processed: xr.Dataset, deep=False):
    """Validation function for the preprocessing stage"""
    ds_processed_schema.validate(ds_processed)
    ds_processed_coords_schema.validate(ds_processed.coords) # this should be part of the dataset validation (maybe raise an issue/pr?)
    if deep:
        nan_test_deep(ds_processed)
        

### For input datasets (with generic steps like regridding, filtering, etc applied) ###
def ds_input_validate(ds_input: xr.Dataset, deep=False):
    """Test function to assert the format of the input dataset.
    If `deep` is True, this will run expensive compuation across the entire dataset."""
    ds_input_schema.validate(ds_input)
    ds_input_coords_schema.validate(ds_input.coords)
    if deep:
        nan_test_deep(ds_input)