from ocean_emulators.schema import ds_processed_coords_schema, ds_processed_schema
from ocean_emulators.utils import split_2d_3d, ensure_nan_consistency
from xarray_schema.components import DimsSchema
import xarray as xr

def input_data_test_deep(ds_input: xr.Dataset):
    """Expensive tests that compute on the entire dataset"""
    ds_nan_test_2d, ds_nan_test_3d = split_2d_3d(ds_input)
    print("2D consistency check")
    ensure_nan_consistency(ds_nan_test_2d, "2D nan consistency check")

    print("3D consistency check")
    ensure_nan_consistency(ds_nan_test_3d, "3D nan consistency check")


#### For processed (model specific) datasets
def ds_processed_validate(ds_processed: xr.Dataset, deep=False):
    ds_processed_schema.validate(ds_processed)
    ds_processed_coords_schema.validate(ds_processed.coords)
    if deep:
        input_data_test_deep(ds_processed)
        

### For input datasets (with generic steps like regridding, filtering, etc applied) ###
def ds_input_validate(ds_input: xr.Dataset, deep=False):
    """Test function to assert the format of the input dataset.
    If `deep` is True, this will run expensive compuation across the entire dataset."""

    expected_data_vars = [
        "thetao",
        "so",
        "uo",
        "vo",
        "zos",
        "hfds",
        "tauvo",
        "tauuo",
        "sithick",
        "siconc",
    ]
    # add the derived mean/std variables
    expected_data_vars_full = []
    for v in expected_data_vars:
        expected_data_vars_full.extend([f"{v}_mean", f"{v}_std"])

    expected_coords = [
        "areacello",
        "dz",
        "x",
        "y",
        "time",
        "lev",
        "lon",
        "lat",
        "wetmask",
    ]
    if not set(ds_input.coords.keys()) == set(expected_coords):
        raise ValueError(
            f"Expected coords {set(expected_coords)} but found {list(set(ds_input.coords.keys()))}"
        )

    expected_sizes = {"x": 360, "y": 180, "lev": 19}
    for di, s in expected_sizes.items():
        if not ds_input.sizes[di] == s:
            raise ValueError(
                f"Expected size ({s}) for dimension {di}, but got {ds_input.sizes[di]}"
            )

    check_attrs = ["m2lines/ocean-emulators_git_hash"]
    for attr in check_attrs:
        if attr not in ds_input.attrs.keys():
            raise ValueError(f"Could not find {attr} in dataset attributes")

    # asser shape of coordinates
    dims_expected_on_coords = {
        "wetmask": ["x", "y", "lev"],
        "areacello": ["x", "y"],
        "lon": ["x", "y"],
        "lat": ["x", "y"],
        "dz": ["lev"],
    }
    for co, expected_dims in dims_expected_on_coords.items():
        if not set(expected_dims) == set(ds_input[co].dims):
            raise ValueError(
                f"Expected dimensions {set(expected_dims)} on {co}, but got {set(ds_input[co].dims)}"
            )

    if deep:
        input_data_test_deep(ds_input)