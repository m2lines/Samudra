# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import xarray as xr

from ocean_preprocessing.schema import (
    ds_input_coords_schema,
    ds_input_schema,
    ds_prediction_coords_schema,
    ds_prediction_schema,
    ds_processed_coords_schema,
    ds_processed_schema,
    optional_vars_2d,
    vars_2d,
    vars_3d,
)
from ocean_preprocessing.utils import ensure_nan_consistency, split_2d_3d


def _nan_test_deep(ds_input: xr.Dataset):
    """Expensive tests that compute on the entire dataset."""
    ds_nan_test_2d, ds_nan_test_3d = split_2d_3d(ds_input)
    print("2D consistency check")
    ensure_nan_consistency(ds_nan_test_2d, "2D nan consistency check")

    print("3D consistency check")
    ensure_nan_consistency(ds_nan_test_3d, "3D nan consistency check")


#### For processed (model specific) datasets
def ds_processed_validate(ds_processed: xr.Dataset, deep=False):
    """Validation function for the preprocessing stage."""
    ds_processed_schema.validate(ds_processed)
    ds_processed_coords_schema.validate(
        ds_processed.coords
    )  # this should be part of the dataset validation (maybe raise an issue/pr?)
    if deep:
        _nan_test_deep(ds_processed)


### For input datasets (with generic steps like regridding, filtering, etc applied) ###
def ds_input_validate(ds_input: xr.Dataset, deep=False):
    """Test function to assert the format of the input dataset.
    If `deep` is True, this will run expensive compuation across the entire dataset.
    """
    ds_input_schema.validate(ds_input)
    ds_input_coords_schema.validate(ds_input.coords)
    # ds_input_attrs_schema.validate(ds_input.attrs) # this does not work as I want, replace with manual check for now
    required_attrs_keys = ["m2lines/ocean_emulators_git_hash"]
    for rk in required_attrs_keys:
        assert rk in ds_input.attrs.keys()
    if deep:
        _nan_test_deep(ds_input)


def ds_flattened_input_validate(ds_input: xr.Dataset) -> None:
    """Validate the resolution-independent contract of a publishable dataset."""
    required_coords = {
        "areacello",
        "dz",
        "lat",
        "lat_b",
        "lev",
        "lon",
        "lon_b",
        "ocean_fraction",
        "time",
        "x",
        "y",
    }
    missing_coords = required_coords - set(ds_input.coords)
    if missing_coords:
        raise ValueError(
            f"Output is missing required coordinates: {sorted(missing_coords)}"
        )

    for bound, center in (("x_b", "x"), ("y_b", "y")):
        if ds_input.sizes[bound] != ds_input.sizes[center] + 1:
            raise ValueError(
                f"Output dimension {bound!r} must be one larger than {center!r}."
            )

    levels = range(ds_input.sizes["lev"])
    required_vars = set(vars_2d)
    required_vars.update(f"{var}_{level}" for var in vars_3d for level in levels)
    required_vars.update(f"mask_{level}" for level in levels)
    missing_vars = required_vars - set(ds_input.data_vars)
    if missing_vars:
        raise ValueError(
            f"Output is missing required variables: {sorted(missing_vars)}"
        )

    allowed_vars = required_vars | set(optional_vars_2d)
    unexpected_vars = set(ds_input.data_vars) - allowed_vars
    if unexpected_vars:
        raise ValueError(
            f"Output contains undeclared variables: {sorted(unexpected_vars)}"
        )

    for var in required_vars | (set(optional_vars_2d) & set(ds_input.data_vars)):
        expected_dims = ("y", "x") if var.startswith("mask_") else ("time", "y", "x")
        if ds_input[var].dims != expected_dims:
            raise ValueError(
                f"Output variable {var!r} has dimensions {ds_input[var].dims}; "
                f"expected {expected_dims}."
            )

    required_attrs = {
        "grid_type",
        "m2lines/cli_args",
        "m2lines/date_created",
        "m2lines/ocean_emulators_git_hash",
        "m2lines/samudra_git_hash",
    }
    missing_attrs = required_attrs - set(ds_input.attrs)
    if missing_attrs:
        raise ValueError(
            f"Output is missing required attributes: {sorted(missing_attrs)}"
        )


### for the final prediction output
def ds_prediction_validate(ds_prediction: xr.Dataset, deep=False):
    warnings.warn("This checks nothing yet")
    ds_prediction_schema.validate(ds_prediction)
    ds_prediction_coords_schema.validate(ds_prediction.coords)
    if deep:
        _nan_test_deep(ds_prediction)
