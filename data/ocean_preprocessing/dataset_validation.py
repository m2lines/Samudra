# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import xarray as xr

from ocean_preprocessing.schema import (
    OM4_3D_VARS,
    OM4_OPTIONAL_2D_VARS,
    OM4_REQUIRED_2D_VARS,
    ds_prediction_coords_schema,
    ds_prediction_schema,
    ds_processed_coords_schema,
    ds_processed_schema,
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


_COMMON_REQUIRED_COORDS = {
    "areacello",
    "dz",
    "lat",
    "lat_b",
    "lev",
    "lon",
    "lon_b",
    "time",
    "x",
    "y",
}
_REQUIRED_ATTRS = {
    "grid_type",
    "m2lines/cli_args",
    "m2lines/date_created",
    "m2lines/ocean_emulators_git_hash",
    "m2lines/samudra_git_hash",
}
_SUPPORTED_GRID_TYPES = {"gaussian", "tripolar"}


def _validate_common_input_contract(ds_input: xr.Dataset) -> None:
    """Validate metadata shared by flattened and depth-resolved outputs."""
    grid_type = ds_input.attrs.get("grid_type")
    if grid_type not in _SUPPORTED_GRID_TYPES:
        raise ValueError(
            f"Output grid_type must be one of {sorted(_SUPPORTED_GRID_TYPES)}; "
            f"got {grid_type!r}."
        )

    required_coords = set(_COMMON_REQUIRED_COORDS)
    if grid_type == "gaussian":
        required_coords.add("ocean_fraction")

    missing_coords = required_coords - set(ds_input.coords)
    if missing_coords:
        raise ValueError(
            f"Output is missing required coordinates: {sorted(missing_coords)}"
        )

    expected_coord_dims = {
        "areacello": ("y", "x"),
        "lat": ("y", "x"),
        "lat_b": ("y_b", "x_b"),
        "lev": ("lev",),
        "lon": ("y", "x"),
        "lon_b": ("y_b", "x_b"),
        "time": ("time",),
        "x": ("x",),
        "y": ("y",),
    }
    if "ocean_fraction" in required_coords:
        expected_coord_dims["ocean_fraction"] = ("lev", "y", "x")
    for coord, expected_dims in expected_coord_dims.items():
        if ds_input[coord].dims != expected_dims:
            raise ValueError(
                f"Output coordinate {coord!r} has dimensions "
                f"{ds_input[coord].dims}; expected {expected_dims}."
            )

    allowed_dz_dims = {("lev",), ("lev", "y", "x")}
    if ds_input.dz.dims not in allowed_dz_dims:
        raise ValueError(
            f"Output coordinate 'dz' has dimensions {ds_input.dz.dims}; "
            f"expected one of {sorted(allowed_dz_dims)}."
        )

    for bound, center in (("x_b", "x"), ("y_b", "y")):
        if ds_input.sizes[bound] != ds_input.sizes[center] + 1:
            raise ValueError(
                f"Output dimension {bound!r} must be one larger than {center!r}."
            )

    missing_attrs = _REQUIRED_ATTRS - set(ds_input.attrs)
    if missing_attrs:
        raise ValueError(
            f"Output is missing required attributes: {sorted(missing_attrs)}"
        )


def _validate_dims(
    ds_input: xr.Dataset, variable_names: set[str], expected_dims: tuple[str, ...]
) -> None:
    for var in variable_names:
        if ds_input[var].dims != expected_dims:
            raise ValueError(
                f"Output variable {var!r} has dimensions {ds_input[var].dims}; "
                f"expected {expected_dims}."
            )


def _validate_dtype(
    ds_input: xr.Dataset, variable_names: set[str], expected_dtype: str
) -> None:
    dtype = np.dtype(expected_dtype)
    for var in variable_names:
        if ds_input[var].dtype != dtype:
            raise ValueError(
                f"Output variable {var!r} has dtype {ds_input[var].dtype}; "
                f"expected {dtype}."
            )


### For input datasets (with generic steps like regridding, filtering, etc applied) ###
def ds_input_validate(ds_input: xr.Dataset, deep=False):
    """Validate a resolution-independent, depth-resolved input dataset."""
    _validate_common_input_contract(ds_input)

    required_vars = set(OM4_REQUIRED_2D_VARS) | set(OM4_3D_VARS)
    missing_vars = required_vars - set(ds_input.data_vars)
    if missing_vars:
        raise ValueError(
            f"Output is missing required variables: {sorted(missing_vars)}"
        )

    if "wetmask" not in ds_input.coords:
        raise ValueError("Output is missing required coordinate: 'wetmask'")

    present_2d_vars = set(OM4_REQUIRED_2D_VARS) | (
        set(OM4_OPTIONAL_2D_VARS) & set(ds_input.data_vars)
    )
    _validate_dims(ds_input, present_2d_vars, ("time", "y", "x"))
    _validate_dims(ds_input, set(OM4_3D_VARS), ("time", "lev", "y", "x"))
    _validate_dtype(ds_input, present_2d_vars | set(OM4_3D_VARS), "float32")
    if ds_input.wetmask.dims != ("lev", "y", "x"):
        raise ValueError(
            f"Output coordinate 'wetmask' has dimensions {ds_input.wetmask.dims}; "
            "expected ('lev', 'y', 'x')."
        )
    _validate_dtype(ds_input, {"wetmask"}, "bool")

    if deep:
        _nan_test_deep(ds_input)


def ds_flattened_input_validate(ds_input: xr.Dataset) -> None:
    """Validate the resolution-independent contract of a flattened dataset."""
    _validate_common_input_contract(ds_input)

    levels = range(ds_input.sizes["lev"])
    required_vars = set(OM4_REQUIRED_2D_VARS)
    required_vars.update(f"{var}_{level}" for var in OM4_3D_VARS for level in levels)
    required_vars.update(f"mask_{level}" for level in levels)
    missing_vars = required_vars - set(ds_input.data_vars)
    if missing_vars:
        raise ValueError(
            f"Output is missing required variables: {sorted(missing_vars)}"
        )

    allowed_vars = required_vars | set(OM4_OPTIONAL_2D_VARS)
    unexpected_vars = set(ds_input.data_vars) - allowed_vars
    if unexpected_vars:
        raise ValueError(
            f"Output contains undeclared variables: {sorted(unexpected_vars)}"
        )

    mask_vars = {var for var in required_vars if var.startswith("mask_")}
    _validate_dims(ds_input, mask_vars, ("y", "x"))
    _validate_dims(
        ds_input,
        (required_vars - mask_vars)
        | (set(OM4_OPTIONAL_2D_VARS) & set(ds_input.data_vars)),
        ("time", "y", "x"),
    )
    _validate_dtype(ds_input, mask_vars, "bool")
    _validate_dtype(
        ds_input,
        set(ds_input.data_vars) & (allowed_vars - mask_vars),
        "float32",
    )


### for the final prediction output
def ds_prediction_validate(ds_prediction: xr.Dataset, deep=False):
    warnings.warn("This checks nothing yet")
    ds_prediction_schema.validate(ds_prediction)
    ds_prediction_coords_schema.validate(ds_prediction.coords)
    if deep:
        _nan_test_deep(ds_prediction)
