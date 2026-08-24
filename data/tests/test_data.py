# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Validation tests for preprocessing datasets."""

import numpy as np
import pytest
import xarray as xr
from ocean_preprocessing.dataset_validation import (
    ds_flattened_input_validate,
    ds_input_validate,
    ds_prediction_validate,
    ds_processed_validate,
)

from tests.data import input_data, prediction_data, processed_data  # noqa


def test_processed_data(processed_data):
    ds_processed_validate(processed_data)


def test_processed_data_allows_snapshot_water_flux(processed_data):
    processed_data["wfo"] = processed_data["hfds"]
    ds_processed_validate(processed_data)


def test_input_data(input_data):
    ds_input_validate(input_data)


def test_prediction_data(prediction_data):
    ds_prediction_validate(prediction_data)


def _flattened_input(*, include_wfo: bool = True) -> xr.Dataset:
    time, levels, ny, nx = 2, 2, 3, 4
    dynamic = np.ones((time, ny, nx), dtype="float32")
    data_vars = {
        name: (("time", "y", "x"), dynamic)
        for name in ["hfds", "tauuo", "tauvo", "zos"]
    }
    if include_wfo:
        data_vars["wfo"] = (("time", "y", "x"), dynamic)
    for name in ["so", "thetao", "uo", "vo"]:
        for level in range(levels):
            data_vars[f"{name}_{level}"] = (("time", "y", "x"), dynamic)
    for level in range(levels):
        data_vars[f"mask_{level}"] = (
            ("y", "x"),
            np.ones((ny, nx), dtype=bool),
        )

    return xr.Dataset(
        data_vars,
        coords={
            "time": range(time),
            "lev": ("lev", [2.5, 10.0]),
            "dz": ("lev", [5.0, 10.0]),
            "x": range(nx),
            "y": range(ny),
            "x_b": range(nx + 1),
            "y_b": range(ny + 1),
            "lon": (("y", "x"), np.zeros((ny, nx))),
            "lat": (("y", "x"), np.zeros((ny, nx))),
            "lon_b": (("y_b", "x_b"), np.zeros((ny + 1, nx + 1))),
            "lat_b": (("y_b", "x_b"), np.zeros((ny + 1, nx + 1))),
            "areacello": (("y", "x"), np.ones((ny, nx))),
            "ocean_fraction": (
                ("lev", "y", "x"),
                np.ones((levels, ny, nx)),
            ),
        },
        attrs={
            "grid_type": "gaussian",
            "m2lines/cli_args": "test",
            "m2lines/date_created": "2026-08-24T00:00:00",
            "m2lines/ocean_emulators_git_hash": "https://example.test/commit/a",
            "m2lines/samudra_git_hash": "https://example.test/commit/a",
        },
    )


@pytest.mark.parametrize("include_wfo", [False, True])
def test_flattened_input_contract_is_resolution_independent(include_wfo):
    ds_flattened_input_validate(_flattened_input(include_wfo=include_wfo))


def test_flattened_input_contract_rejects_undeclared_variable():
    ds = _flattened_input()
    ds["hfgeou"] = ds["hfds"]

    with pytest.raises(ValueError, match="undeclared variables.*hfgeou"):
        ds_flattened_input_validate(ds)


def test_flattened_input_contract_requires_provenance():
    ds = _flattened_input()
    del ds.attrs["m2lines/samudra_git_hash"]

    with pytest.raises(ValueError, match="required attributes.*samudra_git_hash"):
        ds_flattened_input_validate(ds)
