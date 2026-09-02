# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr
from ocean_preprocessing.simulation_preprocessing.gfdl_om4 import (
    normalize_vertical_coords,
    select_om4_variables,
    vertical_cell_metadata,
)


def _ds_with_vertical(names):
    """Build a tiny dataset whose vertical dimension coordinate(s) are ``names``."""
    coords = {name: (name, np.arange(3)) for name in names}
    return xr.Dataset(coords=coords)


def test_normalize_renames_z_l_without_z_i():
    # Snapshot sources expose z_l (cell centers) but no z_i; z_l must still
    # become lev so downstream rechunking on "lev" succeeds.
    ds = _ds_with_vertical(["z_l"])
    out = normalize_vertical_coords(ds)
    assert "lev" in out.coords
    assert "z_l" not in out.coords


def test_normalize_renames_both_when_present():
    # Averaged sources carry both interfaces and centers.
    ds = _ds_with_vertical(["z_l", "z_i"])
    out = normalize_vertical_coords(ds)
    assert {"lev", "ilev"} <= set(out.coords)
    assert "z_l" not in out.coords and "z_i" not in out.coords


def test_normalize_is_noop_without_raw_vertical_coords():
    # Already-normalized data passes through untouched.
    ds = _ds_with_vertical(["lev"])
    out = normalize_vertical_coords(ds)
    assert "lev" in out.coords


def test_averaged_vertical_cell_metadata_matches_processed_schema_dtype():
    ds = xr.Dataset(
        coords={
            "lev": ("lev", [2.5, 10.0]),
            "ilev": ("ilev", [0, 5, 15]),
        }
    )

    dz, ilev = vertical_cell_metadata(ds)

    assert dz.dims == ("lev",)
    assert dz.dtype == np.dtype("float64")
    assert ilev.dtype == np.dtype("float64")
    np.testing.assert_array_equal(dz, [5.0, 10.0])


def _om4_source(*extra_vars: str) -> xr.Dataset:
    required = ["hfds", "so", "tauuo", "tauvo", "thetao", "uo", "vo", "zos"]
    return xr.Dataset(
        {name: ("time", np.ones(2)) for name in [*required, *extra_vars]},
        coords={"time": [0, 1]},
    )


def test_select_om4_variables_retains_declared_snapshot_forcing():
    out = select_om4_variables(_om4_source("wfo", "hfgeou", "wet"))

    assert set(out.data_vars) == {
        "hfds",
        "so",
        "tauuo",
        "tauvo",
        "thetao",
        "uo",
        "vo",
        "wfo",
        "zos",
    }


def test_select_om4_variables_rejects_incomplete_source():
    source = _om4_source().drop_vars("thetao")

    with pytest.raises(ValueError, match="missing required variables.*thetao"):
        select_om4_variables(source)
