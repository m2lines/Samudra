# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import xarray as xr
from ocean_preprocessing.simulation_preprocessing.gfdl_om4 import (
    normalize_vertical_coords,
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
