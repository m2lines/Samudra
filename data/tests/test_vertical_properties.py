# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Layers tile the water column, and the column stops at the sea floor.

Ocean volume is sum(area x ocean_fraction x dz). It matches the volume under the
bathymetry only if bottom cells count just the water above the floor (#973).
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st
from ocean_preprocessing.preprocessing import (
    account_for_partial_depths,
    flatten_by_depth_level,
)
from ocean_preprocessing.simulation_preprocessing.gfdl_om4 import vertical_cell_metadata

from samudra.invariants import geometry, masks, require_no_violations
from tests.ocean_strategies import OM4_INTERFACES, grid_pairs, ocean_state, regular_grid

pytestmark = pytest.mark.invariant(status="agreed")


@st.composite
def bathymetry_cases(draw, grid=None):
    """A state whose wet mask agrees with a sea-floor depth that cuts through its bottom cells."""
    grid = (
        grid
        if grid is not None
        else regular_grid(draw(st.integers(3, 8)), draw(st.integers(4, 12)))
    )
    nlev = draw(st.integers(2, 6))
    ds = draw(ocean_state(grid, nlev=nlev, ntime=1))
    interfaces = OM4_INTERFACES[: nlev + 1]
    bottom = ds.wetmask.sum("lev").values
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    # The floor lies inside the deepest wet layer: below its top, at or above its base.
    top = interfaces[np.maximum(bottom - 1, 0)]
    base = interfaces[bottom]
    fraction = rng.uniform(0.05, 1.0, size=bottom.shape)
    deptho = np.where(bottom > 0, top + fraction * (base - top), np.nan)
    native = xr.Dataset({"deptho": (("y", "x"), deptho)})
    return ds.assign_coords(ilev=("ilev", interfaces)), native


@given(st.integers(1, 19), st.booleans())
def test_vertical_cell_metadata_layers_tile_the_column(nlev, with_interfaces):
    coords = {"lev": 0.5 * (OM4_INTERFACES[1 : nlev + 1] + OM4_INTERFACES[:nlev])}
    if with_interfaces:
        coords["ilev"] = OM4_INTERFACES[: nlev + 1]
    ds = xr.Dataset(coords=coords)
    dz, ilev = vertical_cell_metadata(ds)
    lev = (
        ds.lev if with_interfaces else 0.5 * (OM4_INTERFACES[1:] + OM4_INTERFACES[:-1])
    )
    require_no_violations(geometry.layer_thickness(dz, ilev, lev))


@given(bathymetry_cases())
def test_partial_depths_clip_columns_to_the_sea_floor(case):
    ds, native = case
    out = account_for_partial_depths(ds, native)
    assert ((out.dz >= 0) & (out.dz <= ds.dz)).all()
    column = (out.dz * ds.wetmask).sum("lev")
    wet = ds.wetmask.any("lev")
    np.testing.assert_allclose(
        column.where(wet).values, native.deptho.where(wet).values, rtol=1e-12, atol=1e-9
    )


def bathymetric_volume(ds, native, area):
    return float((area * native.deptho.fillna(0)).sum())


@given(st.data())
def test_partial_depths_survive_regridding(data):
    """With --account_for_partial_depths, the regridded store's volume is the volume under the floor."""
    xesmf = pytest.importorskip("xesmf")
    from ocean_preprocessing.preprocessing import horizontal_regrid

    fine, coarse, _ = data.draw(grid_pairs())
    ds, native = data.draw(bathymetry_cases(grid=fine))
    partial = account_for_partial_depths(ds, native)
    out = horizontal_regrid(partial, coarse)
    fine_area = xr.DataArray(
        np.asarray(
            xesmf.util.cell_area(
                fine.reset_coords()[["lon", "lat", "lon_b", "lat_b"]], 6356
            )
        )
        * 1e6,
        dims=("y", "x"),
    )
    assert out.dz.dims == ("lev", "y", "x"), (
        f"dz has dims {out.dz.dims} after regridding"
    )
    stored = geometry.ocean_volume(out.areacello, out.ocean_fraction, out.dz)
    np.testing.assert_allclose(
        stored, bathymetric_volume(ds, native, fine_area), rtol=1e-6
    )


@pytest.mark.xfail(
    strict=True,
    reason="#973: without partial depths, bottom cells count their full dz and inflate volume",
)
@given(bathymetry_cases())
def test_default_volume_matches_the_bathymetry(case):
    ds, native = case
    area = xr.ones_like(native.deptho)
    require_no_violations(
        geometry.volume_matches_bathymetry(
            area, ds.wetmask.astype(float), ds.dz, native.deptho, rtol=1e-3
        )
    )


@given(bathymetry_cases())
def test_flattening_splits_levels_without_changing_them(case):
    ds, _ = case
    stacked = ds.drop_vars("ilev")
    out = flatten_by_depth_level(stacked.copy())
    found = masks.flattened_store(
        out.assign_coords(ocean_fraction=stacked.wetmask.astype(float))
    )
    for k in range(stacked.sizes["lev"]):
        for name in ("thetao", "so", "uo", "vo"):
            np.testing.assert_array_equal(
                out[f"{name}_{k}"].values, stacked[name].isel(lev=k).values
            )
        np.testing.assert_array_equal(
            out[f"mask_{k}"].values, stacked.wetmask.isel(lev=k).values
        )
    require_no_violations(found)
