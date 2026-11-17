# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Conservative horizontal regridding, checked against what conservation implies.

The fine grid's cells nest exactly inside the coarse grid's, so each coarse value
is an ocean-area-weighted mean of the fine values it covers. These are the same
relations the published 1, 1/2 and 1/4 degree stores must satisfy with each other.
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from samudra.invariants import conservation, geometry, masks, require_no_violations
from tests.ocean_strategies import grid_pairs, ocean_state, regular_grid

xesmf = pytest.importorskip("xesmf")
from ocean_preprocessing.preprocessing import horizontal_regrid  # noqa: E402

pytestmark = pytest.mark.invariant(status="agreed")


def fine_areas(grid: xr.Dataset) -> xr.DataArray:
    """Cell areas as ESMF computes them, with great-circle edges, at the target's radius.

    Conservation holds exactly with respect to these. Areas with constant-latitude
    edges differ by up to ~1e-4 on these coarse test grids.
    """
    area = xesmf.util.cell_area(
        grid.reset_coords()[["lon", "lat", "lon_b", "lat_b"]], 6356
    )
    return xr.DataArray(np.asarray(area), dims=("y", "x"))


@st.composite
def regrid_cases(draw, land=True):
    fine, coarse, factor = draw(grid_pairs())
    ds = draw(ocean_state(fine, land=land))
    return ds, coarse, factor


@given(regrid_cases())
def test_regrid_preserves_ocean_area_weighted_means(case):
    ds, coarse, _ = case
    out = horizontal_regrid(ds, coarse)
    fine_weights = fine_areas(ds) * ds.wetmask
    coarse_weights = out.areacello * out.ocean_fraction
    for name in ("thetao", "so", "uo", "zos"):
        require_no_violations(
            conservation.mean_preserved(
                ds[name],
                fine_weights.isel(lev=0)
                if "lev" not in ds[name].dims
                else fine_weights,
                out[name],
                coarse_weights.isel(lev=0)
                if "lev" not in out[name].dims
                else coarse_weights,
                name,
                rtol=1e-6,
                atol=1e-9,
            )
        )


@given(regrid_cases())
def test_regrid_creates_no_new_extremes_or_variance(case):
    ds, coarse, _ = case
    out = horizontal_regrid(ds, coarse)
    fine_weights = fine_areas(ds) * ds.wetmask
    coarse_weights = out.areacello * out.ocean_fraction
    for name in ("thetao", "so"):
        require_no_violations(
            conservation.no_new_extremes(ds[name], out[name], name, atol=1e-9)
            + conservation.variance_not_increased(
                ds[name], fine_weights, out[name], coarse_weights, name, rtol=1e-6
            )
        )


@given(regrid_cases(), st.floats(-2.0, 30.0))
def test_regrid_keeps_a_constant_constant_up_to_the_coast(case, value):
    """Partly wet coarse cells average only their ocean part, so a constant stays exact there too."""
    ds, coarse, _ = case
    constant = ds.assign(thetao=xr.full_like(ds.thetao, value).where(ds.wetmask))
    out = horizontal_regrid(constant, coarse)
    wet = np.broadcast_to(out.wetmask.values, out.thetao.shape)
    np.testing.assert_allclose(out.thetao.values[wet], value, rtol=1e-9, atol=1e-9)


@given(regrid_cases(), st.floats(-3, 3), st.floats(-3, 3))
def test_regrid_is_linear(case, a, b):
    ds, coarse, _ = case
    combined = ds.assign(thetao=a * ds.thetao + b * ds.so)
    left = horizontal_regrid(combined, coarse).thetao
    regridded = horizontal_regrid(ds, coarse)
    right = a * regridded.thetao + b * regridded.so
    np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-6)


@given(regrid_cases())
def test_regrid_commutes_with_a_longitude_shift(case):
    """Shifting the target grid east by one cell equals rolling the output west by one column."""
    ds, coarse, _ = case
    ny, nx = coarse.sizes["y"], coarse.sizes["x"]
    shifted = regular_grid(ny, nx, lon0=360.0 / nx)
    reference = horizontal_regrid(ds, coarse).thetao
    moved = horizontal_regrid(ds, shifted).thetao
    np.testing.assert_allclose(
        moved.values,
        reference.roll(x=-1, roll_coords=False).values,
        rtol=1e-9,
        atol=1e-9,
    )


@given(regrid_cases())
def test_regrid_masks_agree_with_fields_and_fractions(case):
    ds, coarse, _ = case
    out = horizontal_regrid(ds, coarse)
    found = masks.nest_with_depth(out.wetmask) + masks.ocean_fraction_bounds(
        out.ocean_fraction
    )
    found += masks.mask_matches_fraction(out.wetmask, out.ocean_fraction, "wetmask")
    for name in ("thetao", "so", "uo", "vo"):
        found += masks.nan_matches_mask(out[name], out.wetmask, name)
    require_no_violations(found)


@pytest.mark.xfail(
    strict=True,
    reason="#972: horizontal_regrid computes areacello with the polar radius, 6356 km",
)
@given(st.integers(3, 12), st.integers(4, 24))
def test_regrid_target_areas_tile_the_sphere(ny, nx):
    ds_fine = regular_grid(2 * ny, 2 * nx)
    wet = xr.DataArray(np.ones((1, 2 * ny, 2 * nx), bool), dims=("lev", "y", "x"))
    ds = xr.Dataset(
        {"thetao": (("time", "lev", "y", "x"), np.ones((1, 1, 2 * ny, 2 * nx)))},
        coords={"wetmask": wet, "time": [0]},
    ).assign_coords(ds_fine.coords)
    target = regular_grid(ny, nx)
    out = horizontal_regrid(ds, target)
    require_no_violations(geometry.sphere_area(out.areacello, rtol=1e-6))
