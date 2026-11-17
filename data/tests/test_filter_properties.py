# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The Gaussian spatial filter is a conservative, variance-reducing average over the ocean.

gcm-filters' REGULAR_WITH_LAND grid treats every cell as the same size, so what
it conserves exactly is the plain sum over wet cells. The pipeline filters on the
native tripolar grid, whose cells shrink towards the poles, so the physical
quantity, the area integral, is conserved only if cell areas vary slowly. The
last test measures that on a lat-lon grid.
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from samudra.invariants import conservation, masks, require_no_violations
from tests.ocean_strategies import ocean_state, regular_grid

pytest.importorskip("gcm_filters")
from ocean_preprocessing.preprocessing import spatially_filter  # noqa: E402

pytestmark = pytest.mark.invariant(status="agreed")


@st.composite
def filter_cases(draw):
    grid = regular_grid(draw(st.integers(8, 16)), draw(st.integers(12, 24)))
    ds = draw(ocean_state(grid, nlev=2, ntime=1))
    return ds, draw(st.integers(2, 4))


def run_filter(ds, scale):
    fields = ds[["thetao", "so", "zos"]]
    return spatially_filter(fields, ds.wetmask, filter_scale=scale)


@given(filter_cases())
def test_filter_keeps_land_nan_and_ocean_finite(case):
    ds, scale = case
    out = run_filter(ds, scale)
    found = []
    for name in ("thetao", "so"):
        found += masks.nan_matches_mask(out[name], ds.wetmask, name)
    found += masks.nan_matches_mask(out.zos, ds.wetmask.isel(lev=0), "zos")
    require_no_violations(found)


@given(filter_cases())
def test_filter_conserves_the_wet_sum_and_reduces_variance(case):
    ds, scale = case
    out = run_filter(ds, scale)
    ones = xr.ones_like(ds.wetmask, dtype=float).where(ds.wetmask)
    require_no_violations(
        conservation.integral_preserved(
            ds.thetao, ones, out.thetao, ones, "thetao", rtol=1e-6
        )
        + conservation.variance_not_increased(
            ds.thetao, ones, out.thetao, ones, "thetao", rtol=1e-6
        )
    )


@given(filter_cases())
def test_surface_fields_filter_like_the_top_level(case):
    """A 2-D field and the same values as level 0 of a 3-D field see the same coastline."""
    ds, scale = case
    as_level = (
        ds.zos.expand_dims(lev=ds.lev).where(ds.wetmask).transpose(*ds.thetao.dims)
    )
    out = run_filter(ds.assign(thetao=as_level), scale)
    np.testing.assert_allclose(
        out.zos.values, out.thetao.isel(lev=0).values, rtol=1e-12, atol=1e-12
    )


@given(filter_cases(), st.floats(-2.0, 30.0))
def test_filter_keeps_a_constant_constant_up_to_the_coast(case, value):
    ds, scale = case
    constant = ds.assign(thetao=xr.full_like(ds.thetao, value).where(ds.wetmask))
    out = run_filter(constant, scale)
    wet = np.broadcast_to(ds.wetmask.values, out.thetao.shape)
    np.testing.assert_allclose(out.thetao.values[wet], value, rtol=1e-6, atol=1e-6)


@pytest.mark.invariant(status="proposed")
@given(filter_cases())
def test_filter_conserves_area_integrals_on_a_sphere(case):
    """Proposed: the heat integral should survive filtering to ~0.1%, as for regridding."""
    ds, scale = case
    out = run_filter(ds, scale)
    lat = np.deg2rad(ds.lat)
    area = np.cos(lat) * ds.wetmask
    require_no_violations(
        conservation.integral_preserved(
            ds.thetao, area, out.thetao, area, "thetao", rtol=1e-3
        )
    )
