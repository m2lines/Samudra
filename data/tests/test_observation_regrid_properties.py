# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Observation coarsening (`FiniteCoverageRemap`), checked against what conservation implies.

Each target value is the mean of the finite source values over the part of the
target cell they cover, and the coverage is that part's fraction of the full
cell. So the remap conserves the integral of whatever was observed, weights the
sources independently of their values, and doesn't care where the source grid
puts its longitude seam. Unlike `test_regrid_properties.py`, the grids here don't
nest and the holes are arbitrary, as for satellite swaths and Argo gaps.
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from samudra.invariants import conservation, require_no_violations
from tests.ocean_strategies import regular_grid

xesmf = pytest.importorskip("xesmf")
from ocean_preprocessing.regridding import FiniteCoverageRemap  # noqa: E402

pytestmark = pytest.mark.invariant(status="agreed")


def esmf_areas(grid: xr.Dataset) -> xr.DataArray:
    """Cell areas as ESMF computes them, with great-circle edges; conservation is exact in these."""
    area = xesmf.util.cell_area(grid.reset_coords()[["lon", "lat", "lon_b", "lat_b"]])
    return xr.DataArray(np.asarray(area), dims=("y", "x"))


@st.composite
def observed_fields(draw, lon0: float = 0.0):
    """A source grid, an unrelated target grid, and a field with arbitrary gaps.

    Sizes are drawn independently, so source and target edges rarely line up.
    Cells span at most 90 degrees: ESMF can't represent a cell covering a
    hemisphere or the globe as a polygon.
    """
    ny, nx = draw(st.integers(2, 12)), draw(st.integers(2, 12))
    source = regular_grid(ny, 2 * nx, lon0)
    target = regular_grid(draw(st.integers(2, 6)), draw(st.integers(4, 9)))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    values = rng.normal(15.0, 8.0, size=(ny, 2 * nx))
    values[rng.random(values.shape) < draw(st.floats(0.0, 0.9))] = np.nan
    return source, target, values


@given(observed_fields())
def test_remap_conserves_the_integral_of_what_was_observed(case):
    """Sum of mean * coverage * area over targets equals the area integral of the finite sources."""
    source, target, values = case
    mean, coverage = FiniteCoverageRemap(source, target)(values, min_coverage=0.0)
    observed = xr.DataArray(values, dims=("y", "x"))
    remapped = xr.DataArray(mean.astype(np.float64), dims=("y", "x"))
    weights = esmf_areas(target) * xr.DataArray(coverage, dims=("y", "x"))
    assert ((coverage >= 0) & (coverage <= 1 + 1e-6)).all()
    if np.isfinite(values).any():
        require_no_violations(
            conservation.integral_preserved(
                observed, esmf_areas(source), remapped, weights, "observed", rtol=1e-5
            )
        )
    else:
        assert np.isnan(mean).all() and (coverage == 0).all()


@given(observed_fields(), st.floats(-5.0, 5.0).filter(lambda a: abs(a) > 0.1))
def test_remap_weights_do_not_depend_on_the_values(case, scale):
    """Remapping commutes with a change of units and stays inside the observed range.

    The weights depend only on geometry and on which cells are finite, so
    ``remap(a * v + b) == a * remap(v) + b`` (kelvin to Celsius, say), with the
    same coverage and the same missing cells.
    """
    source, target, values = case
    remap = FiniteCoverageRemap(source, target)
    mean, coverage = remap(values)
    shifted, shifted_coverage = remap(scale * values - 273.15)
    np.testing.assert_array_equal(shifted_coverage, coverage)
    np.testing.assert_array_equal(np.isnan(shifted), np.isnan(mean))
    np.testing.assert_allclose(
        shifted, scale * mean - 273.15, rtol=1e-5, atol=1e-3, equal_nan=True
    )
    if np.isfinite(mean).any():
        require_no_violations(
            conservation.no_new_extremes(
                xr.DataArray(values), xr.DataArray(mean), "observed", atol=1e-4
            )
        )


@given(observed_fields())
def test_remap_does_not_depend_on_where_the_source_puts_its_seam(case):
    """OISST runs 0..360 and DUACS -180..180: the same ocean on either convention remaps alike."""
    source, target, values = case
    nx = values.shape[1]
    east = FiniteCoverageRemap(source, target)(values)
    west_source = regular_grid(values.shape[0], nx, -180.0)
    west = FiniteCoverageRemap(west_source, target)(np.roll(values, nx // 2, axis=1))
    np.testing.assert_allclose(west[1], east[1], atol=1e-5)
    np.testing.assert_allclose(west[0], east[0], rtol=1e-5, atol=1e-4, equal_nan=True)
