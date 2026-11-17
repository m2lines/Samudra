# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""MOM6's staggered grid: face velocities average onto cell centres, and the supergrid splits cleanly.

MOM6 stores u on the east face of each cell (xq) and v on the north face (yq).
Centring averages the two faces around a cell, wrapping in longitude.
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

pytest.importorskip("xgcm")
from ocean_preprocessing.simulation_preprocessing.gfdl_om4 import (
    convert_super_grid,  # noqa: E402
)
from ocean_preprocessing.simulation_preprocessing.interpolate import (  # noqa: E402
    interpolate_to_cell_centers,
)
from xgcm import Grid  # noqa: E402

from tests.ocean_strategies import regular_grid  # noqa: E402

pytestmark = pytest.mark.invariant(status="agreed")


@st.composite
def staggered(draw):
    ny, nx = draw(st.integers(2, 6)), draw(st.integers(3, 8))
    values = st.floats(-2.0, 2.0, allow_subnormal=False)
    u = draw(arrays(np.float64, (ny, nx), elements=values))
    v = draw(arrays(np.float64, (ny, nx), elements=values))
    tracer = draw(arrays(np.float64, (ny, nx), elements=values))
    ds = xr.Dataset(
        {
            "thetao": (("yh", "xh"), tracer),
            "uo": (("yh", "xq"), u),
            "vo": (("yq", "xh"), v),
        },
        coords={
            "xh": np.arange(nx) + 0.5,
            "xq": np.arange(nx) + 1.0,
            "yh": np.arange(ny) + 0.5,
            "yq": np.arange(ny) + 1.0,
        },
    )
    grid = Grid(
        ds,
        coords={
            "X": {"center": "xh", "right": "xq"},
            "Y": {"center": "yh", "right": "yq"},
        },
        boundary={"X": "periodic", "Y": "extend"},
        periodic=["X"],
        autoparse_metadata=False,
    )
    return ds, grid


@given(staggered())
def test_centring_averages_the_faces_around_each_cell(case):
    ds, grid = case
    out = interpolate_to_cell_centers(ds, ds.thetao, grid)
    u, v = ds.uo.values, ds.vo.values
    # East face of the cell to the west wraps around in longitude.
    expected_u = 0.5 * (u + np.roll(u, 1, axis=1))
    # The southern row's missing south face extends from its north face.
    south = np.vstack([v[:1], v[:-1]])
    expected_v = 0.5 * (v + south)
    np.testing.assert_allclose(out.uo.values, expected_u, atol=1e-12)
    np.testing.assert_allclose(out.vo.values, expected_v, atol=1e-12)
    np.testing.assert_array_equal(out.thetao.values, ds.thetao.values)


@given(st.integers(2, 10), st.integers(3, 16), st.floats(-180.0, 180.0))
def test_supergrid_splits_into_centres_and_corners(ny, nx, lon0):
    """Build a supergrid from a known grid; converting it must return that grid."""
    grid = regular_grid(ny, nx, lon0=lon0)
    lat_fine = np.linspace(-90.0, 90.0, 2 * ny + 1)
    lon_fine = lon0 + np.linspace(0.0, 360.0, 2 * nx + 1)
    x, y = np.meshgrid(lon_fine, lat_fine)
    angle = np.random.default_rng(ny * 31 + nx).uniform(-180, 180, size=x.shape)
    supergrid = xr.Dataset(
        {
            "x": (("nyp", "nxp"), x),
            "y": (("nyp", "nxp"), y),
            "angle_dx": (("nyp", "nxp"), angle),
        }
    )
    angle_h, lon_h, lat_h, lon_b, lat_b = convert_super_grid(supergrid)
    np.testing.assert_allclose(lon_h.values, grid.lon.values)
    np.testing.assert_allclose(lat_h.values, grid.lat.values)
    np.testing.assert_allclose(lon_b.values, grid.lon_b.values)
    np.testing.assert_allclose(lat_b.values, grid.lat_b.values)
    np.testing.assert_array_equal(angle_h.values, angle[1::2, 1::2])
