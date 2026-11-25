# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Vector rotation from the tripolar grid's x/y to east/north is a rotation.

Rotations form a group: they preserve speed, compose by adding angles, and undo
with the opposite angle. A sign error or a swapped component breaks one of these.
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from ocean_preprocessing.preprocessing import rotate_vectors

pytestmark = pytest.mark.invariant(status="agreed")

components = arrays(
    np.float64, (3, 4), elements=st.floats(-3.0, 3.0, allow_subnormal=False)
)
angles = arrays(
    np.float64, (3, 4), elements=st.floats(-360.0, 360.0, allow_subnormal=False)
)


def as_da(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(values, dims=("y", "x"))


@given(components, components, angles)
def test_rotation_preserves_speed(u, v, angle):
    east, north = rotate_vectors(as_da(u), as_da(v), as_da(angle))
    np.testing.assert_allclose(np.hypot(east, north), np.hypot(u, v), atol=1e-12)


@given(components, components, angles, angles)
def test_rotations_compose_by_adding_angles(u, v, first, second):
    once_u, once_v = rotate_vectors(as_da(u), as_da(v), as_da(first))
    twice_u, twice_v = rotate_vectors(once_u, once_v, as_da(second))
    direct_u, direct_v = rotate_vectors(as_da(u), as_da(v), as_da(first + second))
    np.testing.assert_allclose(twice_u, direct_u, atol=1e-9)
    np.testing.assert_allclose(twice_v, direct_v, atol=1e-9)


@given(components, components)
def test_quarter_turn_counterclockwise_maps_x_to_north(u, v):
    """The angle is the grid's x axis measured counterclockwise from east.

    If grid x points north (angle 90), a pure x component is a northward current.
    """
    east, north = rotate_vectors(as_da(u), as_da(v), as_da(np.full_like(u, 90.0)))
    np.testing.assert_allclose(east, -v, atol=1e-12)
    np.testing.assert_allclose(north, u, atol=1e-12)
