# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from scripts.prepare_extent_variance import VarianceAggregator, overlap


def test_subcell_variance_distinguishes_mean_flow_and_missing_land():
    # Equal-area latitude halves, equal-width longitude halves.
    agg = VarianceAggregator(
        [-90, 0, 90], [0, 180, 360], [-90, 90], [0, 360], np.ones((2, 2), bool)
    )
    u = np.array([[1.0, 3.0], [1.0, 3.0]])
    v = np.array([[2.0, 2.0], [2.0, 2.0]])
    np.testing.assert_allclose(agg.variance(u, v), [[0.5]])
    np.testing.assert_allclose(agg.variance(u + 100, v - 20), [[0.5]])
    np.testing.assert_allclose(
        agg.variance(np.ones((2, 2)) * 3, np.ones((2, 2)) * 7), 0, atol=1e-12
    )
    wet = np.array([[True, False], [True, False]])
    coast = VarianceAggregator([-90, 0, 90], [0, 180, 360], [-90, 90], [0, 360], wet)
    u[:, 1] = np.nan
    np.testing.assert_allclose(coast.fraction, [[0.5]])
    np.testing.assert_allclose(coast.variance(u, v), 0, atol=1e-12)
    u[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        coast.variance(u, v)


def test_non_nested_latitude_bounds_conserve_integrals():
    sy = np.array([-90.0, -55.0, -10.0, 30.0, 90.0])
    ty = np.array([-90.0, -20.0, 90.0])
    sx = np.linspace(0, 360, 9)
    tx = np.linspace(0, 360, 4)
    agg = VarianceAggregator(sy, sx, ty, tx, np.ones((4, 8), bool))
    field = np.arange(32).reshape(4, 8).astype("f8")
    native_area = (
        np.diff(np.sin(np.deg2rad(sy)))[:, None] * np.diff(np.deg2rad(sx))[None, :]
    )
    np.testing.assert_allclose(
        agg.integrate(field).sum(), (field * native_area).sum(), rtol=1e-14
    )
    np.testing.assert_allclose(agg.fraction, 1, rtol=1e-14)
    with pytest.raises(ValueError, match="cover"):
        overlap([1, 2, 3], [0, 3])
