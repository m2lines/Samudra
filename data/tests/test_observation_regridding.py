# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Observation support policy tested against actual ESMF conservative weights."""

import numpy as np
import pytest
from ocean_preprocessing.regridding import (
    FiniteCoverageRemap,
    global_cell_bounds,
    rectilinear_global_grid,
)

from tests import requires_xesmf


def test_global_edges_include_poles_and_periodic_seam():
    np.testing.assert_allclose(
        global_cell_bounds([-90, -30, 30, 90], latitude=True), [-90, -60, 0, 60, 90]
    )
    np.testing.assert_allclose(
        global_cell_bounds([45, 135, 225, 315]), [0, 90, 180, 270, 360]
    )
    with pytest.raises(ValueError, match="global longitude"):
        global_cell_bounds([0, 1, 2])


@requires_xesmf
def test_esmf_identity_preserves_values_and_missingness():
    lat, lon = [-45, 45], [45, 135, 225, 315]
    remap = FiniteCoverageRemap(lat, lon, lat, lon)
    raw = np.array([[1, np.nan, 3, 4], [5, 6, 7, 8]], dtype=float)
    actual, coverage = remap(raw)
    np.testing.assert_allclose(actual, raw, equal_nan=True, atol=1e-6)
    np.testing.assert_allclose(coverage, np.isfinite(raw).astype(float), atol=1e-6)


@requires_xesmf
def test_esmf_coverage_uses_full_cell_area_not_finite_renormalization():
    source_lon = np.arange(22.5, 360, 45)
    target_lon = np.arange(45, 360, 90)
    remap = FiniteCoverageRemap([-45, 45], source_lon, [-45, 45], target_lon)
    values = np.full((2, 2, 8), 7.0)
    values[0, 0, 0] = np.nan
    values[1, 0, :2] = np.nan
    actual, coverage = remap(values)
    assert coverage[0, 0, 0] == pytest.approx(0.5, abs=1e-6)
    assert coverage[1, 0, 0] == pytest.approx(0, abs=1e-6)
    assert np.isnan(actual[:, 0, 0]).all()
    np.testing.assert_allclose(actual[:, 1], 7)
    accepted, _ = remap(values, min_coverage=0.4)
    assert accepted[0, 0, 0] == pytest.approx(7)
    assert np.isnan(accepted[1, 0, 0])


@requires_xesmf
def test_esmf_poles_and_shifted_seam_keep_complete_global_support():
    source_lon = np.arange(22.5, 360, 45)
    target_lon = np.arange(0, 360, 90)
    remap = FiniteCoverageRemap(
        [-90, -30, 30, 90], source_lon, [-60, 0, 60], target_lon
    )
    actual, coverage = remap(np.full((4, 8), 7.0))
    np.testing.assert_allclose(actual, 7.0, atol=1e-6)
    np.testing.assert_allclose(coverage, 1.0, atol=1e-6)


@requires_xesmf
def test_esmf_full_support_preserves_spherical_integral():
    import xesmf as xe

    source_lat, source_lon = [-60, 0, 60], np.arange(22.5, 360, 45)
    target_lat, target_lon = [-45, 45], np.arange(45, 360, 90)
    raw = np.arange(24, dtype=float).reshape(3, 8)
    remap = FiniteCoverageRemap(source_lat, source_lon, target_lat, target_lon)
    actual, coverage = remap(raw)
    source_area = xe.util.cell_area(
        rectilinear_global_grid(source_lat, source_lon)
    ).values
    target_area = xe.util.cell_area(
        rectilinear_global_grid(target_lat, target_lon)
    ).values
    np.testing.assert_allclose(coverage, 1, atol=1e-6)
    np.testing.assert_allclose(
        (actual * target_area).sum(), (raw * source_area).sum(), rtol=1e-6
    )
