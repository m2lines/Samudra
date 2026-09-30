# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "global_physical_heat_content",
    Path(__file__).parents[1] / "scripts/global_physical_heat_content.py",
)
assert spec is not None and spec.loader is not None
heat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heat)


def test_global_area_and_constant_heat_total():
    gaussian_lat = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(180)[0]))
    area = heat.cell_areas(gaussian_lat, np.arange(0.5, 360, 1))
    np.testing.assert_allclose(area.sum(), 4 * np.pi * 6371000.0**2, rtol=1e-14)
    support = np.ones(area.shape, dtype=bool)
    values = np.full((2, *area.shape), 1e9)
    np.testing.assert_allclose(
        heat.heat_total(values, area, support), 1e9 * area.sum() / 1e21
    )


def test_missing_cells_are_excluded_without_extrapolation():
    area = np.array([[2e12, 1e12]])
    support = np.array([[True, False]])
    values = np.array([[[1e9, np.nan]], [[2e9, np.nan]]])
    np.testing.assert_allclose(heat.heat_total(values, area, support), [2, 4])
    with pytest.raises(ValueError, match="Nonfinite"):
        heat.heat_total(values, area, np.ones_like(support))
