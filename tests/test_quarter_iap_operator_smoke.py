# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch


def script():
    path = Path(__file__).parents[1] / "scripts" / "quarter_iap_operator_smoke.py"
    spec = importlib.util.spec_from_file_location("quarter_iap_operator_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_physical_mean_and_adjoint():
    lat, lon = np.array([-60, -20, 20, 60]), np.array([45, 135, 225, 315])
    target_lat, target_lon = np.array([-45, 45]), np.array([0, 180])
    mask = np.ones((2, 4, 4), bool)
    mask[1, :2] = False
    values = np.broadcast_to(
        np.arange(4, dtype=float)[None, :, None], mask.shape
    ).copy()
    values[~mask] = np.nan
    reference, coverage, gradient, loss = script().cpu_reference(
        lat, lon, target_lat, target_lon, mask, values
    )
    # South latitudes split at -40 degrees; source rows are values 0 and 1.
    south = np.sin(np.deg2rad(40))
    np.testing.assert_allclose(reference[0, 0], south)
    np.testing.assert_allclose(reference[:, 1], 3 - south)
    assert np.isnan(reference[1, 0]).all()
    np.testing.assert_array_equal(coverage[1, 0], 0)
    np.testing.assert_allclose(gradient.sum(), 1)
    assert (gradient[~mask] == 0).all()
    assert np.isfinite(loss)
    for index in [(0, 0, 0), (0, 2, 3), (1, 3, 1)]:
        changed = values.copy()
        changed[index] += 0.01
        changed_loss = script().cpu_reference(
            lat, lon, target_lat, target_lon, mask, changed
        )[-1]
        np.testing.assert_allclose(
            (changed_loss - loss) / 0.01, gradient[index], rtol=1e-10
        )


@pytest.mark.cuda
def test_native_iap_real_geometry_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    grid, iap = os.environ.get("QUARTER_IAP_GRID"), os.environ.get("QUARTER_IAP_RAW")
    if not grid or not iap:
        pytest.skip("Set QUARTER_IAP_GRID and QUARTER_IAP_RAW for real geometry")
    module = script()
    result = module.diagnose(*module.load_geometry(Path(grid), Path(iap), 28))
    assert result["status"] == "passed"
    assert all(
        policy["bitwise_gradient"]
        for variant in result["variants"].values()
        for policy in variant["policies"]
    )


@pytest.mark.parametrize("operator_name", ["NativeIapOperator", "DiagnosticDenseIap"])
def test_both_implementations_against_independent_reference(operator_name):
    module = script()
    lat, lon = np.array([-60, -20, 20, 60]), np.array([45, 135, 225, 315])
    target_lat, target_lon = np.array([-45, 45]), np.array([0, 180])
    mask = np.ones((2, 4, 4), bool)
    mask[1, :2] = False
    field = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    field[~mask] = np.nan
    expected, _, adjoint, expected_loss = module.cpu_reference(
        lat, lon, target_lat, target_lon, mask, field
    )
    operator = getattr(module, operator_name)(lat, lon, target_lat, target_lon, mask)
    physical = torch.tensor(field, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = operator(physical)
        weights = operator.supported * operator.area
        loss = (
            torch.where(operator.supported, result, 0) * weights
        ).sum() / weights.sum()
    loss.backward()
    assert physical.grad is not None
    assert result.dtype == torch.float32
    np.testing.assert_allclose(result.detach(), expected, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(physical.grad, adjoint, rtol=1e-6)
    np.testing.assert_allclose(loss.item(), expected_loss, rtol=1e-6)
