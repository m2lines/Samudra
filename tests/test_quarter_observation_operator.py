# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from samudra.experiments.observation_prepare import ConservativeRemap
from samudra.experiments.quarter_observation_operator import NativeIapOperator


def test_native_iap_projection_matches_reference_and_propagates_gradients():
    lat, lon = np.array([-67.5, -22.5, 22.5, 67.5]), np.arange(22.5, 360, 45)
    target_lat, target_lon = np.array([-45, 45]), np.arange(0, 360, 90)
    mask = np.ones((2, 4, 8), dtype=bool)
    mask[1, 0, 0] = False
    operator = NativeIapOperator(lat, lon, target_lat, target_lon, mask)
    prediction = torch.arange(128, dtype=torch.float32).reshape(2, 2, 4, 8)
    prediction.requires_grad_()
    actual = operator(prediction)
    reference = ConservativeRemap(lat, lon, target_lat, target_lon)
    expected, coverage = reference(np.where(mask, prediction.detach().numpy(), np.nan))
    np.testing.assert_allclose(actual.detach(), expected, atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(operator.coverage, coverage[0], atol=1e-6)
    torch.nan_to_num(actual).sum().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert (prediction.grad[:, 1, 0, 0] == 0).all()
    assert (prediction.grad > 0).any()


def test_native_iap_projection_rejects_nonfinite_wet_predictions():
    operator = NativeIapOperator(
        [-45, 45], [90, 270], [-45, 45], [90, 270], np.ones((1, 2, 2), dtype=bool)
    )
    values = torch.ones((1, 1, 2, 2))
    values[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="Nonfinite model prediction"):
        operator(values)


def test_native_iap_projection_preserves_constant_under_autocast():
    operator = NativeIapOperator(
        [-67.5, -22.5, 22.5, 67.5],
        np.arange(22.5, 360, 45),
        [-45, 45],
        np.arange(0, 360, 90),
        np.ones((2, 4, 8), dtype=bool),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = operator(torch.full((3, 2, 4, 8), 7.125))
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.full((3, 2, 2, 4), 7.125))
