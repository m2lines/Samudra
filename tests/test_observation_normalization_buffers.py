# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact dtype, rounding and missing-support behavior of bounded normalization."""

import numpy as np
import pytest

from samudra.observations.model_data import normalize_physical_fields


@pytest.mark.parametrize(
    "mean_dtype,std_dtype",
    [(np.float32, np.float32), (np.float32, np.float64), (np.float64, np.float32)],
)
def test_normalization_matches_original_expression_exactly(mean_dtype, std_dtype):
    rng = np.random.default_rng(439)
    values = rng.normal(size=(26, 77, 3, 5)).astype(np.float32)
    mask = rng.random((77, 3, 5)) > 0.25
    mean = rng.normal(size=77).astype(mean_dtype)
    std = rng.uniform(0.1, 3, size=77).astype(std_dtype)
    values[:, ~mask] = np.nan
    expected = np.where(
        mask, (values - mean[:, None, None]) / std[:, None, None], 0
    ).astype(np.float32)
    actual = normalize_physical_fields(values, mask, mean, std)
    np.testing.assert_array_equal(actual, expected)
    if mean_dtype == std_dtype == np.float32:
        assert actual is values
    values[:, mask] = np.nan
    with pytest.raises(ValueError, match="Nonfinite model target"):
        normalize_physical_fields(values, mask, mean, std)
