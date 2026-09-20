# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from samudra.experiments.surface_adaptation_analysis import (
    paired_year_bootstrap,
    unique_rows,
)


def test_paired_bootstrap_preserves_exact_multiplicative_improvement():
    # Unequal year counts ensure the estimand remains origin-weighted.
    reference = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])
    dates = ["2014-12", "2015-01", "2015-02", "2015-03", "2016-01", "2016-02"]
    result = paired_year_bootstrap(reference * 0.81, reference, dates)
    for name in ("rmse_reduction_pct", "ci95_low_pct", "ci95_high_pct"):
        assert result[name] == pytest.approx(10.0)
    assert result["year_blocks"] == 3
    assert result["origins"] == 6


def test_paired_bootstrap_is_reproducible_and_retains_signed_differences():
    reference = np.arange(1, 10, dtype=float)
    dates = [f"{2000 + i}-01" for i in range(9)]
    result = paired_year_bootstrap(reference * 1.44, reference, dates)
    assert result == paired_year_bootstrap(reference * 1.44, reference, dates)
    assert result["rmse_reduction_pct"] == pytest.approx(-20)


def test_bootstrap_rejects_nonpaired_or_invalid_inputs():
    with pytest.raises(ValueError, match="paired"):
        paired_year_bootstrap([1, 2], [1], ["2000", "2001"])
    with pytest.raises(ValueError, match="two calendar-year"):
        paired_year_bootstrap([1, 2], [1, 2], ["2000", "2000"])
    with pytest.raises(ValueError, match="positive reference"):
        paired_year_bootstrap([1, 2], [0, 0], ["2000", "2001"])


def test_duplicate_origin_identity_is_an_error():
    row = dict(mode="inferred", origin_index="0", origin_time="2014-11-04")
    with pytest.raises(ValueError, match="Duplicate metric key"):
        unique_rows([row, row.copy()], ("mode", "origin_index", "origin_time"))
