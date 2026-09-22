# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

import pandas as pd
import pytest

INTERVALS = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/compare_initializer_wave.py")
)["paired_intervals"]


def fixture():
    return pd.DataFrame(
        {
            "origin_time": ["2014-12-01", "2015-01-01", "2015-02-01", "2016-01-01"],
            "normalized_mse": [1.0, 4.0, 9.0, 16.0],
        }
    )


def test_identical_pairs_have_exact_zero_interval():
    frame = fixture()
    result = INTERVALS(frame, frame, draws=100)
    assert result["difference_ci_low"] == result["difference_ci_high"] == 0
    assert result["rmse_reduction_percent"] == 0
    assert result["reference_rmse"] == pytest.approx((30 / 4) ** 0.5)


def test_proportional_skill_retains_paired_percent_gain():
    reference = fixture()
    candidate = reference.copy()
    candidate["normalized_mse"] *= 0.25
    result = INTERVALS(reference, candidate, draws=100)
    assert result["reduction_ci_low"] == result["reduction_ci_high"] == 50
    assert result["difference_ci_high"] < 0


def test_mismatched_dates_rejected():
    reference = fixture()
    candidate = reference.copy()
    candidate.loc[0, "origin_time"] = "2014-11-01"
    with pytest.raises(ValueError, match="matched origins"):
        INTERVALS(reference, candidate)
