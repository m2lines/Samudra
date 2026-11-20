# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Reference provenance and export scoring tests, independent of a GPU archive."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE = (
    Path(__file__).parents[1] / "experiments/llc-observation-six-hour/export_viewer.py"
)
spec = importlib.util.spec_from_file_location("llc_export", MODULE)
assert spec is not None and spec.loader is not None
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_calendar_context_preserves_dates_and_prioritizes_exact():
    available = pd.date_range("2011-09-13", "2012-11-15T12", freq="6h")
    wanted = pd.to_datetime(["2012-11-15T12", "2012-11-16T00", "2012-12-01T06"])
    actual, exact = export.context_times(available, wanted)
    assert (
        actual.tolist()
        == pd.to_datetime(["2012-11-15T12", "2011-11-16T00", "2011-12-01T06"]).tolist()
    )
    assert exact.tolist() == [True, False, False]
    np.testing.assert_array_equal(actual.month, wanted.month)
    np.testing.assert_array_equal(actual.day, wanted.day)
    np.testing.assert_array_equal(actual.hour, wanted.hour)


def test_one_offset_across_new_year_and_no_fabricated_leap_day():
    available = pd.date_range("2011-09-13", "2012-11-15T12", freq="6h")
    actual, exact = export.context_times(
        available, pd.to_datetime(["2013-12-30T06", "2014-01-10T12"])
    )
    assert (
        actual.tolist() == pd.to_datetime(["2011-12-30T06", "2012-01-10T12"]).tolist()
    )
    assert not exact.any()
    with pytest.raises(ValueError, match="calendar-date-matched"):
        export.context_times(
            pd.date_range("2011-01-01", "2011-12-31", freq="6h"), ["2016-02-29"]
        )


def test_scoring_uses_paired_wet_support_and_area():
    pred = np.array([[[3.0, 10.0, np.nan]]])
    target = np.array([[[1.0, 4.0, np.nan]]])
    area = np.array([[[2.0, 1.0, 0.0]]])
    result = export.score(
        pred, target, np.ones_like(pred, dtype=bool), area, np.array([2.0])
    )
    assert result["mse_physical"] == pytest.approx([44 / 3])
    assert result["mse_normalized"] == pytest.approx([11 / 3])
    assert result["cells"] == [2]
    pred[0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        export.score(
            pred, target, np.ones_like(pred, dtype=bool), area, np.array([2.0])
        )


def test_missing_validation_cannot_pass_publication_gate():
    path = MODULE.with_name("verify_viewer.py")
    spec = importlib.util.spec_from_file_location("llc_verify", path)
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    import copy

    good: dict = {
        str(h): {
            method: dict(
                mse_normalized=[1.0, 2.0], mse_physical=[3.0, 4.0], cells=[5, 6]
            )
            for method in ("model", "persistence")
        }
        for h in (24, 48, 72)
    }
    verifier.check_scores(good, 2)
    bad_cases = [{}, {"24": good["24"]}]
    partial = copy.deepcopy(good)
    del partial["48"]["persistence"]
    bad_cases.append(partial)
    nonfinite = copy.deepcopy(good)
    nonfinite["72"]["model"]["mse_physical"][1] = float("nan")
    bad_cases.append(nonfinite)
    for bad in bad_cases:
        with pytest.raises(ValueError):
            verifier.check_scores(bad, 2)
