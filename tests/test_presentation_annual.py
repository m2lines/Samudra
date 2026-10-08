# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "presentation_annual",
    Path(__file__).parents[1] / "scripts/collect_presentation_annual.py",
)
assert spec is not None and spec.loader is not None
annual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annual)


def test_annual_pooling_preserves_equal_case_squared_error():
    result = annual.pooled([{"sst_rmse": 0.0}, {"sst_rmse": 2.0}])
    assert result["sst_rmse"] == pytest.approx(2**0.5)
    assert result["sst_rmse"] != 1.0  # Averaging case RMSE would understate error.


def test_pooling_cannot_hide_a_failed_forecast():
    with pytest.raises(ValueError, match="Nonfinite"):
        annual.pooled([{"sst_rmse": 1.0}, {"sst_rmse": float("nan")}])


def test_fixed_lead_ohc_compares_calendar_month_not_instantaneous_day():
    record = {
        "leads": {"30": {"sst_rmse": 1}, "365": {"sst_rmse": 2}},
        "monthly_ohc": {
            "2018-01": {"0_700": 3, "700_2000": 4},
            "2018-12": {"0_700": 5, "700_2000": 6},
        },
    }
    short = annual.endpoint(record, "2018-01-01", 30)
    long = annual.endpoint(record, "2018-01-01", 365)
    assert short["ohc_0_700_rmse"] == 3
    assert long["ohc_0_700_rmse"] == 5
    assert long["sst_rmse"] == 2


def test_output_hash_audit_requires_every_result_and_detects_changes(tmp_path):
    from samudra.experiments.annual_artifacts import annual_output_hashes

    complete = {
        "origins": {
            "2018-01-01": {
                "metrics": "2018-01-01.json",
                "arrays": "2018-01-01.npz",
                "persistence": "2018-01-01-persistence.json",
            }
        }
    }
    names = [
        "input.json",
        "COMPLETE.json",
        "2018-01-01.json",
        "2018-01-01.npz",
        "2018-01-01-persistence.json",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"original")
    before = annual_output_hashes(tmp_path, complete)
    assert len(before) == 5
    (tmp_path / "2018-01-01.npz").write_bytes(b"changed")
    assert annual_output_hashes(tmp_path, complete) != before
    (tmp_path / "2018-01-01-persistence.json").unlink()
    with pytest.raises(ValueError, match="Missing or invalid"):
        annual_output_hashes(tmp_path, complete)
