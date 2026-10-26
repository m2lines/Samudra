# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact forcing reconstruction for the fixed-example X-ray export."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "xray_export", Path(__file__).parents[1] / "scripts/export_monthly_step_anatomy.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_five_day_forcing_and_timestamp_guard(tmp_path):
    directory = tmp_path / "era5/2015"
    directory.mkdir(parents=True)
    for i in range(1, 6):
        stamp = f"2015-01-{i:02d}"
        np.savez(
            directory / (stamp + ".npz"),
            time=stamp,
            values=np.full((8, 180, 360), i, dtype=np.float32),
        )
    stats = {"atmosphere_mean": np.ones(8), "atmosphere_std": np.full(8, 2)}
    result, proof = module.era5_for_intervals(
        tmp_path, [("2015-01-01", "2015-01-06")], stats
    )
    np.testing.assert_array_equal(result, np.ones((1, 8, 180, 360)))
    assert len(proof["daily_sha256"]) == 5
    with pytest.raises(ValueError, match="five-day"):
        module.era5_for_intervals(tmp_path, [("2015-01-01", "2015-01-05")], stats)
    np.savez(
        directory / "2015-01-01.npz", time="2015-01-02", values=np.zeros((8, 180, 360))
    )
    with pytest.raises(ValueError, match="timestamp"):
        module.era5_for_intervals(tmp_path, [("2015-01-01", "2015-01-06")], stats)
