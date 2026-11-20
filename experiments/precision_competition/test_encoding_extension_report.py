# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched-seed comparisons exclude missing qualified branches."""

import pytest

from .encoding_extension_report import paired_rows


def test_matched_seed_contrast_and_missing_gate():
    common = {"lr": 1e-4, "phase": "noisy", "view": "ema"}
    rows = [
        {
            **common,
            "kind": "cont_shrink_normalized",
            "seed": 11,
            "rmse": 0.1,
            "control_bits_lost": 0.5,
        },
        {
            **common,
            "kind": "cont_shrink_mse",
            "seed": 11,
            "rmse": 0.2,
            "control_bits_lost": 2.0,
        },
        {
            **common,
            "kind": "cont_shrink_mse",
            "seed": 22,
            "rmse": 0.3,
            "control_bits_lost": 3.0,
        },
    ]
    result = paired_rows(rows)
    assert len(result) == 1
    assert result[0]["seed"] == 11
    assert result[0]["copy_bits_gain"] == pytest.approx(1.0)
    assert result[0]["interference_penalty_reduction"] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_rows(rows + [rows[0]])
