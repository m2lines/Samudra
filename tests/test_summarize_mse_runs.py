# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from scripts.summarize_mse_runs import markdown_table, select_best_row


def test_select_best_row_ignores_missing_and_nonfinite_losses():
    rows: list[dict[str, object]] = [
        {"epoch": 0},
        {"epoch": 1, "val/mean/loss": math.nan},
        {"epoch": 2, "val/mean/loss": 0.4},
        {"epoch": 3, "val/mean/loss": 0.3},
    ]

    assert select_best_row(rows)["epoch"] == 3


def test_select_best_row_requires_a_finite_validation_loss():
    with pytest.raises(ValueError, match="No finite"):
        select_best_row([{"val/mean/loss": math.inf}])


def test_markdown_table_links_run_and_formats_metrics():
    table = markdown_table(
        [
            {
                "name": "baseline",
                "url": "https://example.com/run",
                "epoch": 4,
                "all": 0.3512822986,
            }
        ]
    )

    assert "[baseline](https://example.com/run)" in table
    assert "| 4 | 0.351282 |" in table
