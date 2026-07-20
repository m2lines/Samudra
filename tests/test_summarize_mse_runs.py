# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from scripts.summarize_mse_runs import (
    markdown_table,
    scan_best_row_and_family,
    select_best_row,
    select_best_row_and_family,
    validate_run_config,
)


def test_validate_run_config_accepts_only_plain_one_step_mse():
    validate_run_config({"config": {"loss": "mse", "steps": [1]}})

    with pytest.raises(ValueError, match="plain MSE"):
        validate_run_config(
            {
                "config": {
                    "loss": {"type": "dynamic", "metric": "mse"},
                    "steps": [1],
                }
            }
        )
    with pytest.raises(ValueError, match="one-step"):
        validate_run_config({"config": {"loss": "mse", "steps": [4]}})


def test_select_best_row_ignores_missing_and_nonfinite_losses():
    rows: list[dict[str, object]] = [
        {"epoch": 0},
        {"epoch": 1, "val/mean/loss": math.nan},
        {"epoch": 2, "val/mean/loss": 0.4},
        {"epoch": 3, "val/mean/loss": 0.3},
    ]

    assert select_best_row(rows)["epoch"] == 3


def test_select_best_row_requires_a_finite_validation_loss():
    with pytest.raises(ValueError, match="No finite validation loss"):
        select_best_row([{"val/mean/loss": math.inf}])


def test_select_best_row_prefers_explicit_unweighted_metric_family():
    rows = [
        {
            "epoch": 1,
            "val/mean/loss": 0.4,
            "val/resolution/180x360/unweighted_normalized_mse/mean/loss": 0.2,
        },
        {
            "epoch": 2,
            "val/mean/loss": 0.3,
            "val/resolution/180x360/unweighted_normalized_mse/mean/loss": 0.25,
        },
    ]

    row, family = select_best_row_and_family(rows)

    assert row["epoch"] == 1
    assert family.name == "unweighted_1deg"


def test_scan_best_row_checks_mutually_exclusive_metric_families_separately():
    class FakeRun:
        def __init__(self):
            self.requested_keys: list[list[str]] = []

        def scan_history(self, *, keys, page_size):
            self.requested_keys.append(keys)
            assert page_size == 1000
            if "val/resolution/180x360/unweighted_normalized_mse/mean/loss" in keys:
                return [
                    {
                        "epoch": 2,
                        "val/resolution/180x360/unweighted_normalized_mse/mean/loss": 0.2,
                    }
                ]
            return []

    run = FakeRun()
    row, family = scan_best_row_and_family(run)

    assert row["epoch"] == 2
    assert family.name == "unweighted_1deg"
    assert len(run.requested_keys) == 2


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
