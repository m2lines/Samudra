# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.summarize_latent_ar_runs import (
    ablation_key,
    lead_key,
    markdown_table,
    select_best_row,
    summarize_row,
    validate_run_config,
)


def test_validate_run_config_requires_true_physical_leads():
    validate_run_config(
        {
            "config": {
                "target_time_mode": "forecast",
                "train_processor_depths": [1, 2, 4],
                "validation_processor_depths": [1, 2, 4],
            }
        }
    )

    with pytest.raises(ValueError, match="forecast targets"):
        validate_run_config(
            {
                "config": {
                    "target_time_mode": "current",
                    "train_processor_depths": [1, 2, 4],
                    "validation_processor_depths": [1, 2, 4],
                }
            }
        )


def test_select_best_row_uses_true_one_step_loss():
    rows = [
        {"epoch": 1, lead_key(1): 0.3, lead_key(4): 0.4},
        {"epoch": 2, lead_key(1): 0.2, lead_key(4): 0.5},
    ]

    assert select_best_row(rows)["epoch"] == 2


def test_summarize_row_computes_forcing_sensitivity():
    row = {
        "epoch": 3,
        lead_key(1): 0.2,
        lead_key(2): 0.4,
        lead_key(4): 0.8,
        ablation_key("zero", 1): 0.3,
        ablation_key("batch_shuffle", 2): 0.5,
        ablation_key("time_reverse", 4): 1.0,
    }

    summary = summarize_row(row)

    assert summary["zero_lead_1_relative_increase"] == pytest.approx(0.5)
    assert summary["batch_shuffle_lead_2_relative_increase"] == pytest.approx(0.25)
    assert summary["time_reverse_lead_4_relative_increase"] == pytest.approx(0.25)


def test_markdown_table_links_run():
    table = markdown_table(
        [
            {
                "name": "latent-ar",
                "url": "https://example.com",
                "epoch": 4,
                "lead_1": 0.1,
            }
        ]
    )

    assert "[latent-ar](https://example.com)" in table
    assert "| 4 | 0.1 |" in table
