# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.summarize_latent_ar_runs import (
    ZERO_DEPTH_KEY,
    ablation_key,
    high_wavenumber_key,
    lead_key,
    markdown_table,
    persistence_key,
    route_ablation_key,
    route_lead_key,
    route_persistence_key,
    route_spatial_key,
    route_zero_depth_key,
    select_best_row,
    select_terminal_row,
    summarize_row,
    summarize_run,
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


def test_select_terminal_row_uses_latest_validation_step():
    rows = [
        {"epoch": 1, "_step": 10, lead_key(1): 0.3},
        {"epoch": 2, "_step": 20, lead_key(1): 0.2},
        {"epoch": 3, "_step": 30, lead_key(1): 0.4},
    ]

    assert select_terminal_row(rows)["epoch"] == 3


def test_summarize_row_computes_forcing_sensitivity():
    row = {
        "epoch": 3,
        ZERO_DEPTH_KEY: 0.01,
        lead_key(1): 0.2,
        lead_key(2): 0.4,
        lead_key(4): 0.8,
        persistence_key(1): 0.4,
        ablation_key("zero", 1): 0.3,
        ablation_key("batch_shuffle", 2): 0.5,
        ablation_key("time_reverse", 4): 1.0,
    }

    summary = summarize_row(row)

    assert summary["zero_depth_reconstruction"] == 0.01
    assert summary["zero_lead_1_relative_increase"] == pytest.approx(0.5)
    assert summary["lead_1_persistence_reduction"] == pytest.approx(0.5)
    assert summary["batch_shuffle_lead_2_relative_increase"] == pytest.approx(0.25)
    assert summary["time_reverse_lead_4_relative_increase"] == pytest.approx(0.25)


def test_high_wavenumber_key_uses_one_degree_validation_route():
    assert high_wavenumber_key("uo") == (
        "val/resolution/180x360/spatial/high_wavenumber_power_ratio/variable/uo"
    )


def test_summarize_row_includes_exact_route_evidence():
    route = "180x360_to_180x360"
    row = {
        "epoch": 4,
        lead_key(1): 0.2,
        route_zero_depth_key(route): 0.01,
        route_lead_key(route, 1): 0.3,
        route_persistence_key(route, 1): 0.6,
        route_ablation_key(route, "zero", 1): 0.45,
    }

    summary = summarize_row(row, [route])["routes"][route]

    assert summary["zero_depth_reconstruction"] == 0.01
    assert summary["lead_1_persistence_reduction"] == pytest.approx(0.5)
    assert summary["zero_lead_1_relative_increase"] == pytest.approx(0.5)


def test_summarize_run_uses_latest_available_spatial_metrics():
    class FakeRun:
        config = {
            "config": {
                "target_time_mode": "forecast",
                "train_processor_depths": [1, 2, 4],
                "validation_processor_depths": [1, 2, 4],
            }
        }
        path = ["entity", "project", "run"]
        name = "latent-ar"
        state = "finished"
        url = "https://example.com"

        def scan_history(self, *, keys, page_size):
            del page_size
            if high_wavenumber_key("thetao") in keys:
                return iter(
                    [
                        {"epoch": 1, high_wavenumber_key("thetao"): 0.8},
                        {"epoch": 7, high_wavenumber_key("thetao"): 0.9},
                    ]
                )
            return iter([{"epoch": 12, lead_key(1): 0.1}])

    summary = summarize_run(FakeRun())

    assert summary["epoch"] == 12
    assert summary["spatial_metric_epoch"] == 7
    assert summary["high_wavenumber_power_ratio_thetao"] == 0.9


def test_summarize_run_includes_latest_exact_route_spatial_metrics():
    route = "180x360_to_360x720"
    high_k_key = route_spatial_key(route, "high_wavenumber_power_ratio", "uo")
    seam_key = route_spatial_key(route, "patch_seam_jump_ratio", "uo")

    class FakeRun:
        config = {
            "config": {
                "target_time_mode": "forecast",
                "train_processor_depths": [1, 2, 4],
                "validation_processor_depths": [1, 2, 4],
            }
        }
        path = ["entity", "project", "run"]
        name = "latent-ar"
        state = "finished"
        url = "https://example.com"

        def scan_history(self, *, keys, page_size):
            del page_size
            if high_k_key in keys:
                return iter(
                    [
                        {"epoch": 1, high_k_key: 0.2, seam_key: 0.8},
                        {"epoch": 17, high_k_key: 0.4, seam_key: 0.9},
                    ]
                )
            return iter([{"epoch": 17, lead_key(1): 0.1}])

    summary = summarize_run(FakeRun(), [route])

    assert summary["spatial_metric_epoch"] == 17
    assert summary["route_spatial"][route]["epoch"] == 17
    assert summary["route_spatial"][route]["high_wavenumber_power_ratio"]["uo"] == 0.4
    assert summary["route_spatial"][route]["patch_seam_jump_ratio"]["uo"] == 0.9


def test_summarize_run_requests_only_configured_boundary_ablations():
    requested_history_keys = []

    class FakeRun:
        config = {
            "config": {
                "target_time_mode": "forecast",
                "train_processor_depths": [1, 2, 4],
                "validation_processor_depths": [1, 2, 4],
                "validation_boundary_ablations": ["zero"],
            }
        }
        path = ["entity", "project", "run"]
        name = "latent-ar"
        state = "finished"
        url = "https://example.com"

        def scan_history(self, *, keys, page_size):
            del page_size
            if high_wavenumber_key("thetao") in keys:
                return iter([])
            requested_history_keys.extend(keys)
            return iter([{"epoch": 12, lead_key(1): 0.1}])

    summarize_run(FakeRun())

    assert ablation_key("zero", 1) in requested_history_keys
    assert ablation_key("batch_shuffle", 1) not in requested_history_keys
    assert ablation_key("time_reverse", 1) not in requested_history_keys


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
    assert "| 4 | None | 0.1 |" in table
