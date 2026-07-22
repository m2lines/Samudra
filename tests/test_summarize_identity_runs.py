# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import math
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "summarize_identity_runs.py"
    spec = importlib.util.spec_from_file_location("summarize_identity_runs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(epoch: int, mse: float):
    row = {
        "identity/epoch": epoch,
        "identity/mean/mse": mse,
        "identity/grid_height": 180,
        "identity/grid_width": 360,
        "identity/actual_samples": 64,
        "identity/loss/variable/thetao_loss": mse + 0.1,
        "identity/loss/variable/so_loss": mse + 0.2,
        "identity/loss/variable/uo_loss": mse + 0.3,
        "identity/loss/variable/vo_loss": mse + 0.4,
        "identity/loss/variable/zos_loss": mse + 0.5,
    }
    for channel, high_k, seam in (("thetao_0", 0.8, 1.1), ("so_0", 0.6, 0.9)):
        row[f"identity/high_wavenumber_power_ratio/channel/{channel}"] = high_k
        row[f"identity/patch_seam_jump_ratio/channel/{channel}"] = seam
    return row


def _add_route(row, route: str, mse: float, resampler_mse: float):
    prefix = f"identity/route/{route}"
    row[f"{prefix}/mean/mse"] = mse
    for channel in ("thetao_0", "so_0"):
        row[f"{prefix}/deterministic_resampler_normalized_mse/channel/{channel}"] = (
            resampler_mse
        )
        row[f"{prefix}/high_wavenumber_power_ratio/channel/{channel}"] = 0.9
        row[f"{prefix}/patch_seam_jump_ratio/channel/{channel}"] = 1.1


def test_summarize_trajectory_reports_best_and_final_rows():
    script = _load_script()

    summary = script.summarize_trajectory(
        [_row(1, 0.8), _row(2, 0.3), _row(3, 0.4)], name="identity-1deg"
    )

    assert summary["name"] == "identity-1deg"
    assert summary["grid"] == "180x360"
    assert summary["samples"] == 64
    assert summary["best_epoch"] == 2
    assert summary["best_mse"] == pytest.approx(0.3)
    assert summary["final_mse"] == pytest.approx(0.4)
    assert summary["temperature"] == pytest.approx(0.5)
    assert summary["high_wavenumber_ratio"] == pytest.approx(0.7)
    assert summary["patch_seam_ratio"] == pytest.approx(1.0)


def test_markdown_table_contains_identity_evidence():
    script = _load_script()
    summary = script.summarize_trajectory([_row(1, 0.4)], name="identity-1deg")

    table = script.markdown_table([summary])

    assert "identity-1deg" in table
    assert "180x360" in table
    assert "high_wavenumber_ratio" in table


def test_summarize_path_names_direct_json_after_file(tmp_path):
    script = _load_script()
    metrics_path = tmp_path / "identity_1deg_metrics.json"
    metrics_path.write_text(json.dumps([_row(1, 0.4)]))

    summary = script.summarize_path(metrics_path)

    assert summary["name"] == "identity_1deg_metrics"


def test_cross_resolution_routes_compare_model_with_resampler():
    script = _load_script()
    row = _row(1, 0.4)
    _add_route(row, "180x360_to_360x720", 0.3, 0.1)

    summary = script.summarize_trajectory([row], name="cross-resolution")
    route = summary["routes"]["180x360_to_360x720"]

    assert route["model_mse"] == pytest.approx(0.3)
    assert route["resampler_mse"] == pytest.approx(0.1)
    assert route["excess_mse"] == pytest.approx(0.2)
    table = script.route_markdown_table([summary])
    assert "180x360_to_360x720" in table
    assert "resampler_mse" in table


def test_undefined_one_cell_patch_seam_is_reported_as_nan():
    script = _load_script()
    row = _row(1, 0.4)
    for key in list(row):
        if "patch_seam_jump_ratio" in key:
            row[key] = float("nan")

    summary = script.summarize_trajectory([row], name="one-cell-patches")

    assert math.isnan(summary["patch_seam_ratio"])
