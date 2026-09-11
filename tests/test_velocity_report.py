# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd
import pytest

from samudra.experiments.velocity_transfer.report import (
    METHODS,
    REGIONS,
    SEEDS,
    write_report,
)


@pytest.fixture
def completed_campaign(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    campaign = {
        "stage": "complete",
        "allocated_gpu_hours_at_last_gate": 1075.0,
        "budget_gpu_hours": 1344,
        "selected_transfer_arm": "D3",
        "code_commit": "synthetic-test-code",
        "runtime_commit": "synthetic-test-runtime",
    }
    (root / "campaign.json").write_text(json.dumps(campaign))
    for split in ("validation", "test"):
        for variant in ("D0", "D3"):
            for seed in SEEDS:
                path = root / f"{split}-confirm-{variant}-s{seed}"
                path.mkdir()
                manifest = {
                    "complete": True,
                    "split": split,
                    "linear_damping": 0.25,
                    "anchors": 4,
                    "config": {
                        "variant": variant,
                        "seed": seed,
                        "gpu_hours": 128,
                        "widths": [64, 96, 128, 192],
                        "data_root": "synthetic-data",
                    },
                }
                (path / "manifest.json").write_text(json.dumps(manifest))
                rows = []
                for month in (1, 4, 7, 10):
                    anchor = pd.Timestamp(year=2021, month=month, day=1)
                    for step in (1, 2, 4, 6):
                        for region in REGIONS:
                            for method in METHODS:
                                squared_error = float(month * 2)
                                if method == "samudra" and variant == "D3":
                                    squared_error *= 0.81
                                rows.append(
                                    {
                                        "anchor": str(anchor),
                                        "target": str(
                                            anchor + pd.Timedelta(days=step * 5)
                                        ),
                                        "lead_step": step,
                                        "region": region,
                                        "method": method,
                                        "weight": float(month),
                                        "weighted_squared_error": squared_error,
                                        "actual_lead_days": step * 5,
                                        "valid_cell_count": 10,
                                        "bias_u_m_per_s": 0.1,
                                        "bias_v_m_per_s": -0.2,
                                    }
                                )
                pd.DataFrame(rows).to_csv(path / "scores.csv", index=False)
    return root


def test_report_pools_physical_errors_and_preserves_seed_bias(
    completed_campaign, tmp_path
):
    output = tmp_path / "report"
    write_report(completed_campaign, output)
    text = (output / "report.md").read_text()
    assert "+10.00%" in text
    assert "1075.00 A100 GPU-hours" in text
    assert "not operational issuance-vintage" in text
    metrics = pd.read_csv(output / "metrics-by-seed.csv")
    selected = metrics[(metrics.variant == "D3") & (metrics.method == "samudra")]
    assert selected.vector_rmse_m_per_s.to_numpy() == pytest.approx(2**0.5 * 0.9)
    assert selected.bias_u_m_per_s.to_numpy() == pytest.approx(0.1)
    assert selected.bias_v_m_per_s.to_numpy() == pytest.approx(-0.2)
    assert set(selected.seed) == set(SEEDS)


@pytest.mark.parametrize(
    "corruption", ["incomplete", "baseline", "missing_date", "budget"]
)
def test_report_rejects_unfinished_or_unmatched_evidence(
    completed_campaign, tmp_path, corruption
):
    root = completed_campaign
    if corruption in ("incomplete", "budget"):
        path = root / "campaign.json"
        manifest = json.loads(path.read_text())
        manifest[
            "stage"
            if corruption == "incomplete"
            else "allocated_gpu_hours_at_last_gate"
        ] = "evaluate" if corruption == "incomplete" else 1400
        path.write_text(json.dumps(manifest))
    else:
        path = root / "test-confirm-D3-s15/scores.csv"
        scores = pd.read_csv(path)
        if corruption == "baseline":
            scores.loc[scores.method == "persistence", "weighted_squared_error"] *= 1.1
        else:
            scores = scores[scores.anchor != scores.anchor.iloc[0]]
        scores.to_csv(path, index=False)
    output = tmp_path / "invalid-report"
    with pytest.raises(ValueError):
        write_report(root, output)
    assert not output.exists()


def test_report_requires_full_audited_date_coverage(completed_campaign, tmp_path):
    audit = {
        "duacs": {
            "splits": {
                "validation": {
                    "windows": 5,
                    "first_anchor": "2021-01-01",
                    "last_anchor": "2021-10-01",
                }
            }
        }
    }
    (completed_campaign / "data-audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="audited date cohort"):
        write_report(completed_campaign, tmp_path / "invalid-report")
