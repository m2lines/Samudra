# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "observation_report",
    Path(__file__).parents[1] / "scripts/analyze_observation_predictions.py",
)
assert spec is not None and spec.loader is not None
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_anomaly_correlation_amplitude_and_fixed_support():
    reference = np.array([[[1.0, 2.0], [3.0, np.nan]], [[2.0, 3.0], [4.0, np.nan]]])
    climatology = np.ones_like(reference)
    area = np.array([[1.0, 1.0], [0.5, 0.0]])
    prediction = climatology + 2 * (reference - climatology)
    result = report.anomaly_statistics(prediction, reference, climatology, area)
    assert result["correlation"] == pytest.approx(1)
    assert result["rms_amplitude_ratio"] == pytest.approx(2)
    assert result["pointwise_rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["accepted_values"] == 6
    prediction[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="fixed reference support"):
        report.anomaly_statistics(prediction, reference, climatology, area)


def test_climatology_has_zero_anomaly_amplitude_and_undefined_correlation():
    reference = np.arange(8).reshape(2, 2, 2).astype(float)
    climatology = np.ones_like(reference)
    result = report.anomaly_statistics(
        climatology, reference, climatology, np.ones((2, 2))
    )
    assert result["correlation"] is None
    assert result["rms_amplitude_ratio"] == 0


def test_post_selection_report_pairs_all_years_and_rejects_changed_reference(
    tmp_path, monkeypatch
):
    import json
    import sys

    origins = np.array(
        [f"{year}-{month:02d}" for year in range(2015, 2023) for month in range(1, 13)]
    )
    prediction = np.ones((96, 6, 2, 2, 2))
    reference = np.full((96, 6, 4, 2, 2), 2.0)
    ohc = np.ones((96, 2, 2, 2))
    filenames = [
        "selected-predictions.npz",
        "selected-inferred-persistence.npz",
        "selected-inferred-anomaly-persistence.npz",
        "source-with-zero-forcing.npz",
        "source-inferred-persistence.npz",
        "seasonal-climatology.npz",
        "inferred-anomaly-persistence.npz",
    ]
    for filename in filenames:
        np.savez(
            tmp_path / filename,
            prediction=prediction,
            reference=reference,
            predicted_ohc=ohc,
            reference_ohc=ohc * 2,
            origins=origins,
        )
    (tmp_path / "COMPLETE.json").write_text("{}")
    (tmp_path / "evaluation-input.json").write_text('{"split":"test"}')
    np.savez(
        tmp_path / "grid.npz",
        lat=np.array([-10, 10]),
        lon=np.array([0, 1]),
        mask=np.ones((1, 2, 2), dtype=bool),
    )
    metrics = {key: 1.0 for key in report.PROTOCOL["integrated"]}
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {"control": {"metrics": metrics}, "spectral_keys": ["sst/region/day30"]}
        )
    )
    calls = []

    def score(**kwargs):
        calls.append(len(kwargs["prediction"]))
        return {"metrics": metrics, "spectra": {"sst/region/day30": {"error_dex": 0.2}}}

    monkeypatch.setattr(report, "score", score)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze",
            "--evaluation",
            str(tmp_path),
            "--grid",
            str(tmp_path / "grid.npz"),
            "--selection-reference",
            str(tmp_path / "selection.json"),
            "--output",
            str(tmp_path / "report.json"),
        ],
    )
    report.main()
    result = json.loads((tmp_path / "report.json").read_text())
    assert calls == [12] * 56
    assert result["paired_year_composite_differences"]["source"]["mean"] == 0
    assert len(result["methods"]["selected"]["annual"]) == 8
    reference[0, 0, 0, 0, 0] = np.nan
    np.savez(
        tmp_path / filenames[0],
        prediction=prediction,
        reference=reference,
        predicted_ohc=ohc,
        reference_ohc=ohc * 2,
        origins=origins,
    )
    with pytest.raises(ValueError, match="Different reporting cohort/support"):
        report.main()


def test_heldout_plot_requires_complete_selection_and_matching_cohort():
    from copy import deepcopy

    plot_spec = importlib.util.spec_from_file_location(
        "observation_plot",
        Path(__file__).parents[1] / "scripts/plot_observation_snapshot.py",
    )
    assert plot_spec is not None and plot_spec.loader is not None
    plot = importlib.util.module_from_spec(plot_spec)
    plot_spec.loader.exec_module(plot)
    origins = [
        f"{year}-{month:02d}" for year in range(2015, 2023) for month in range(1, 13)
    ]
    metrics = {
        "origins": origins,
        "spectra": {
            "sst/region/day30": {"k_rad_km": [1.0, 2.0], "reference_power": [1.0, 2.0]}
        },
    }
    frozen = {"spectral_keys": ["sst/region/day30"], "control": metrics}
    snapshot: dict[str, Any] = {
        "arms": {
            "fitting": {"selection-reference": frozen},
            "primary": {
                "TRAIN_COMPLETE": {},
                "best": {"checkpoint_sha256": "selected"},
            },
        },
        "evaluations": {},
    }
    with pytest.raises(ValueError, match="No complete held-out"):
        plot.candidates_for_split(snapshot, "test")
    evaluation: dict[str, Any] = {
        "COMPLETE": {"origins": 96},
        "selected": {
            "sha256": "selected",
            "split": "test",
            "metrics": deepcopy(metrics),
        },
        **{
            key: deepcopy(metrics)
            for key in (
                "seasonal-climatology",
                "source-with-zero-forcing",
                "source-inferred-persistence",
                "inferred-anomaly-persistence",
            )
        },
    }
    snapshot["evaluations"]["primary-evaluation"] = evaluation
    with pytest.raises(ValueError, match="completed selected checkpoint"):
        plot.candidates_for_split(snapshot, "test")
    snapshot["arms"]["primary"]["TRAIN_COMPLETE"] = {"completed_utc": "recorded"}
    _, _, candidates = plot.candidates_for_split(snapshot, "test")
    assert len(candidates) == 5
    snapshot["arms"]["adapter-only"] = deepcopy(snapshot["arms"]["primary"])
    snapshot["evaluations"]["adapter-evaluation"] = deepcopy(evaluation)
    _, _, candidates = plot.candidates_for_split(snapshot, "test")
    assert "Adapter only (selected)" in candidates
    evaluation["selected"]["metrics"]["origins"].pop()
    with pytest.raises(ValueError, match="Incomplete held-out cohort"):
        plot.candidates_for_split(snapshot, "test")
