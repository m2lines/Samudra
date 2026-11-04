# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Transition semantics, masking, controls and calibration invariants."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import onedeg_transition_benchmark as benchmark


@pytest.fixture
def diagnostics():
    mask = np.ones((77, 5, 6), dtype=bool)
    mask[:, 2, 2:4] = False
    return benchmark.TransitionDiagnostics(
        {
            "mask": mask,
            "mean": np.zeros(77),
            "std": np.ones(77),
            "area": np.ones((5, 6)),
            "lev": np.arange(19),
            "lat": np.linspace(-40, 40, 5),
            "lon": np.arange(6) * 60,
        }
    )


def test_exact_transitions_have_zero_errors_even_with_invalid_land(diagnostics):
    rng = np.random.default_rng(73)
    before = rng.normal(size=(77, 5, 6))
    after = before + rng.normal(size=before.shape)
    after[~diagnostics.mask] = np.nan
    thresholds = {k: np.ones(77) for k in benchmark.REGIONS}
    rows = diagnostics.measure(after, before, after, before, thresholds)
    for row in rows:
        assert row["state_rmse_normalized"] == 0
        assert row["increment_rmse_normalized"] == 0
        assert row["increment_spatial_correlation"] == pytest.approx(1)
        assert row["increment_rms_ratio"] == pytest.approx(1)
        if row["region"] != "interior":
            assert row["increment_gradient_rmse_normalized_per_km"] == 0


def test_persistence_cannot_pass_variability_metric(diagnostics):
    previous = np.zeros((77, 5, 6))
    truth = np.random.default_rng(74).normal(size=previous.shape)
    rows = diagnostics.measure(
        previous, previous, truth, previous, {k: np.ones(77) for k in benchmark.REGIONS}
    )
    for row in rows:
        assert row["increment_rms_ratio"] == 0
        assert row["increment_rms_ratio_distance"] == 1
        assert row["increment_spatial_correlation"] is None
        assert row["increment_rmse_normalized"] > 0


def test_ar_state_offset_is_distinct_from_increment_error(diagnostics):
    previous = np.ones((77, 5, 6))
    truth = previous * 2
    rows = diagnostics.measure(
        truth + 7,
        previous + 7,
        truth,
        previous,
        {k: np.ones(77) for k in benchmark.REGIONS},
    )
    assert all(row["state_rmse_normalized"] == 7 for row in rows)
    assert all(row["increment_rmse_normalized"] == 0 for row in rows)


def test_calibration_excludes_land_and_initial_input_interval(diagnostics):
    window = np.arange(14)[:, None, None, None] * np.ones((14, 77, 5, 6))
    window[0] = -10000
    window[:, ~diagnostics.mask] = 10000
    thresholds = diagnostics.calibrate([window], 0.99)
    for value in thresholds.values():
        np.testing.assert_array_equal(value, 1)


def test_temporal_phases():
    assert [benchmark.phase(i, "autoregressive") for i in range(4)] == [
        "initial_first_lead",
        "within_pair",
        "between_calls",
        "within_pair",
    ]
    assert benchmark.phase(2, "teacher_forced") == "reinitialized_first_lead"


def test_acceptance_requires_dynamics_and_paired_windows():
    protocol = json.loads(
        (
            Path(__file__).parents[1]
            / "experiments/onedeg-weight-edits/transition_protocol.json"
        ).read_text()
    )
    summaries = []
    for split in ("selection", "new_local_test"):
        for mode in protocol["modes"]:
            for window in range(4):
                for candidate in protocol["candidates"]:
                    original = candidate == "original"
                    metrics = {
                        "deep/coast/increment_gradient_rmse_normalized_per_km": 1
                        if original
                        else 0.7,
                        "deep/wet/increment_rmse_normalized": 1 if original else 0.7,
                        "all/wet/state_rmse_normalized": 1,
                        "untargeted/wet/state_rmse_normalized": 1,
                        "per_channel_state_rmse": [1.0] * 77,
                        "deep/wet/increment_spatial_correlation": None
                        if candidate == "deep_persistence"
                        else 0.5,
                        "deep/wet/correlation_valid_fraction": 0
                        if candidate == "deep_persistence"
                        else 1,
                        "deep/wet/increment_rms_ratio_distance": 0.2
                        if original
                        else 0.1,
                    }
                    summaries.append(
                        dict(
                            candidate=candidate,
                            split=split,
                            mode=mode,
                            window=str(window),
                            metrics=metrics,
                        )
                    )
    report = benchmark.acceptance_report(summaries, protocol)
    for row in report:
        assert row["passes_all"] == (row["candidate"] != "deep_persistence")
        assert row["checks"]["deep_increment_rmse_ratio"][
            "paired_window_bootstrap_95"
        ] == pytest.approx([0.7, 0.7])
    with pytest.raises(ValueError, match="Incomplete paired"):
        benchmark.acceptance_report(summaries[:-1], protocol)


@pytest.mark.parametrize("mode", ["teacher_forced", "autoregressive"])
@pytest.mark.parametrize(
    "candidate", ["original", "external_blend", "deep_persistence"]
)
def test_rollout_uses_correct_previous_state_and_restarts(monkeypatch, mode, candidate):
    # Mock only the expensive neural forward; exercise actual rollout bookkeeping.
    shape = (77, 2, 3)
    states = np.arange(14)[:, None, None, None] * np.ones(
        (14, *shape), dtype=np.float32
    )
    boundary = np.zeros((14, 4, 2, 3))

    def predict(model, history, forcing, grid, device):
        return np.stack([history[-1] + 2, history[-1] + 4])

    monkeypatch.setattr(benchmark, "predict", predict)
    transitions = list(
        benchmark.trajectory(None, states, boundary, None, "cpu", candidate, mode)
    )
    assert len(transitions) == 12
    expected_history = states[:2].copy()
    for index, pred, previous in transitions:
        if index % 2 == 0:
            if mode == "teacher_forced":
                expected_history = states[index : index + 2].copy()
            last = expected_history[-1].copy()
            expected_pair = np.stack([last + 2, last + 4])
            alpha = {"original": 0, "external_blend": 0.5, "deep_persistence": 1}[
                candidate
            ]
            expected_pair[:, benchmark.DEEP] = (1 - alpha) * expected_pair[
                :, benchmark.DEEP
            ] + alpha * last[None, benchmark.DEEP]
            np.testing.assert_array_equal(previous, last)
        else:
            np.testing.assert_array_equal(previous, expected_pair[0])
        np.testing.assert_array_equal(pred, expected_pair[index % 2])
        if index % 2:
            expected_history = expected_pair
