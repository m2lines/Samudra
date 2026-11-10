# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Protect paired scientific contrasts and the finite-displacement identity."""

import copy

import numpy as np
import pytest

from .noise_report import probe_decomposition, summarize, trajectory_rows


def test_finite_displacement_decomposition_and_corruption():
    displacement = np.arange(16 * 64).reshape(16, 64) / 10000
    error = np.linspace(-0.1, 0.2, 64)
    change = np.mean((error + displacement) ** 2 - error**2)
    alignment = np.mean(2 * error * displacement.mean(0))
    record = dict(
        displacements=displacement.tolist(),
        mean_mse_change=change,
        error_alignment_term=alignment,
    )
    result = probe_decomposition(record)
    assert result["displacement_second_moment"] == pytest.approx(
        np.mean(displacement**2)
    )
    assert result["corrected_squared_mean"] < result["mean_displacement_squared"]
    with pytest.raises(AssertionError):
        probe_decomposition({**record, "mean_mse_change": change + 0.01})


def fixture_bundle():
    def history(raw, ema):
        return [
            dict(
                step=s,
                examples=s * 256,
                raw={"bits": r},
                ema={"0.99": {"bits": e}, "0.999": {"bits": e - 1}},
            )
            for s, r, e in zip([0, 256, 512], raw, ema, strict=True)
        ]

    return dict(
        entry=dict(id="seed11-bits-6-setting4", stage="perturb", lr=0.0003, batch=256),
        trajectories=[
            dict(condition="main", sigma=0.0, history=history([6, 7, 8], [6, 6.5, 7])),
            dict(
                condition="learned", sigma=1.0, history=history([6, 4, 6], [6, 5, 6.5])
            ),
        ],
    )


def test_penalty_uses_paired_control_and_matching_weight_view():
    rows = trajectory_rows(fixture_bundle())
    intervention = [r for r in rows if r["condition"] == "learned"]
    endpoint = next(r for r in intervention if r["view"] == "raw" and r["step"] == 256)
    assert endpoint["bits_lost_from_start"] == 2
    assert endpoint["control_bits_lost"] == 3
    recovery = next(r for r in intervention if r["view"] == "0.99" and r["step"] == 512)
    assert recovery["control_bits_lost"] == 0.5
    assert recovery["bits_lost_from_start"] == -0.5
    assert recovery["phase"] == "recovery" and recovery["phase_step"] == 256
    assert recovery["examples"] == 131072 and recovery["phase_examples"] == 65536


def test_unmatched_checkpoint_is_not_silently_compared():
    bundle = copy.deepcopy(fixture_bundle())
    bundle["trajectories"][0]["history"].pop(1)
    with pytest.raises(KeyError):
        trajectory_rows(bundle)


def test_duplicate_seed_is_not_counted_as_independent_replicate():
    with pytest.raises(ValueError, match="Repeated seed"):
        summarize(
            [
                {"seed": 11, "group": "a", "value": 1},
                {"seed": 11, "group": "a", "value": 2},
            ],
            ["group"],
            ["value"],
        )
