# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Paired precision contrasts and zero-error handling."""

import math

import pytest

from .encoding_report import bits, curve_rows, penalty, summarize


def fixture_bundle():
    def history(steps, errors):
        return [
            {
                "step": s,
                "raw": {"rmse": e, "bits": -math.log2(e)},
                "ema": {"rmse": e / 2, "bits": -math.log2(e / 2)},
            }
            for s, e in zip(steps, errors, strict=True)
        ]

    return {
        "result": {
            "metadata": {"entry": {"id": "a", "kind": "mse", "lr": 1e-4, "seed": 11}}
        },
        "phases": {
            "probe4-main": {"curves": history([0, 100, 256], [0.1, 0.05, 0.025])},
            "probe4-noisy": {
                "curves": history([0, 1, 100, 256], [0.1, 0.11, 0.2, 0.4])
            },
        },
    }


def test_exact_error_does_not_invent_finite_precision():
    assert bits(0) == math.inf
    assert penalty(0, 0) is None
    assert penalty(0.1, 0) == math.inf
    assert penalty(0, 0.1) == -math.inf


def test_observed_control_only_and_paired_weight_view():
    rows = curve_rows(fixture_bundle())
    early = next(
        r
        for r in rows
        if r["phase"] == "probe4-noisy" and r["step"] == 1 and r["view"] == "raw"
    )
    assert early["control_bits_lost"] is None
    assert not early["control_observed"]
    assert early["bits_lost_from_start"] == pytest.approx(math.log2(1.1))
    endpoint = next(
        r
        for r in rows
        if r["phase"] == "probe4-noisy" and r["step"] == 256 and r["view"] == "ema"
    )
    assert endpoint["control_bits_lost"] == pytest.approx(4)
    assert endpoint["bits_lost_from_start"] == pytest.approx(2)


def test_no_duplicate_seed_aggregation():
    with pytest.raises(ValueError, match="Repeated seed"):
        summarize([{"seed": 11, "kind": "mse", "bits": 4}] * 2, ["kind"], ["bits"])
