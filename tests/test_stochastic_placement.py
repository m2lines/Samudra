# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Scientific checks for masked spatial placement diagnostics."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "placement", Path(__file__).parents[1] / "scripts/diagnose_stochastic_placement.py"
)
assert spec is not None and spec.loader is not None
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)


def test_displaced_feature_keeps_amplitude_and_improves_with_neighborhood():
    truth = np.zeros((21, 40))
    truth[10, 10] = 1
    prediction = np.roll(truth, 2, axis=1)
    mask = np.ones(truth.shape, dtype=bool)
    lat = np.linspace(-10, 10, 21)
    weights = np.cos(np.deg2rad(lat))[:, None] * mask
    assert placement.weighted_mean(truth**2, weights) == placement.weighted_mean(
        prediction**2, weights
    )
    scores = []
    for radius in (0, 1, 2, 4, 8):
        ref = placement.neighborhood_fraction(truth, mask, lat, radius)
        pred = placement.neighborhood_fraction(prediction, mask, lat, radius)
        scores.append(placement.fractions_score(pred, ref, weights)["fss"])
    assert scores[0] == 0
    assert np.all(np.diff(scores) > 0)
    assert placement.fractions_score(np.zeros_like(truth), truth, weights)["fss"] == 0


def test_land_is_excluded_and_longitude_wraps_without_latitude_wrap():
    mask = np.ones((7, 8), dtype=bool)
    mask[:, 3] = False
    values = mask.astype(float)
    values[~mask] = 999
    fractions = placement.neighborhood_fraction(values, mask, np.zeros(7), 1)
    np.testing.assert_allclose(fractions, 1)
    impulse = np.zeros((7, 8))
    impulse[0, 0] = 1
    fractions = placement.neighborhood_fraction(
        impulse, np.ones_like(mask), np.zeros(7), 1
    )
    assert fractions[0, -1] > 0
    assert fractions[-1, 0] == 0


def test_spectrum_matches_parseval_and_amplitude_scaling():
    rng = np.random.default_rng(2026)
    u, v = rng.normal(size=(2, 20, 40))
    mask = np.ones(u.shape, dtype=bool)
    mask[:3, :4] = False
    _, power, _ = placement.regional_spectrum(u, v, mask)
    _, scaled, _ = placement.regional_spectrum(2 * u, 2 * v, mask)
    np.testing.assert_allclose(scaled, 4 * power)
    taper = np.hanning(20)[:, None] * np.hanning(40)[None, :] * mask
    expected = sum(
        np.sum((np.where(mask, x - x[mask].mean(), 0) * taper) ** 2) for x in (u, v)
    ) / np.sum(taper**2)
    np.testing.assert_allclose(power.sum(), expected)


def test_no_event_fss_is_undefined():
    zeros = np.zeros((2, 2))
    assert placement.fractions_score(zeros, zeros, np.ones_like(zeros))["fss"] is None


def test_case_metrics_use_paired_support_and_distinguish_member_from_mean():
    gold = np.zeros((2, 7, 8))
    gold[0] = 1
    members = np.stack([gold, -gold])
    masks = np.ones_like(gold, dtype=bool)
    masks[:, 0, 0] = False
    gold[:, 0, 0] = np.nan
    members[:, :, 0, 0] = np.nan
    amplitude, _, _ = placement.analyze_case(
        members,
        gold,
        masks,
        ["uo_0", "vo_0"],
        np.linspace(-55, -50, 7),
        np.linspace(0, 7, 8),
        dict(variant="test", month="2000-01", step=1),
    )
    values = {row["sample"]: row for row in amplitude if row["region"] == "global"}
    assert values["member-1"]["vector_power"] == 1
    assert values["member-2"]["vector_power"] == 1
    assert values["ensemble-mean"]["vector_power"] == 0
    assert values["reference"]["wet_cells"] == 55
