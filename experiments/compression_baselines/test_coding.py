# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from coding import ToleranceBins, interval_prob, kernel_interval_prob


def test_interval_probs_sum_to_one():
    b = ToleranceBins(0.01, 1.0)
    idx = np.arange(-2000, 2000)  # +-20 units: all but ~1e-9 of the mass
    a, e = b.edges(idx)
    for fam in ("gauss", "laplace"):
        assert abs(interval_prob(fam, 0.3, 0.7, a, e).sum() - 1) < 1e-8
    centers = np.array([[0.1, -0.4, 1.2]] * len(idx))
    w = np.full(centers.shape, 1 / 3)
    p = kernel_interval_prob(centers, w, np.full(len(idx), 0.2), a, e)
    assert abs(p.sum() - 1) < 1e-8


def test_bits_match_discretized_entropy():
    rng = np.random.default_rng(0)
    sigma, b = 0.5, ToleranceBins(0.005, 0.5)
    y = rng.normal(0, sigma, 200_000)
    a, e = b.tendency_interval(np.zeros_like(y), y)
    cost = b.bits(interval_prob("gauss", 0.0, sigma, a, e), a, e).mean()
    expected = 0.5 * np.log2(2 * np.pi * np.e * sigma**2) - np.log2(b.width)
    assert abs(cost - expected) < 0.01


def test_wrong_model_costs_more():
    rng = np.random.default_rng(1)
    b = ToleranceBins(0.01, 1.0)
    y = rng.normal(0, 1, 50_000)
    a, e = b.tendency_interval(np.zeros_like(y), y)
    good = b.bits(interval_prob("gauss", 0, 1, a, e), a, e).mean()
    bad = b.bits(interval_prob("gauss", 0.5, 1, a, e), a, e).mean()
    assert bad > good


def test_far_tail_costs_are_finite_and_ordered():
    # 40 scale units out, 1 - cdf cancels to 0 without the survival-function path.
    b = ToleranceBins(1e-3, 1.0)
    a, e = b.tendency_interval(np.zeros(2), np.array([40.0, 45.0]))
    p = interval_prob("laplace", 0.0, 1.0, a, e)
    assert (p > 0).all() and p[0] > p[1]
    assert np.isfinite(b.bits(np.zeros(2), a, e)).all()


def test_non_finite_values_are_rejected():
    with pytest.raises(ValueError):
        ToleranceBins(0.1, 1.0).index(np.array([np.nan]))
