# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from coding import Bins, bits, interval_prob, kernel_interval_prob


def test_interval_probs_sum_to_one():
    b = Bins(-5, 5, 1000)
    idx = np.arange(b.k)
    a, e = b.edges(idx)
    for fam in ("gauss", "laplace"):
        assert abs(interval_prob(fam, 0.3, 0.7, a, e).sum() - 1) < 1e-9
    centers = np.array([[0.1, -0.4, 1.2]] * b.k)
    w = np.full(centers.shape, 1 / 3)
    assert (
        abs(kernel_interval_prob(centers, w, np.full(b.k, 0.2), a, e).sum() - 1) < 1e-9
    )


def test_bits_match_discretized_entropy():
    rng = np.random.default_rng(0)
    sigma, b = 0.5, Bins(-10, 10, 8192)
    y = rng.normal(0, sigma, 200_000)
    a, e = b.edges(b.index(y))
    cost = bits(interval_prob("gauss", 0.0, sigma, a, e), b.k).mean()
    expected = 0.5 * np.log2(2 * np.pi * np.e * sigma**2) - np.log2(b.width)
    assert abs(cost - expected) < 0.01


def test_wrong_model_costs_more():
    rng = np.random.default_rng(1)
    b = Bins(-10, 10, 4096)
    y = rng.normal(0, 1, 50_000)
    a, e = b.edges(b.index(y))
    good = bits(interval_prob("gauss", 0, 1, a, e), b.k).mean()
    bad = bits(interval_prob("gauss", 0.5, 1, a, e), b.k).mean()
    assert bad > good
