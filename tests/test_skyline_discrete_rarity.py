# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Accepted coverage replay must preserve both samples and RNG draws."""

import numpy as np
import torch

from samudra.skyline.coverage import sample_coverage
from samudra.skyline.discrete_rarity import ReplaySource, point_batch
from tests.test_skyline_comparison import synthetic_source


def test_point_batch_matches_full_crop_targets_and_weights():
    source, frame = synthetic_source("all_fields")
    depths = np.array([0, 13, 18])
    points = [(0, 0), (0, 0), (0, 0)]
    expected = source.sample(frame, np.random.default_rng(7), 3, 3, depths, points)
    actual = point_batch(source, frame, depths, points)
    for key, value in actual.items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_coverage_replay_preserves_accepted_draws_and_rng():
    source, frame = synthetic_source("all_fields")
    replay = ReplaySource.__new__(ReplaySource)
    replay.__dict__.update(source.__dict__)
    reference_rng = np.random.default_rng(31)
    replay_rng = np.random.default_rng(31)
    config = dict(candidate_pool=128, clusters_per_depth=3, base_fraction=0.5)
    for _ in range(5):
        expected = sample_coverage(source, frame, reference_rng, 16, 3, config)
        actual = sample_coverage(replay, frame, replay_rng, 16, 3, config)
        for key, value in actual.items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        assert reference_rng.bit_generator.state == replay_rng.bit_generator.state
