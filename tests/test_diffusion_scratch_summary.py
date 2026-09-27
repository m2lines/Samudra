# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from samudra.experiments.diffusion_scratch_summary import compare_crps


def records(group, key, score):
    return [
        dict(
            origin=f"{year}-{month:02}",
            statistics={group: dict(weight=[[0.0, 2.0]], **{key: [[0.0, 2 * score]]})},
        )
        for year in range(2015, 2023)
        for month in range(1, 13)
    ]


def test_single_unchanged_baseline_compared_with_both_diffusion_seeds():
    baseline = records("point_mass_interior", "absolute_error", 1.0)
    seeds = {
        "B-1729": records("interior", "fair_crps", 0.5),
        "B-1730": records("interior", "fair_crps", 1.0),
    }
    result = compare_crps(baseline, seeds, draws=20)
    assert result["per_seed_fractional_improvement"] == {"B-1729": 0.5, "B-1730": 0.0}
    assert result["seed_mean_fractional_improvement"] == 0.25
    assert result["paired_year_percentile_95_interval"] == [0.25, 0.25]
    missing = copy.deepcopy(seeds)
    missing["B-1729"].pop()
    with pytest.raises(ValueError, match="complete ordered year"):
        compare_crps(baseline, missing, draws=20)
    seeds["B-1729"][0]["statistics"]["interior"]["weight"] = [[0.0, 3.0]]
    with pytest.raises(AssertionError):
        compare_crps(baseline, seeds, draws=20)
