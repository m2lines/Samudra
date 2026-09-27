# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any

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


def test_structure_pools_within_field_moments_with_matching_pair_support():
    from samudra.experiments.diffusion_scratch_summary import summarize_structure

    values: list[dict[str, Any]] = []
    for mean, area, pair_area in ((1.0, 1.0, 3.0), (3.0, 3.0, 1.0)):
        values.append(
            dict(
                statistics=dict(
                    example=dict(
                        mean=[[mean, None]],
                        variance=[[mean**2, None]],
                        area=[area, 0.0],
                        zonal_increment_mse=[[mean, None]],
                        zonal_pair_area=[pair_area, 0.0],
                        meridional_increment_mse=[[mean, None]],
                        meridional_pair_area=[area, 0.0],
                    )
                )
            )
        )
    result = summarize_structure(values)["example"]
    assert result["mean"] == [2.5, None]
    assert result["variance"] == [7.0, None]
    assert result["zonal_increment_mse"] == [1.5, None]
    values[0]["statistics"]["example"]["mean"][0][0] = None
    with pytest.raises(ValueError, match="Nonfinite structure"):
        summarize_structure(values)
