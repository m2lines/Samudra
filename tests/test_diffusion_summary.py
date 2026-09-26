# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest

from samudra.experiments.diffusion_summary import paired_year_bootstrap, pool


def test_pool_handles_variable_lead_counts_without_mixing_rank_bins():
    records = []
    for leads in (6, 7):
        weight = np.ones((1, leads, 2))
        records.append(
            dict(
                statistics=dict(
                    surface=dict(
                        weight=weight.tolist(),
                        rank_weights=(np.ones((9, 1, leads, 2)) / 9).tolist(),
                    )
                )
            )
        )
    np.testing.assert_allclose(
        pool(records, "surface", "rank_weights").sum(0),
        pool(records, "surface", "weight"),
    )
    np.testing.assert_array_equal(pool(records, "surface", "weight"), [13, 13])


def test_paired_year_bootstrap_known_improvement_and_support_rejection():
    records: dict[str, Any] = {}
    for name in ("A-1729", "A-1730", "B-1729", "B-1730"):
        group, key, score = (
            ("point_mass_interior", "absolute_error", 1.0)
            if name.startswith("A")
            else ("interior", "fair_crps", 0.75)
        )
        records[name] = [
            dict(
                origin=f"{year}-{month:02d}",
                statistics={
                    group: dict(weight=[[0.0, 2.0]], **{key: [[0.0, score * 2]]})
                },
            )
            for year in range(2015, 2023)
            for month in range(1, 13)
        ]
    result = paired_year_bootstrap(records, draws=20)
    assert result["fractional_interior_crps_improvement"] == 0.25
    assert result["percentile_95_interval"] == [0.25, 0.25]
    records["B-1729"][0]["statistics"]["interior"]["weight"] = [[0.0, 3.0]]
    with pytest.raises(AssertionError):
        paired_year_bootstrap(records, draws=20)
