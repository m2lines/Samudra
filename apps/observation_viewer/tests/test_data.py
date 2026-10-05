# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Protect physical coordinates, calendar labels, and missing-data diagnostics."""

import numpy as np
import pytest

from data import edges, interval, limits, paired_stats


def test_latitude_geometry_keeps_irregular_centers_and_poles():
    np.testing.assert_allclose(
        edges([-88, -30, 10, 87], (-90, 90)), [-90, -59, -10, 48.5, 90]
    )
    with pytest.raises(ValueError, match="increasing"):
        edges([20, 10])


def test_only_paired_cells_contribute_to_weighted_error():
    prediction = np.array([[2.0, np.nan], [4.0, 9000.0]])
    reference = np.array([[0.0, 0.0], [0.0, np.nan]])
    result = paired_stats(prediction, reference, [0, 60])
    assert result["cells"] == 2
    assert result["bias"] == pytest.approx(8 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(8))


def test_empty_support_reports_unavailable():
    assert paired_stats(np.full((2, 2), np.nan), np.ones((2, 2)), [0, 30]) == {
        "rmse": None,
        "bias": None,
        "cells": 0,
    }


def test_intervals_are_five_days_and_respect_leap_years():
    assert interval("2015-01-01", 5) == "Day 30 · 2015-01-26–2015-01-30"
    assert interval("2021-01-01", 72) == "Day 365 · 2021-12-27–2021-12-31"
    assert interval("2020-01-01", 72) == "Day 365 · 2020-12-26–2020-12-30"


def test_difference_limits_are_symmetric_and_handle_empty_constant_fields():
    low, high = limits(np.array([-8, 2, np.nan]), symmetric=True)
    assert low == -high and high > 2
    assert limits(np.array([np.nan])) == (0, 1)
    assert limits(np.array([5, 5])) == (4.5, 5.5)


def test_same_date_sources_route_to_distinct_predictions_and_gold(tmp_path):
    import json

    from data import Catalog

    arrays = {"obs.npy": 2, "om4.npy": 7, "gold.npy": 5, "iap.npy": 3}
    files = {}
    for name, value in arrays.items():
        data = np.full((2, 4, 2, 2), value, dtype=np.float32)
        np.save(tmp_path / name, data)
        files[name] = {"shape": list(data.shape), "dtype": str(data.dtype)}
    meta = {
        "schema_version": 1,
        "lat": [-30, 30],
        "lon": [90, 270],
        "depths": [2.5, 10],
        "files": files,
        "interior_examples": {
            "2015-01-01": {
                "models": {"mixed": "obs.npy"},
                "references": {"observations": "iap.npy", "om4": "gold.npy"},
            },
            "2015-01-01-om4": {
                "models": {"mixed": "om4.npy"},
                "references": {"om4": "gold.npy"},
            },
        },
    }
    (tmp_path / "catalog.json").write_text(json.dumps(meta))
    catalog = Catalog(tmp_path)
    observation, context = catalog.values("interior", "mixed", "2015-01-01")
    prediction, gold = catalog.values("interior", "mixed", "2015-01-01-om4", "om4")
    assert np.all(observation == 2) and np.all(context == 3)
    assert np.all(prediction - gold == 2)
    assert np.all(catalog.values("interior", "mixed", "2015-01-01", "om4")[1] == gold)
    with pytest.raises(KeyError):
        catalog.values("interior", "mixed", "2015-01-01-om4", "observations")
