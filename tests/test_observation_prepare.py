# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from samudra.experiments.observation_prepare import ConservativeRemap, interpolate_depth


def test_conservative_remap_constant_and_integral():
    source_lat = np.arange(-89.5, 90, 1)
    source_lon = np.arange(0.5, 360, 1)
    target_lat = np.arange(-89, 90, 2)
    target_lon = np.arange(1, 360, 2)
    r = ConservativeRemap(source_lat, source_lon, target_lat, target_lon)
    field = np.broadcast_to(np.sin(np.deg2rad(source_lat))[:, None] + 2, (180, 360))
    mapped, coverage = r(field)
    np.testing.assert_allclose(coverage, 1, atol=1e-6)
    np.testing.assert_allclose((mapped * r.area).sum(), 1440, rtol=1e-6)
    constant, _ = r(np.full_like(field, 7))
    np.testing.assert_allclose(constant, 7)


def test_periodic_cell_spanning_prime_meridian():
    lat = np.array([-45, 45])
    r = ConservativeRemap(lat, np.arange(0, 360, 90), lat, np.arange(45, 360, 90))
    field = np.broadcast_to([0.0, 1.0, 2.0, 3.0], (2, 4))
    mapped, coverage = r(field)
    np.testing.assert_allclose(mapped[0], [0.5, 1.5, 2.5, 1.5])
    np.testing.assert_allclose(coverage, 1)


def test_missing_area_is_not_silently_filled():
    r = ConservativeRemap([-45, 45], [45, 135, 225, 315], [-45, 45], [90, 270])
    values = np.array([[1.0, np.nan, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]])
    mapped, coverage = r(values)
    assert np.isnan(mapped[0, 0])
    assert coverage[0, 0] == 0.5
    assert mapped[1, 0] == 3


def test_vertical_interpolation_does_not_bridge_missing_levels():
    values = np.array([1.0, np.nan, 3.0, 4.0])[:, None, None]
    result = interpolate_depth(values, [1, 10, 100, 1000], [0, 1, 5, 50, 550, 1500])[
        :, 0, 0
    ]
    np.testing.assert_allclose(
        result, [np.nan, 1, np.nan, np.nan, 3.5, np.nan], equal_nan=True
    )


def test_month_support_and_leap_year():
    from samudra.experiments.observation_data import month_intervals

    for month, leads, last in [("2012-02", 6, 4), ("2013-02", 6, 3), ("2013-07", 7, 1)]:
        starts, ends, weights = month_intervals(month)
        assert len(weights) == leads
        assert (ends[18] - starts[0]).days == 95
        assert ends[18] == starts[19]
        assert starts[19].strftime("%Y-%m") == month
        assert np.isclose(weights.sum(), 1)
        assert np.isclose(weights[-1] / weights[0], last / 5)


def test_five_day_mean_rejects_incomplete_cells():
    from samudra.experiments.observation_data import strict_mean

    fields = np.ones((5, 2, 2))
    fields[0, 0, 0] = np.nan
    result = strict_mean(fields)
    assert np.isnan(result[0, 0])
    assert result[1, 1] == 1
