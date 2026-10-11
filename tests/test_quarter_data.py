# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import datetime
import math
from types import SimpleNamespace
from typing import cast

import cftime
import numpy as np
import pytest
import torch

from samudra.experiments.initializer_wave import InitializerWave
from samudra.experiments.quarter_data import QuarterOm4Windows
from samudra.utils.data import BatchPreprocessor, ChannelStatistics, Masks


@pytest.fixture
def source_and_reader():
    # Storage and canonical orders deliberately differ; a sliced source starts
    # at physical frame 11, so a reader treating relative indices as physical fails.
    storage_names = ("hfds", "so_0", "thetao_0", "tauvo", "zos", "tauuo")
    names = ("zos", "thetao_0", "so_0")
    rng = np.random.default_rng(61)
    raw = rng.normal(size=(70, 6, 3, 4)).astype("f4")
    raw += np.arange(70, dtype="f4")[:, None, None, None] * 0.13
    mask = torch.ones(3, 3, 4, dtype=torch.bool)
    mask[0, 0, 0] = False
    mask[1, 1, 2] = False
    mask[2, 2, 1] = False
    boundary_mask = torch.ones(3, 4, dtype=torch.bool)
    boundary_mask[2, 3] = False
    raw[:, storage_names.index("thetao_0"), 1, 2] = np.nan
    raw[30, storage_names.index("so_0"), 1, 1] = np.nan  # native wet NaN policy
    raw[:, storage_names.index("tauuo"), 2, 3] = np.nan
    all_dates = np.array(
        [
            cftime.DatetimeNoLeap(1999, 10, 1) + datetime.timedelta(days=5 * i)
            for i in range(70)
        ]
    )
    physical = np.arange(11, 51)
    mean = np.array([1.2, -3.1, 2.72, 0.37, 10.3, -0.91], dtype="f4")
    std = np.array([0.7, 2.6, 0.3, 1.9, 3.3, 0.81], dtype="f4")

    def statistics(channels):
        positions = [storage_names.index(n) for n in channels]
        return ChannelStatistics(mean[positions], std[positions])

    source = SimpleNamespace(
        data_layout=SimpleNamespace(prognostic_var_names=names),
        time=SimpleNamespace(values=all_dates[physical]),
        resolution=(
            torch.tensor([-60.0, 0.0, 45.0]),
            torch.tensor([0.5, 90.5, 180.5, 270.5]),
        ),
        masks=Masks(mask, boundary_mask),
        statistics=statistics,
    )

    class Reader:
        def __init__(self):
            self.calls = []

        def physical_indices(self, relative):
            return physical[relative]

        def read_unique(self, indices, *, channels):
            self.calls.append((indices.copy(), channels))
            selected = [storage_names.index(n) for n in channels]
            return torch.from_numpy(raw[indices][:, selected].copy())

    return source, Reader(), raw, storage_names, physical


def reference_sample(source, raw, storage_names, physical, start, observation):
    """Independent full-history preparation through the pre-existing pipeline."""
    names = source.data_layout.prognostic_var_names
    forcing_names = ("tauuo", "tauvo", "hfds")
    preparer = BatchPreprocessor(source, names, forcing_names)
    window = raw[physical[start : start + 25]]
    state = preparer.prepare_prognostic(
        torch.from_numpy(window[:, [storage_names.index(n) for n in names]]),
        torch.device("cpu"),
    ).reshape(1, 25, len(names), 3, 4)
    boundary = preparer.prepare_boundary(
        torch.from_numpy(window[:, [storage_names.index(n) for n in forcing_names]]),
        torch.device("cpu"),
    ).reshape(1, 25, 3, 3, 4)
    surface_ids = [names.index("thetao_0"), names.index("zos")]
    lat, lon = (coordinate.deg2rad() for coordinate in source.resolution)
    xyz = torch.stack(
        (
            lat.cos()[:, None] * lon.cos()[None],
            lat.cos()[:, None] * lon.sin()[None],
            lat.sin()[:, None].expand(3, 4),
        )
    )
    day = source.time.values[start + 18].dayofyr
    phase = 2 * math.pi * (day - 1) / 365.25
    season = torch.tensor([math.sin(phase), math.cos(phase)])[:, None, None].expand(
        2, 3, 4
    )
    native_sample = (
        state[:, :19, surface_ids],
        boundary[:, :19],
        torch.cat((xyz, season))[None],
        state[:, 17:19],
        boundary[:, 18:24],
        state[:, 19:25],
    )
    stats = source.statistics(names)
    old = SimpleNamespace(
        sample=lambda *_: native_sample,
        scaling_contract={} if observation is not None else None,
        native_mean=torch.from_numpy(stats.mean),
        native_std=torch.from_numpy(stats.std),
        mean=torch.from_numpy(
            observation.mean if observation is not None else stats.mean
        ),
        std=torch.from_numpy(observation.std if observation is not None else stats.std),
        mask=source.masks.prognostic,
        initializer=SimpleNamespace(surface=surface_ids),
    )
    return InitializerWave.model_sample(cast(InitializerWave, old), None, [start])


@pytest.mark.parametrize("observation_normalization", [False, True])
def test_windows_match_existing_pipeline_at_edges_and_overlaps(
    source_and_reader, observation_normalization
):
    source, reader, raw, storage_names, physical = source_and_reader
    observation = (
        ChannelStatistics(
            np.array([-0.71, 7.13, 0.5], dtype="f4"),
            np.array([1.13, 2.17, 0.29], dtype="f4"),
        )
        if observation_normalization
        else None
    )
    windows = QuarterOm4Windows(source, reader, observation_statistics=observation)
    assert len(windows) == 16
    for start in (0, 15, 4, 5, 4):
        actual = windows.model_sample([start])
        expected = reference_sample(
            source, raw, storage_names, physical, start, observation
        )
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        # Masked cells are zero even after the native -> observation round trip.
        assert (actual[3][0, :, 0, 0, 0] == 0).all()
        assert (actual[1][0, :, :, 2, 3] == 0).all()


def test_restricted_origin_views_and_reads_are_bounded(source_and_reader):
    source, reader, _, _, physical = source_and_reader
    origins = np.array([15, 0, 6])
    windows = QuarterOm4Windows(source, reader, origins=origins)
    origins[0] = 1  # caller mutation must not silently change the sampler
    static_tensor_elements = sum(
        v.numel() for v in vars(windows).values() if isinstance(v, torch.Tensor)
    )
    for index, start in ((0, 15), (1, 0), (2, 6), (0, 15)):
        outputs = windows.model_sample([index])
        calls = reader.calls[-3:]
        for (indices, names), a, b, expected_names in zip(
            calls,
            [start, start + 17, start],
            [start + 19, start + 25, start + 24],
            [
                ("thetao_0", "zos"),
                ("zos", "thetao_0", "so_0"),
                ("tauuo", "tauvo", "hfds"),
            ],
            strict=True,
        ):
            np.testing.assert_array_equal(indices, physical[a:b])
            assert names == expected_names
        assert sum(len(t) * len(c) for t, c in calls) == 19 * 2 + 8 * 3 + 24 * 3
        # An output cannot corrupt a future read, and no output becomes a cache.
        outputs[0].fill_(1e9)
        assert (
            sum(
                v.numel() for v in vars(windows).values() if isinstance(v, torch.Tensor)
            )
            == static_tensor_elements
        )
    assert len(reader.calls) == 12
    assert windows.model_sample([0])[0].abs().max() < 1e9


@pytest.mark.parametrize("ids", [[], [0, 1], [-1], [16], [0.5]])
def test_invalid_requests_do_not_read(source_and_reader, ids):
    source, reader, *_ = source_and_reader
    windows = QuarterOm4Windows(source, reader)
    with pytest.raises(ValueError, match="one valid"):
        windows.model_sample(ids)
    assert not reader.calls


@pytest.mark.parametrize("origins", [[], [-1], [16], [0.5], [[0]], [2**63 - 1]])
def test_invalid_origin_views_fail_before_reads(source_and_reader, origins):
    source, reader, *_ = source_and_reader
    with pytest.raises(ValueError, match="Origins"):
        QuarterOm4Windows(source, reader, origins=origins)
    assert not reader.calls


def test_missing_time_bin_fails_before_reads(source_and_reader):
    source, reader, *_ = source_and_reader
    source.time.values[20] += datetime.timedelta(days=1)
    with pytest.raises(ValueError, match="five-day"):
        QuarterOm4Windows(source, reader)
    assert not reader.calls


def test_numpy_dates_use_origin_season_not_storage_index(source_and_reader):
    source, reader, *_ = source_and_reader
    source.time.values = np.datetime64("2000-01-01") + np.arange(40) * np.timedelta64(
        5, "D"
    )
    windows = QuarterOm4Windows(source, reader, origins=[15])
    context = windows.model_sample([0])[2]
    # A leap-year date deliberately distinguishes calendar-day phase handling.
    expected_date = datetime.date(2000, 1, 1) + datetime.timedelta(days=33 * 5)
    phase = 2 * math.pi * (expected_date.timetuple().tm_yday - 1) / 365.25
    expected = torch.tensor([math.sin(phase), math.cos(phase)])
    torch.testing.assert_close(context[0, 3:, 0, 0], expected, rtol=0, atol=0)


def test_bad_native_output_is_rejected(source_and_reader):
    source, reader, *_ = source_and_reader
    windows = QuarterOm4Windows(source, reader)
    read = reader.read_unique
    reader.read_unique = lambda *args, **kwargs: read(*args, **kwargs).double()
    with pytest.raises(ValueError, match="ordered fp32"):
        windows.model_sample([0])


def test_invalid_normalization_is_rejected_without_reads(source_and_reader):
    source, reader, *_ = source_and_reader
    stats = ChannelStatistics(np.zeros(3, dtype="f4"), np.array([1, 0, 1], dtype="f4"))
    with pytest.raises(ValueError, match="positive scales"):
        QuarterOm4Windows(source, reader, observation_statistics=stats)
    assert not reader.calls


@pytest.mark.parametrize(
    "date_type", [cftime.DatetimeJulian, cftime.DatetimeGregorian, datetime.datetime]
)
def test_om4_calendar_retains_verified_leap_year_boundary(date_type):
    from samudra.experiments.quarter_data import valid_om4_interval

    assert valid_om4_interval(date_type(2012, 12, 28, 12), date_type(2013, 1, 3, 12))
    assert not valid_om4_interval(
        date_type(2011, 12, 28, 12), date_type(2012, 1, 3, 12)
    )
    assert not valid_om4_interval(date_type(2012, 6, 1, 12), date_type(2012, 6, 7, 12))


def test_numpy_om4_calendar_retains_verified_leap_year_boundary():
    from samudra.experiments.quarter_data import valid_om4_interval

    assert valid_om4_interval(
        np.datetime64("2012-12-28T12:00"), np.datetime64("2013-01-03T12:00")
    )
    assert not valid_om4_interval(
        np.datetime64("2012-06-01T12:00"), np.datetime64("2012-06-07T12:00")
    )
