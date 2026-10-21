# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Calendar, causal-state, and restart invariants of member replay."""

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from samudra.experiments.stochastic_state_replay import (
    ReplayPool,
    continue_members,
    replay_parts,
    window_layout,
)


def states(count=9, offset=0):
    return [
        torch.full((2, 1, 2, 3, 1, 1), float(i + offset), requires_grad=True)
        for i in range(count)
    ]


def test_exact_month_overlap_and_rollover():
    start, end, dates, weights, duration, position = window_layout(
        "2001-12-01", "2002-01"
    )
    assert (start, end) == (6, 13)
    assert dates[0] == pd.Timestamp("2001-12-31")
    np.testing.assert_allclose(weights, np.array([4, 5, 5, 5, 5, 5, 2]) / 31)
    assert duration == 31 and position == 46.5
    assert weights.sum() == pytest.approx(1)
    assert window_layout("2000-01-01", "2000-02")[4] == 29
    with pytest.raises(ValueError, match="monthly origins"):
        window_layout("2000-01-02", "2000-02")


def test_boundary_capture_precedes_overlapping_bin_and_detaches_members():
    pool = ReplayPool()
    trajectory = states()
    trajectory[6].data[1] += 100
    pool.add_next("2001-12-01", "2001-12", trajectory, 20)
    entry = pool.draw(np.random.default_rng(1), 20)
    assert entry["month"] == "2002-01" and entry["bin_index"] == 6
    assert not entry["latent"].requires_grad
    torch.testing.assert_close(entry["latent"], trajectory[6])
    trajectory[6].data.zero_()
    assert entry["latent"][0].mean() == 6
    assert entry["latent"][1].mean() == 106


def test_split_horizon_capacity_and_original_age_survive_checkpoint():
    pool = ReplayPool()
    pool.add_next("2013-07", "2013-07", states(), 10)
    assert pool.draw(np.random.default_rng(1), 10) is None
    pool.add_next("2001-01", "2001-12", states(), 10)
    assert pool.draw(np.random.default_rng(1), 10) is None
    for step in range(10):
        pool.add_next("2001-01", "2001-01", states(), step)
    assert len(pool.entries[1]) == 8
    pool.add_next("2001-01", "2001-02", states(), 200, origin_step=0)
    assert pool.entries[2][0]["created_step"] == 200
    assert pool.entries[2][0]["origin_step"] == 0
    stream = io.BytesIO()
    torch.save(pool.state_dict(), stream)
    stream.seek(0)
    restored = ReplayPool()
    restored.load_state_dict(torch.load(stream, weights_only=False))
    a, b = np.random.default_rng(19), np.random.default_rng(19)
    for _ in range(20):
        left, right = pool.draw(a, 250), restored.draw(b, 250)
        assert left["month"] == right["month"]
        torch.testing.assert_close(left["latent"], right["latent"])
    restored.draw(b, 251)
    assert not restored.entries[
        2
    ]  # Chain extension never refreshed initialization age.
    assert restored.draw(b, 260) is None
    with pytest.raises(ValueError, match="configuration"):
        ReplayPool(max_age=1).load_state_dict(pool.state_dict())


def test_continuation_preserves_member_histories_without_reinitialization():
    class Model:
        arm = "fresh"

        def advance_members(self, state, forcing, context, *, generator, task):
            assert task == "observation"
            noise = torch.rand(state[:, :, -1].shape, generator=generator)
            return torch.stack(
                (state[:, :, -1], state[:, :, -1] + forcing + noise), 2
            ), None

    initial = states(1)[0]
    initial.data[1] += 100
    force = torch.ones(1, 3, 3, 1, 1)
    contexts = torch.zeros(1, 3, 5, 1, 1)
    result = continue_members(
        Model(), initial, force, contexts, torch.Generator().manual_seed(4)
    )
    repeated = continue_members(
        Model(), initial, force, contexts, torch.Generator().manual_seed(4)
    )
    assert len(result) == 4 and not result[0].requires_grad
    for index in range(1, 4):
        torch.testing.assert_close(result[index][:, :, 0], result[index - 1][:, :, 1])
        torch.testing.assert_close(result[index], repeated[index])
        assert result[index][1].mean() > 99
    with pytest.raises(ValueError, match="AR1"):
        continue_members(
            SimpleNamespace(arm="ar1"), initial, force, contexts, torch.Generator()
        )


def test_replay_rejects_leaking_month_and_mismatched_state_timestamp_before_loading():
    entry = {"origin": "2013-01-01", "month": "2013-08", "bin_index": 42}
    with pytest.raises(ValueError, match="training split"):
        next(replay_parts(None, None, None, entry, 1))
    entry.update(month="2013-02", bin_index=0)
    with pytest.raises(ValueError, match="timestamp"):
        next(replay_parts(None, None, None, entry, 1))


def test_interval_cache_uses_original_phase_raw_surface_and_normalization(tmp_path):
    from samudra.experiments.stochastic_state_replay import ReplayIntervals

    dates = pd.date_range("2001-01-01", periods=10)
    np.save(
        tmp_path / "starts.npy", dates.values.astype("datetime64[D]").astype(np.int64)
    )
    values = np.arange(10 * 5 * 2 * 3).reshape(10, 5, 2, 3).astype(np.float32)
    values[4, 0, 0, 0] = np.nan
    np.save(tmp_path / "values.npy", values)
    data = SimpleNamespace(
        grid={
            "mask": np.ones((77, 2, 3), bool),
            "mean": np.ones(77),
            "std": np.full(77, 2),
            "lat": np.array([-20.0, 20.0]),
            "lon": np.array([0.0, 120.0, 240.0]),
        },
        stats={"atmosphere_mean": np.ones(3), "atmosphere_std": np.full(3, 4)},
        tensor=lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32),
    )
    cache = ReplayIntervals(tmp_path)
    result = cache.load(data, dates[[4, 9]])
    np.testing.assert_allclose(
        result["raw_surface"][0], values[[4, 9], :2], equal_nan=True
    )
    assert result["surface"][0, 0, 0, 0, 0] == 0
    assert result["validity"][0, 0, 0, 0, 0] == 0
    np.testing.assert_allclose(result["atmosphere"][0], (values[[4, 9], 2:] - 1) / 4)
    with pytest.raises(ValueError, match="training cache"):
        cache.load(data, pd.date_range("2002-01-01", periods=1))
