"""Naming the tile that diverged, not just that the row did.

A face row advances 36 tiles on one shared cursor. When it runs away, every
tile in the row is discarded together -- so unless the report says which tile
went first and by how much, a diverging run tells you nothing about where the
problem is.
"""

import math

import pytest
import torch

from ocean_emulators.replay import (
    TileDivergence,
    describe_divergence,
    diagnose_replay_state,
)

LIMIT = 50.0


def healthy(tiles: int = 4, channels: int = 2, size: int = 5) -> torch.Tensor:
    return torch.full((tiles, channels, size, size), 1.5)


def test_a_healthy_row_reports_nothing() -> None:
    assert diagnose_replay_state(healthy(), max_state_sigma=LIMIT) == []


def test_an_ungrouped_row_is_treated_as_one_tile() -> None:
    """A `[C, H, W]` row predates grouping; it must still be diagnosable."""
    state = torch.zeros(3, 4, 4)
    assert diagnose_replay_state(state, max_state_sigma=LIMIT) == []
    state[1, 2, 2] = 10 * LIMIT
    found = diagnose_replay_state(state, max_state_sigma=LIMIT)
    assert [entry.tile_index for entry in found] == [0]


def test_only_the_offending_tiles_are_reported() -> None:
    state = healthy()
    state[2, 0, 1, 1] = 99.0
    found = diagnose_replay_state(state, max_state_sigma=LIMIT)
    assert [entry.tile_index for entry in found] == [2]
    assert found[0].max_abs == pytest.approx(99.0)
    assert found[0].nonfinite_cells == 0
    assert found[0].limit == LIMIT


def test_a_nonfinite_cell_diverges_regardless_of_the_limit() -> None:
    """Without a sigma limit the only test left is finiteness, and it must
    still fire -- a NaN seed poisons every rollout drawn from it."""
    state = healthy()
    state[1, 0, 0, 0] = float("nan")
    for limit in (0.0, LIMIT):
        found = diagnose_replay_state(state, max_state_sigma=limit)
        assert [entry.tile_index for entry in found] == [1]
        assert found[0].nonfinite_cells == 1


def test_the_magnitude_reported_for_a_nonfinite_tile_is_the_finite_one() -> None:
    """`.abs().max()` over a tile holding a NaN is NaN, which throws away how
    far the finite cells had actually run before overflowing."""
    state = healthy()
    state[3] = 0.0
    state[3, 0, 0, 0] = float("inf")
    state[3, 1, 1, 1] = -123.0
    found = diagnose_replay_state(state, max_state_sigma=LIMIT)
    assert found[0].max_abs == pytest.approx(123.0)
    assert math.isfinite(found[0].max_abs)


def test_an_entirely_nonfinite_tile_reports_zero_rather_than_crashing() -> None:
    state = healthy()
    state[0] = float("nan")
    found = diagnose_replay_state(state, max_state_sigma=LIMIT)
    assert found[0].max_abs == 0.0
    assert found[0].nonfinite_cells == state[0].numel()


def test_a_zero_limit_disables_the_magnitude_check() -> None:
    state = healthy()
    state[2, 0, 1, 1] = 1e6
    assert diagnose_replay_state(state, max_state_sigma=0.0) == []


def test_the_limit_is_exclusive_so_a_state_exactly_at_it_survives() -> None:
    state = healthy()
    state[2, 0, 1, 1] = LIMIT
    assert diagnose_replay_state(state, max_state_sigma=LIMIT) == []


def test_a_bad_rank_state_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unexpected shape"):
        diagnose_replay_state(torch.zeros(2, 2), max_state_sigma=LIMIT)


def test_the_summary_ranks_nonfinite_above_merely_large() -> None:
    """A NaN is a harder failure than a large value, so it leads the report
    even when a different tile has the bigger finite magnitude."""
    found = [
        TileDivergence(tile_index=7, max_abs=1e9, nonfinite_cells=0, limit=LIMIT),
        TileDivergence(tile_index=2, max_abs=60.0, nonfinite_cells=3, limit=LIMIT),
    ]
    summary = describe_divergence(found)
    assert summary.index("tile 2") < summary.index("tile 7")
    assert "2 tile(s) diverged" in summary


def test_the_summary_truncates_a_whole_bad_face() -> None:
    found = [
        TileDivergence(tile_index=index, max_abs=100.0 + index, nonfinite_cells=0, limit=LIMIT)
        for index in range(36)
    ]
    summary = describe_divergence(found)
    assert "36 tile(s) diverged" in summary
    assert "and 33 more tile(s)" in summary
    # Worst first: tile 35 has the largest magnitude.
    assert "tile 35" in summary


def test_the_summary_of_a_clean_row_says_so() -> None:
    assert describe_divergence([]) == "no divergence"
