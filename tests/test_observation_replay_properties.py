# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Replay pools, cached-state boundaries and replay calendar arithmetic.

No data, model or GPU: replay entries carry tagged constant tensors, so the
oracle can identify every state the pool returns.
"""

import copy

import numpy as np
import pandas as pd
import torch
from hypothesis import example, given
from hypothesis import strategies as st

from samudra.observations.constants import (
    FORECAST_STEPS,
    MAX_REPLAY_MONTHS,
    STATE_HISTORY_STEPS,
    STEP_DAYS,
)
from samudra.observations.evaluate import complete_month_weights
from samudra.observations.intervals import calendar_weights, replay_month
from samudra.observations.replay import ReplayEntry, ReplayPool, states_at


def entry(tag: int, month: int, origin_update: int) -> ReplayEntry:
    return ReplayEntry(
        "obs",
        tag,
        month,
        start_lead=6 * month,
        origin_update=origin_update,
        states=torch.full((1, STATE_HISTORY_STEPS, 1, 1, 1), float(tag)),
    )


# Each operation either stores a state continued from an initialization that
# happened `age` updates ago, or advances one optimizer update and draws.
operations = st.lists(
    st.one_of(
        # Two months, one at the horizon edge: buckets overflow often enough for
        # capacity bugs to show up within the PR example budget.
        st.tuples(
            st.just("add"),
            st.sampled_from([1, MAX_REPLAY_MONTHS - 1]),
            st.integers(0, 6),
        ),
        st.tuples(st.just("draw"), st.just(0), st.just(0)),
    ),
    max_size=40,
)


def stored(pool: ReplayPool) -> dict[int, list[int]]:
    """The checkpointed contents: tags per month, in storage order."""
    entries = pool.state_dict()["entries"]
    return {
        m: [int(e.origin) for e in bucket] for m, bucket in entries.items() if bucket
    }


# Overflowing a one-entry bucket must evict the older state, not the newer one.
@example([("add", 1, 0), ("add", 1, 0), ("draw", 0, 0)], 1, 5, 0)
@given(
    operations,
    st.integers(1, 3),
    st.integers(0, 5),
    st.integers(0, 2**32 - 1),
)
def test_pool_matches_reference_capacity_and_age_window(ops, per_month, max_age, seed):
    """Oracle: per-month FIFO lists of the newest `per_month` entries, filtered by
    original trajectory age. After every operation the checkpointed contents equal
    the reference; every draw comes from the eligible set, and the pool returns
    nothing only when that set is empty."""
    pool = ReplayPool("obs", seed, per_month=per_month, max_age=max_age)
    reference: dict[int, list[tuple[int, int]]] = {}
    update = 0
    for tag, (kind, month, age) in enumerate(ops):
        if kind == "add":
            origin_update = max(0, update - age)
            pool.add(entry(tag, month, origin_update))
            bucket = reference.setdefault(month, [])
            bucket.append((tag, origin_update))
            del bucket[:-per_month]
            assert stored(pool) == {
                m: [t for t, _ in b] for m, b in reference.items() if b
            }
            continue
        update += 1
        # Drawing discards expired entries; age only grows, so they never return.
        reference = {
            m: [(t, born) for t, born in bucket if update - born <= max_age]
            for m, bucket in reference.items()
        }
        eligible = {tag for bucket in reference.values() for tag, _ in bucket}
        drawn = pool.draw(update)
        if not eligible:
            assert drawn is None
            continue
        assert drawn is not None
        assert drawn.origin in eligible
        assert torch.equal(
            drawn.states, torch.full_like(drawn.states, float(drawn.origin))
        )


@given(operations, st.data(), st.integers(0, 2**32 - 1))
def test_restored_pool_draws_exactly_like_the_original(ops, data, seed):
    """Oracle: the uninterrupted pool. A pool restored from a checkpoint taken at
    any point must make identical draws afterwards, and the checkpoint must not
    alias the live pool's later mutations."""
    split = data.draw(st.integers(0, len(ops)))
    original = ReplayPool("obs", seed, per_month=2, max_age=3)
    update = 0

    def run(pool: ReplayPool, sequence, start: int, update: int) -> list:
        draws = []
        for tag, (kind, month, age) in enumerate(sequence, start):
            if kind == "add":
                pool.add(entry(tag, month, max(0, update - age)))
            else:
                update += 1
                drawn = pool.draw(update)
                draws.append(None if drawn is None else drawn.origin)
        return draws

    run(original, ops[:split], 0, update)
    update = sum(kind == "draw" for kind, _, _ in ops[:split])
    checkpoint = copy.deepcopy(original.state_dict())
    expected = run(original, ops[split:], split, update)
    restored = ReplayPool("obs", seed + 1, per_month=2, max_age=3)
    restored.load_state_dict(checkpoint)
    assert run(restored, ops[split:], split, update) == expected


@given(st.integers(1, 6), st.data())
def test_cached_states_are_the_last_two_frames_of_the_trajectory(length, data):
    """Oracle: concatenate the initial pair with the predictions; after `steps`
    model calls the state is the two frames ending at position steps + 2."""
    steps = data.draw(st.integers(0, length))
    initial = torch.arange(STATE_HISTORY_STEPS, dtype=torch.float32)
    predictions = torch.arange(length, dtype=torch.float32) + STATE_HISTORY_STEPS
    shape = (1, -1, 1, 1, 1)
    result = states_at(initial.view(shape), predictions.view(shape), steps)
    trajectory = torch.cat((initial, predictions))
    assert torch.equal(
        result.flatten(), trajectory[steps : steps + STATE_HISTORY_STEPS]
    )


month_starts = st.dates(
    min_value=pd.Timestamp("1950-01-01").date(),
    max_value=pd.Timestamp("2100-12-01").date(),
).map(lambda day: pd.Timestamp(day).replace(day=1))


@given(month_starts)
def test_replay_months_hand_off_and_cover_their_calendar_month(origin):
    """Oracle: day arithmetic on the five-day lattice from the origin. Each month's
    steps [start, end) must cover the whole month, the cached boundary must be
    the step containing the next month's first day, and that boundary must be
    exactly where the next month's replay begins."""
    previous_next = 0
    for offset in range(MAX_REPLAY_MONTHS):
        first = origin + pd.DateOffset(months=offset)
        last = first + pd.offsets.MonthBegin()
        start, end, following = replay_month(origin, first)
        begin_day, end_day = (first - origin).days, (last - origin).days
        assert start == previous_next
        assert start * STEP_DAYS <= begin_day < (start + 1) * STEP_DAYS
        assert (end - 1) * STEP_DAYS < end_day <= end * STEP_DAYS
        assert following * STEP_DAYS <= end_day < (following + 1) * STEP_DAYS
        previous_next = following


@given(month_starts, st.integers(0, MAX_REPLAY_MONTHS - 1))
def test_training_and_evaluation_month_weights_agree(origin, offset):
    """Oracle: the evaluator's independent calendar-overlap operator. Training's
    replay weights for one month must equal the evaluator's row for that month
    on the same five-day lattice, and sum to one."""
    first = origin + pd.DateOffset(months=offset)
    last = first + pd.offsets.MonthBegin()
    start, end, _ = replay_month(origin, first)
    if end > FORECAST_STEPS:
        return
    starts = [origin + pd.Timedelta(days=STEP_DAYS * k) for k in range(start, end)]
    ends = [stamp + pd.Timedelta(days=STEP_DAYS) for stamp in starts]
    weights = calendar_weights(
        [s.to_pydatetime() for s in starts],
        [e.to_pydatetime() for e in ends],
        first.to_pydatetime(),
        last.to_pydatetime(),
    )
    months, matrix = complete_month_weights(
        np.array(starts, dtype="datetime64[ns]"), np.array(ends, dtype="datetime64[ns]")
    )
    np.testing.assert_allclose(
        weights, matrix[months.index(first.strftime("%Y-%m"))], atol=1e-6
    )
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-6)
