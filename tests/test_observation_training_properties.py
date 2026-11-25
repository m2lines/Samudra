# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint/resume equivalence of the observation training loop.

Uses the recipe harness's tiny sources and objective: a one-weight model whose
loss consumes every RNG stream, and replay states tagged by source and lead.
"""

import tempfile
from pathlib import Path
from typing import Any, cast
from unittest import mock

import torch
from hypothesis import assume, given
from hypothesis import strategies as st

from samudra.observations import training as baseline
from tests.test_observation_baseline_harness import (
    TinySources,
    make_state,
    recipe,
    tiny_objective,
    validate,
)

SCHEDULERS = st.one_of(
    st.none(),
    st.just({"type": "cosine"}),
    st.just({"type": "linear_cooldown", "final_lr": 1e-6}),
    st.just({"type": "cosine_with_warmup", "warmup_epochs": 1, "warmup_lr": 1e-6}),
)


@st.composite
def plans(draw):
    main = draw(st.integers(1, 5))
    post = draw(st.integers(0, 3))
    accumulation = draw(st.integers(1, 3))
    scheduler, post_scheduler = draw(SCHEDULERS), draw(SCHEDULERS)
    # Warmup must fit inside its own phase.
    assume(scheduler is None or scheduler["type"] != "cosine_with_warmup" or main > 1)
    assume(
        post_scheduler is None
        or post_scheduler["type"] != "cosine_with_warmup"
        or post > 1
    )
    cfg = recipe(
        draw(st.sampled_from(["obs", "om4-obs", "om4-cm4-obs", "om4-cm4"])),
        main_updates=main,
        post_training_updates=post,
        gradient_accumulation_steps=accumulation,
        fresh_examples=draw(st.integers(1, accumulation)),
        replay={
            "per_month": draw(st.integers(1, 2)),
            "max_age": draw(st.integers(0, 4)),
        },
        scheduler=scheduler,
        post_training_scheduler=post_scheduler,
        post_training_learning_rate=draw(st.sampled_from([None, 5e-5])),
        validation_freq=draw(st.integers(1, 3)),
        save_freq=draw(st.integers(1, 3)),
    )
    return cfg, draw(st.integers(1, cfg.total_updates))


def run(cfg, state, output: Path, records: list, **kwargs) -> None:
    baseline.train(
        cfg,
        state,
        cast(baseline.Sources, TinySources()),
        output,
        {"plan": "resume"},
        validate,
        records.append,
        **kwargs,
    )


@given(plans())
def test_resume_at_any_update_matches_uninterrupted_training(plan):
    """Oracle: the uninterrupted run. Stopping after any update, at a phase
    boundary or not, and restoring the latest checkpoint must reproduce every
    logged record, the weights, optimizer moments, EMA, best score, exposure and
    all later replay draws exactly."""
    cfg, stop = plan
    with (
        mock.patch.object(baseline, "objective", tiny_objective),
        tempfile.TemporaryDirectory() as directory,
    ):
        full_dir, part_dir = Path(directory, "full"), Path(directory, "part")
        full_dir.mkdir()
        part_dir.mkdir()
        full, full_records = make_state(cfg), cast(list[dict[str, Any]], [])
        run(cfg, full, full_dir, full_records)
        resumed_records: list[dict[str, Any]] = []
        run(cfg, make_state(cfg), part_dir, resumed_records, stop_after=stop)
        resumed = make_state(cfg)
        resumed.restore(part_dir / "ckpt.pt", {"plan": "resume"})
        run(cfg, resumed, part_dir, resumed_records)

    assert resumed_records == full_records
    torch.testing.assert_close(resumed.model.weight, full.model.weight, rtol=0, atol=0)
    assert resumed.ema == full.ema
    assert (resumed.best_score, resumed.exposure) == (full.best_score, full.exposure)
    left = resumed.optimizer.state[resumed.model.weight]
    right = full.optimizer.state[full.model.weight]
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    for name in TinySources.names:
        for _ in range(5):
            mine = resumed.pools[name].draw(cfg.total_updates)
            theirs = full.pools[name].draw(cfg.total_updates)
            assert (mine is None) == (theirs is None)
            if mine is not None and theirs is not None:
                assert (mine.origin, mine.month, mine.origin_update) == (
                    theirs.origin,
                    theirs.month,
                    theirs.origin_update,
                )
