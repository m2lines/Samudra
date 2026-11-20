# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint branches preserve state and reject scientific recipe changes."""

import hashlib

import pytest
import torch

from samudra.skyline.branching import load_branch, load_resume, scheduled_learning_rate
from samudra.skyline.multitask import learning_rate_factor


@pytest.fixture
def branch(tmp_path):
    original = dict(world_size=2, crop=33, learning_rate=2e-4, max_updates=25000)
    state = dict(
        config=original,
        source={"dataset": "same"},
        step=25000,
        rank_states=[{"rng": 123, "stream": 456}, {"rng": 789, "stream": 987}],
        model={"weight": torch.tensor([3.0])},
        optimizer={"state": {0: {"exp_avg": torch.tensor([0.25])}}},
        ema={"num_updates": 25000},
    )
    path = tmp_path / "parent.pt"
    torch.save(state, path)
    config = dict(
        original,
        max_updates=28000,
        learning_rate=8e-4,
        learning_rate_schedule="constant",
        branch_checkpoint=dict(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            step=25000,
        ),
    )
    return config, state


def test_branch_preserves_optimizer_rng_ema_and_parent_file(branch):
    config, original = branch
    loaded = load_branch(config, original["source"])
    assert loaded["rank_states"] == original["rank_states"]
    assert loaded["ema"] == original["ema"]
    torch.testing.assert_close(loaded["model"]["weight"], original["model"]["weight"])
    torch.testing.assert_close(
        loaded["optimizer"]["state"][0]["exp_avg"], torch.tensor([0.25])
    )
    assert loaded["config"]["learning_rate"] == 2e-4


@pytest.mark.parametrize("change", [{"crop": 17}, {"world_size": 1}])
def test_branch_rejects_changed_recipe(branch, change):
    config, state = branch
    with pytest.raises(ValueError, match="scientific configuration"):
        load_branch(dict(config, **change), state["source"])


def test_branch_rejects_checksum_source_and_step(branch):
    config, state = branch
    with pytest.raises(ValueError, match="provenance"):
        load_branch(config, {"dataset": "different"})
    config["branch_checkpoint"]["step"] = 24999
    with pytest.raises(ValueError, match="update differs"):
        load_branch(config, state["source"])
    config["branch_checkpoint"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        load_branch(config, state["source"])


def test_constant_schedule_has_no_warmup_or_decay_and_default_is_unchanged():
    config = dict(learning_rate=4e-4, warmup=200, max_updates=28000)
    for step in [1, 25000, 25001, 28000]:
        assert (
            scheduled_learning_rate(
                dict(config, learning_rate_schedule="constant"), step
            )
            == 4e-4
        )
        assert scheduled_learning_rate(config, step) == 4e-4 * learning_rate_factor(
            step, 200, 28000
        )


def test_manual_resume_requires_authorization_and_exact_provenance(branch):
    config, original = branch
    path = config["branch_checkpoint"]["path"]
    with pytest.raises(FileExistsError):
        load_resume(original["config"], original["source"], path, False)
    loaded = load_resume(original["config"], original["source"], path, True)
    assert loaded["step"] == 25000
    assert loaded["rank_states"] == original["rank_states"]
    for changed_config, changed_source in [
        (dict(original["config"], learning_rate=8e-4), original["source"]),
        (original["config"], {"dataset": "different"}),
    ]:
        with pytest.raises(ValueError, match="provenance changed"):
            load_resume(changed_config, changed_source, path, True)
