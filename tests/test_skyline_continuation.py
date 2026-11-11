# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Continuation preserves trained/sampling state and changes only declared controls."""

import copy

import numpy as np
import pytest
import torch

from samudra.skyline.continuation import (
    checkpoint_sha256,
    comparison_learning_rate,
    continuation_learning_rate,
    load_comparison_checkpoint,
    restore_comparison_state,
    validate_continuation,
)
from samudra.skyline.ema import restore_ema
from samudra.skyline.multitask import learning_rate_factor


def schedule():
    return dict(
        start_step=2000,
        rewarm_updates=100,
        start_lr=2e-5,
        peak_lr=2e-4,
        end_lr=2e-5,
        end_step=25000,
    )


def test_schedule_is_continuous_and_matches_endpoints():
    s = schedule()
    assert continuation_learning_rate(2000, s) == pytest.approx(2e-5)
    assert continuation_learning_rate(2001, s) == pytest.approx(2.18e-5)
    assert continuation_learning_rate(2050, s) == pytest.approx(1.1e-4)
    assert continuation_learning_rate(2100, s) == pytest.approx(2e-4)
    assert continuation_learning_rate(25000, s) == pytest.approx(2e-5)
    rates = [continuation_learning_rate(i, s) for i in range(2000, 25001)]
    assert np.all(np.diff(rates[:101]) > 0)
    assert np.all(np.diff(rates[100:]) < 0)
    for step in (1999, 25001):
        with pytest.raises(ValueError):
            continuation_learning_rate(step, s)
    cfg = dict(learning_rate=2e-4, warmup=100, max_updates=2000)
    for step in (1, 100, 2000):
        assert comparison_learning_rate(step, cfg) == 2e-4 * learning_rate_factor(
            step, 100, 2000
        )


def parent_fixture(tmp_path):
    torch.manual_seed(3)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema_config = dict(decay=0.999, faster_decay_at_start=True)
    ema = restore_ema(model, ema_config)
    x = torch.randn(2, 3)
    for _ in range(2):
        optimizer.zero_grad()
        model(x).square().sum().backward()
        optimizer.step()
        ema(model)
    rng = np.random.default_rng(7)
    rng.random(11)
    cfg = dict(
        world_size=1,
        microbatch=32,
        accumulation=8,
        max_updates=2,
        schedule_updates=2,
        learning_rate=1e-3,
        warmup=1,
        model={"width": 32},
        coverage={"base_fraction": 0.5},
        ema=ema_config,
    )
    source = {"metadata_sha256": "example", "input_offsets": [-1, 0]}
    state = dict(
        config=cfg,
        source=source,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        ema=ema.get_state(True),
        step=2,
        best={"raw": 0.5, "ema": 0.4},
        rank_states=[
            dict(
                rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.tensor([], dtype=torch.uint8),
                stream={"origin": 17, "used": 512},
            )
        ],
    )
    parent = tmp_path / "parent.pt"
    torch.save(state, parent)
    child = dict(
        cfg,
        max_updates=10,
        schedule_updates=10,
        physical_microbatch=128,
        prefetch_microbatches=True,
        continuation_parent=str(parent),
        continuation_parent_sha256=checkpoint_sha256(parent),
        continuation_schedule=dict(
            start_step=2,
            rewarm_updates=2,
            start_lr=1e-3,
            peak_lr=2e-3,
            end_lr=1e-3,
            end_step=10,
        ),
    )
    return state, child, source, x


def test_continuation_restores_optimizer_ema_sampling_and_keeps_parent_immutable(
    tmp_path,
):
    state, cfg, source, x = parent_fixture(tmp_path)
    loaded, fresh, receipt = load_comparison_checkpoint(cfg, tmp_path / "child", source)
    assert fresh and receipt["parent_step"] == 2
    m1, m2 = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    o1, o2 = torch.optim.AdamW(m1.parameters()), torch.optim.AdamW(m2.parameters())
    r1, r2 = np.random.default_rng(4), np.random.default_rng(9)
    for model, opt, rng, checkpoint in ((m1, o1, r1, state), (m2, o2, r2, loaded)):
        step, best, stream = restore_comparison_state(
            model, opt, rng, checkpoint, 0, torch.device("cpu")
        )
        assert (
            step == 2
            and best == state["best"]
            and stream == state["rank_states"][0]["stream"]
        )
        assert torch.equal(torch.get_rng_state(), state["rank_states"][0]["torch_rng"])
    np.testing.assert_array_equal(r1.random(12), r2.random(12))
    e1 = restore_ema(m1, cfg["ema"], state["ema"], 2)
    e2 = restore_ema(m2, cfg["ema"], loaded["ema"], 2)
    for model, opt, ema in ((m1, o1, e1), (m2, o2, e2)):
        opt.zero_grad()
        model(x).square().sum().backward()
        opt.step()
        ema(model)
    for p, q in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.equal(p, q)
    assert int(e1.num_updates) == int(e2.num_updates) == 3
    for k, v in e1.get_state(True)["ema_params"].items():
        assert torch.equal(v, e2.get_state(True)["ema_params"][k])
    assert (
        checkpoint_sha256(cfg["continuation_parent"])
        == cfg["continuation_parent_sha256"]
    )


def test_protected_changes_and_missing_state_fail(tmp_path):
    state, cfg, source, _ = parent_fixture(tmp_path)
    for key, value in (
        ("microbatch", 64),
        ("accumulation", 4),
        ("coverage", {}),
        ("model", {"width": 64}),
        ("unknown_performance_switch", True),
    ):
        with pytest.raises(ValueError, match="protected configuration"):
            validate_continuation(state, dict(cfg, **{key: value}), source)
    for key, state_value in (("step", 1), ("rank_states", []), ("source", {})):
        with pytest.raises(ValueError):
            validate_continuation(dict(state, **{key: state_value}), cfg, source)
    for value in (0, 16, 48, 512):
        with pytest.raises(ValueError, match="Physical microbatch"):
            validate_continuation(state, dict(cfg, physical_microbatch=value), source)
    bad = copy.deepcopy(state)
    bad["ema"]["num_updates"] = 1
    with pytest.raises(ValueError, match="EMA count"):
        validate_continuation(bad, cfg, source)
    with pytest.raises(ValueError, match="checksum"):
        load_comparison_checkpoint(
            dict(cfg, continuation_parent_sha256="wrong"), tmp_path / "child", source
        )


def test_resume_remains_strict_and_does_not_revalidate_parent_file(tmp_path):
    state, cfg, source, _ = parent_fixture(tmp_path)
    output = tmp_path / "child"
    output.mkdir()
    cfg = dict(cfg, resume=True)
    torch.save(
        dict(state, config=cfg, continuation={"parent_step": 2}), output / "latest.pt"
    )
    # A requeued child uses its own checkpoint, even if the parent was moved.
    (tmp_path / "parent.pt").unlink()
    loaded, fresh, receipt = load_comparison_checkpoint(cfg, output, source)
    assert not fresh and loaded["step"] == 2 and receipt == {"parent_step": 2}
    with pytest.raises(ValueError, match="changed on resume"):
        load_comparison_checkpoint(dict(cfg, log_every=100), output, source)
    with pytest.raises(FileExistsError):
        load_comparison_checkpoint(dict(cfg, resume=False), output, source)
