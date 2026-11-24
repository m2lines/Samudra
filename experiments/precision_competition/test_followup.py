# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""EMA, exact-state forks, schedule, and validation-only selection invariants."""

import json
from dataclasses import replace

import pytest
import torch

from .followup import FollowupConfig, losses, scheduled_lr, train
from .followup_suite import checkpoint_digest, select_recipe


def tiny(**kwargs):
    return FollowupConfig(
        width=16,
        depth=1,
        heads=2,
        tokens=64,
        batch=4,
        steps=4,
        eval_every=2,
        validation_size=8,
        test_size=8,
        **kwargs,
    )


def test_schedule_floor_and_original_constant():
    c = tiny(schedule="cosine")
    assert scheduled_lr(c, 1) == c.lr / 100
    assert scheduled_lr(c, 100) == c.lr
    assert scheduled_lr(c, 3000) == c.lr_end
    assert scheduled_lr(c, 12000) == c.lr_end
    assert scheduled_lr(replace(c, schedule="constant"), 12000) == c.lr


def test_loss_scaling_is_explicit():
    c = tiny(main_loss_scale=4.0, aux_loss_scale=0.25)
    y = torch.ones(8)
    main, aux = losses(y, y, {"main": y * 0, "auxiliary": y * 0}, c)
    assert main.item() == 0.25 and aux.item() == 4.0


def test_ema_resume_matches_uninterrupted_and_does_not_change_raw(tmp_path):
    c = tiny()
    a, b, raw = (tmp_path / name for name in ("a", "b", "raw"))
    train(c, a, "cpu")
    train(c, b, "cpu", stop_after=2)
    train(c, b, "cpu")
    train(replace(c, ema_decays=[]), raw, "cpu")
    left = torch.load(a / "latest.pt", weights_only=False)
    right = torch.load(b / "latest.pt", weights_only=False)
    noema = torch.load(raw / "latest.pt", weights_only=False)
    for key in left["model"]:
        torch.testing.assert_close(
            left["model"][key], right["model"][key], atol=0, rtol=0
        )
        torch.testing.assert_close(
            left["model"][key], noema["model"][key], atol=0, rtol=0
        )
        for decay in left["ema"]:
            torch.testing.assert_close(
                left["ema"][decay][key], right["ema"][decay][key], atol=0, rtol=0
            )
    assert left["history"][-1]["ema"] == right["history"][-1]["ema"]


def test_fork_preserves_full_state_and_hash_guard(tmp_path):
    c = tiny()
    train(c, tmp_path / "parent", "cpu", stop_after=2)
    path = tmp_path / "parent" / "latest.pt"
    fork = replace(
        c,
        stage="competition",
        steps=0,
        parent_checkpoint=str(path),
        parent_sha256=checkpoint_digest(path),
    )
    train(fork, tmp_path / "fork", "cpu")
    parent = torch.load(path, weights_only=False)
    child = torch.load(tmp_path / "fork" / "latest.pt", weights_only=False)
    assert child["step"] == 0
    for key in parent["model"]:
        torch.testing.assert_close(
            parent["model"][key], child["model"][key], atol=0, rtol=0
        )
    for key in ("data_rng", "torch_rng"):
        torch.testing.assert_close(parent[key], child[key], atol=0, rtol=0)
    for index, values in parent["optimizer"]["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(
                value, child["optimizer"]["state"][index][name], atol=0, rtol=0
            )
    with pytest.raises(ValueError, match="hash mismatch"):
        train(replace(fork, parent_sha256="wrong"), tmp_path / "bad", "cpu")


def test_validation_selection_ignores_test_and_ema():
    from dataclasses import asdict

    records = []
    for lr, bits in ((3e-4, 5.0), (3e-5, 8.0)):
        for seed in (11, 22, 33):
            c = tiny(lr=lr, seed=seed)
            value = {"config": asdict(c), "test": {"bits": 100 - bits}}
            history = [{"bits": bits, "ema": {"0.99": {"bits": 100 - bits}}}] * 3
            records.append((value, history, None))
    chosen = select_recipe(records)
    assert all(row[1]["config"]["lr"] == 3e-5 for row in chosen)


def test_qualification_manifest_count():
    from pathlib import Path

    entries = json.loads(
        Path(__file__).with_name("followup-qualification.json").read_text()
    )
    assert len(entries) == 36
    assert len({e["id"] for e in entries}) == 36
