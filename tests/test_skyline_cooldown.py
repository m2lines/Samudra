# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Schedule endpoints and full-state retention across restart boundaries."""

from typing import Any

import pytest
import torch

from samudra.skyline.branching import scheduled_learning_rate
from samudra.skyline.run import retain_checkpoint, save_atomic


@pytest.mark.parametrize("peak", [2e-4, 4e-4, 8e-4])
def test_warmup_plateau_and_common_cooldown_endpoint(peak):
    config = dict(
        learning_rate_schedule="warmup_stable_cosine",
        learning_rate=peak,
        warmup=100,
        cooldown_start=25000,
        schedule_updates=30000,
        max_updates=4,
        final_learning_rate=2e-5,
    )
    assert scheduled_learning_rate(config, 1) == peak / 100
    assert scheduled_learning_rate(config, 100) == peak
    assert scheduled_learning_rate(config, 101) == peak
    assert scheduled_learning_rate(config, 25000) == peak
    assert scheduled_learning_rate(config, 27500) == pytest.approx((peak + 2e-5) / 2)
    assert scheduled_learning_rate(config, 30000) == 2e-5
    assert scheduled_learning_rate(config, 25001) < peak
    for step in (0, 30001):
        with pytest.raises(ValueError):
            scheduled_learning_rate(config, step)


def test_full_checkpoint_survives_latest_replacement_and_repairs_interrupted_link(
    tmp_path,
):
    latest = tmp_path / "latest.pt"
    original: dict[str, Any] = dict(
        step=10000,
        model=torch.tensor([1.0]),
        optimizer={"moment": 0.25},
        rng=torch.get_rng_state(),
        ema={"updates": 10000},
    )
    save_atomic(original, latest)
    # A resumed process can create the missing link after the latest save.
    retain_checkpoint(latest, 10000, [10000, 25000, 30000])
    retain_checkpoint(latest, 10000, [10000, 25000, 30000])
    save_atomic(dict(step=10001, model=torch.tensor([2.0])), latest)
    saved = torch.load(tmp_path / "checkpoint-0010000.pt", weights_only=False)
    assert saved["step"] == 10000
    assert saved["optimizer"] == original["optimizer"]
    assert saved["ema"] == original["ema"]
    assert torch.equal(saved["rng"], original["rng"])
    assert torch.equal(saved["model"], original["model"])
    retain_checkpoint(latest, 10001, [10000, 25000, 30000])
    assert not (tmp_path / "checkpoint-0010001.pt").exists()


def test_optimizer_restart_across_cooldown_matches_uninterrupted(tmp_path):
    config = dict(
        learning_rate_schedule="warmup_stable_cosine",
        learning_rate=0.02,
        warmup=1,
        cooldown_start=3,
        max_updates=5,
        final_learning_rate=0.002,
    )
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)

    def update(step):
        optimizer.param_groups[0]["lr"] = scheduled_learning_rate(config, step)
        optimizer.zero_grad()
        model(torch.randn(4, 2)).square().mean().backward()
        optimizer.step()

    for step in range(1, 4):
        update(step)
    path = tmp_path / "latest.pt"
    save_atomic(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=torch.get_rng_state(),
            step=3,
        ),
        path,
    )
    for step in (4, 5):
        update(step)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    saved = torch.load(path, weights_only=False)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"])
    for step in range(saved["step"] + 1, 6):
        update(step)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])
