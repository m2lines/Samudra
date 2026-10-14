# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from train_multitask import MultitaskTrainer
from train_score import ScoreTrainer

from samudra.utils.ema import EMATracker
from samudra.utils.schedule import CosineWithWarmupConfig


@pytest.mark.parametrize("finetune", [True, False])
def test_reset_only_when_starting_finetune(monkeypatch, finetune):
    cfg = SimpleNamespace(
        reset_ema_on_finetune=True,
        ema_decay=0.99,
        faster_decay_at_start=False,
        finetune=finetune,
    )
    model = torch.nn.Linear(2, 1)
    old = EMATracker(model)
    old.num_updates = torch.tensor(6160)

    def initialize(self, config):
        # ScoreTrainer assigns cfg after base construction/checkpoint loading.
        assert not hasattr(self, "cfg")
        self.model = model
        self._ema = old
        with torch.no_grad():
            self.model.weight.fill_(3)
        self.cfg = config

    monkeypatch.setattr(ScoreTrainer, "__init__", initialize)
    trainer = MultitaskTrainer(cfg)
    if finetune:
        assert trainer._ema is not old
        assert trainer._ema.num_updates == 0
        assert float(trainer._ema.decay) == pytest.approx(0.99)
        assert not trainer._ema._faster_decay_at_start
        torch.testing.assert_close(
            trainer._ema._ema_params["weight"], trainer.model.weight
        )
    else:
        assert trainer._ema is old
        assert trainer._ema.num_updates == 6160


def test_flat_twenty_then_cosine_ten():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    scheduler = CosineWithWarmupConfig(warmup_lr=1e-4, warmup_epochs=20).build(
        optimizer, 30
    )
    rates = []
    for _ in range(30):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[:20] == pytest.approx([1e-4] * 20)
    assert rates[21] < rates[20]
    assert all(b <= a for a, b in zip(rates, rates[1:]))
    assert optimizer.param_groups[0]["lr"] == 0
