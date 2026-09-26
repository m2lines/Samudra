# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from samudra.experiments.diffusion_fit import batch_at, bounded_fit


def test_interrupted_fit_resumes_exact_optimizer_rng_and_data_order(tmp_path):
    x = torch.arange(12, dtype=torch.float32).reshape(6, 2) / 12
    y = x.sum(1, keepdim=True)

    def build(seed):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(2, 4), nn.Dropout(0.3), nn.Linear(4, 1))
        return model, torch.optim.AdamW(model.parameters(), lr=0.01)

    def run(model, optimizer, directory, limit=None, signature="paired"):
        return bounded_fit(
            model,
            optimizer,
            lambda step: (
                (model(x[batch_at(step, 6, 2, 7)]) - y[batch_at(step, 6, 2, 7)])
                .square()
                .mean()
            ),
            lambda step: (model(x) - y).square().mean(),
            directory,
            signature,
            max_updates=9,
            max_seconds=600,
            checkpoint_every=2,
            validate_every=3,
            max_new_updates=limit,
        )

    whole, whole_opt = build(11)
    whole_state = run(whole, whole_opt, tmp_path / "whole")
    partial, partial_opt = build(11)
    state = run(partial, partial_opt, tmp_path / "resume", limit=4)
    assert state["step"] == 4 and not state["complete"]
    restored, restored_opt = build(99)
    state = run(restored, restored_opt, tmp_path / "resume")
    assert state["step"] == 9 and state["complete"]
    assert state["best"] == whole_state["best"]
    for name, tensor in whole.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
    for index, values in whole_opt.state_dict()["state"].items():
        for name, tensor in values.items():
            torch.testing.assert_close(
                tensor, restored_opt.state_dict()["state"][index][name], rtol=0, atol=0
            )
    with pytest.raises(ValueError, match="protocol"):
        run(restored, restored_opt, tmp_path / "resume", signature="different")


def test_paired_schedule_covers_each_full_batch_once_per_epoch():
    first = [i for step in range(3) for i in batch_at(step, 6, 2, 7)]
    second = [i for step in range(3, 6) for i in batch_at(step, 6, 2, 7)]
    assert sorted(first) == sorted(second) == list(range(6))
    assert first != second
