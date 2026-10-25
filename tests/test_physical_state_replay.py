# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Timestamp alignment and persistent physical members in replay."""

import copy
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import torch

from samudra.experiments.physical_state_replay import (
    PhysicalReplayPool,
    continue_members,
    native_replay_objective,
    states_from_predictions,
)


def states(length=8):
    return [
        torch.full((2, 1, 2, 1, 1, 1), float(i), requires_grad=True)
        for i in range(length)
    ]


def reader():
    first = datetime(2000, 1, 1)
    bounds = np.array(
        [
            [first + timedelta(days=i * 5), first + timedelta(days=(i + 1) * 5)]
            for i in range(30)
        ],
        dtype=object,
    )
    return SimpleNamespace(
        windows=[(19, 25, None, 30), (24, 30, None, 30)], bounds=bounds
    )


def test_observation_pool_caches_overlapping_bin_not_month_endpoint_and_expires():
    pool = PhysicalReplayPool("observation", max_age=5)
    pool.add_observation("2000-01", "2000-01", states(), 10)
    entry = pool.draw(10)
    assert entry["bin_index"] == 6  # February 1 lies inside Jan31-Feb5 bin.
    assert entry["state"].mean() == 6
    assert not entry["state"].requires_grad
    clone = PhysicalReplayPool("observation", max_age=5)
    clone.load_state_dict(copy.deepcopy(pool.state_dict()))
    assert clone.draw(15)["month"] == "2000-02"
    assert clone.draw(16) is None


def test_native_pool_selects_matching_history_and_rejects_calendar_gap():
    data = reader()
    data.windows = [
        (np.int64(start), np.int64(stop), weights, duration)
        for start, stop, weights, duration in data.windows
    ]
    pool = PhysicalReplayPool("om4")
    pool.add_native(data, 0, states(), 3)
    entry = pool.draw(3)
    assert entry["window_index"] == 1 and entry["start_index"] == 24
    assert entry["state"].mean() == 5
    assert json.loads(json.dumps({"lead_days": entry["lead_days"]})) == {
        "lead_days": 25
    }
    data.bounds[26, 0] += timedelta(days=1)
    bad = PhysicalReplayPool("om4")
    bad.add_native(data, 0, states(), 3)
    assert bad.draw(3) is None


def test_member_history_shift_and_no_reinitialization():
    initial = states(1)[0].detach()
    initial[1] = 10
    prediction = torch.stack([initial[:, :, -1] + i for i in (1, 2, 3)], 2)
    history = states_from_predictions(initial, prediction)
    torch.testing.assert_close(history[-1][:, :, 0], prediction[:, :, -2])

    class Model:
        def advance(self, member, forcing, context, mask, generator):
            return torch.stack((member[:, -1], member[:, -1] + 1), 1)

    result = continue_members(
        Model(),
        initial,
        torch.zeros(1, 3, 8, 1, 1),
        torch.zeros(1, 3, 5, 1, 1),
        torch.ones(1, 1, 1),
        torch.Generator(),
    )
    assert result[-1][0, 0, -1] == 3
    assert result[-1][1, 0, -1] == 13


def test_native_replay_uses_sampled_history_not_gold_as_condition():
    model = SimpleNamespace(history=2)
    parameter = torch.tensor(1.0, requires_grad=True)
    observed = []

    def transition(member, forcing, context, target, mask, weights, generator):
        observed.extend(member[:, -1].flatten().tolist())
        assert bool((target == 100).all())
        return parameter.square()

    model.transition_loss = transition
    model.advance = lambda member, *args: torch.stack(
        (member[:, -1], member[:, -1] + 1), 1
    )
    batch = dict(
        fields=torch.full((1, 4, 1, 1, 1), 100.0),
        mask=torch.ones(1, 1, 1),
        contexts=torch.zeros(1, 4, 5, 1, 1),
    )
    source = SimpleNamespace(
        windows=[(19, 21, None, None)], sample=lambda index, device: batch
    )
    cache = SimpleNamespace(load=lambda *args: torch.zeros(1, 4, 8, 1, 1))
    entry = dict(
        source="om4",
        window_index=0,
        origin_index=0,
        start_index=19,
        state=torch.zeros(2, 1, 2, 1, 1, 1),
    )
    loss, output = native_replay_objective(
        model,
        SimpleNamespace(device="cpu", area=torch.ones(1, 1)),
        source,
        cache,
        entry,
        torch.Generator(),
    )
    loss.backward()
    assert parameter.grad == 2
    assert observed == [0, 0, 1, 1]
    assert output[-1].flatten().tolist() == [1, 2, 1, 2]
    assert not output[-1].requires_grad
